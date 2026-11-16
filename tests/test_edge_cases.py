"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = p, n < p, Stufen nahe 1, leere Gruppen, Bänder bei oder unter 0, konstante Zielgröße,
doppelte x, Kreuzen der Quantilgeraden, ungültige Stufen."""

import math

import numpy as np
import pytest

import que_algorithm as A
import que_constants as C
import que_evaluation as ev


def test_n_equals_p_fits_exactly():
    """n = p = 2: die Gerade geht durch beide Punkte, beide Residuen sind 0 (die Optimalitätsbedingung gilt mit #{r < 0} = 0 und #{r <= 0} = 2)."""
    X = np.array([[1.0, 0.0], [1.0, 1.0]])
    y = np.array([1.0, 4.0])
    b, obj = A.qr_fit(X, y, 0.9)
    assert b == pytest.approx([1.0, 3.0]) and obj == pytest.approx(0.0, abs=1e-9)


def test_n_equals_p_plus_one_keeps_the_optimality_condition():
    rng = np.random.default_rng(1)
    for _ in range(20):
        X = np.column_stack([np.ones(3), rng.uniform(0, 5, 3)])
        y = rng.normal(size=3)
        tau = float(rng.choice([0.2, 0.5, 0.8]))
        b, _ = A.qr_fit(X, y, tau)
        r = y - X @ b
        assert int(np.sum(r < -C.RES_ZERO)) <= 3 * tau + 1e-9 <= int(np.sum(r <= C.RES_ZERO)) + 1e-9


def test_fewer_observations_than_parameters_is_refused_with_a_message():
    with pytest.raises(ValueError, match="n = 1 < p = 2"):
        A.qr_fit(np.array([[1.0, 2.0]]), np.array([3.0]), 0.5)
    with pytest.raises(ValueError, match="Freiheitsgrade"):
        A.sigma_hat(np.array([1.0, -1.0]), 2)


@pytest.mark.parametrize("tau", [0.0, 1.0, -0.1, 1.5])
def test_levels_outside_zero_one_are_refused(tau):
    with pytest.raises(ValueError):
        A.z_of(tau)
    with pytest.raises(ValueError):
        A.qr_fit(np.ones((3, 1)), np.array([1.0, 2.0, 3.0]), tau)


def test_high_level_with_few_points_covers_everything_in_training():
    """n = 5, tau = 0,95: n·(1 − tau) = 0,25 < 1: alle fünf Punkte liegen auf oder unter der Geraden (nur eine Konstante: der größte Wert)."""
    y = np.array([3.0, 1.0, 4.0, 1.5, 9.0])
    b, _ = A.qr_fit(np.ones((5, 1)), y, 0.95)
    assert b[0] == pytest.approx(9.0) and int(np.sum(y <= b[0] + 1e-9)) == 5


def test_all_y_equal_gives_that_constant_and_a_zero_variance_band():
    X = np.column_stack([np.ones(6), np.arange(6.0)])
    y = np.full(6, 2.5)
    b, r = A.ols_fit(X, y)
    assert A.sigma_hat(r, 2) == pytest.approx(0.0, abs=1e-12)
    bq, obj = A.qr_fit(X, y, 0.9)
    assert obj == pytest.approx(0.0, abs=1e-9) and (X @ bq) == pytest.approx(y)


def test_duplicate_x_values_do_not_break_the_fit():
    X = np.column_stack([np.ones(6), np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])])
    y = np.array([1.0, 2.0, 2.0, 4.0, 3.0, 7.0])
    b, obj = A.qr_fit(X, y, 0.5)
    assert A.pinball(y - X @ b, 0.5) == pytest.approx(obj, abs=1e-9)


def test_group_without_observation_is_nan_and_never_zero():
    out = A.group_means(np.array([0.9, 0.8]), [np.array([0, 1]), np.array([], dtype=int)])
    assert math.isnan(out[1]) and out[0] == pytest.approx(0.85)
    assert ev.de(float("nan")) == "–"        # die Anzeige zeigt einen Strich, nicht 0,000


def test_raw_band_at_or_below_zero_covers_nothing_instead_of_log_of_zero():
    c = A.cover_lognormal_raw(np.array([-5.0, 0.0]), np.zeros(2), np.ones(2))
    assert list(c) == [0.0, 0.0]
    # im echten Lauf: τ = 0,1 und ein Band auf der Rohskala deckt bei Export und Umschlag nichts (die Untergrenze liegt unter 0)
    rec = ev.fit_cover(C.WELT_C8, 3000, 3, 0.1, methods=("ols1_roh",))["methods"]["ols1_roh"]
    assert rec["groups"][1] < 1e-3 and rec["groups"][2] < 1e-3          # kaum ein Export- oder Umschlag-Container hat ein Band über 0


def test_crossing_is_reported_and_never_sorted_away():
    low, high = np.array([0.0, 1.0]), np.array([1.0, 0.5])         # die Gerade der niedrigen Stufe überholt bei x = 10: 10 > 6
    assert A.crossing(low, high, (0.0, 10.0)) is True
    assert A.crossing(np.array([0.0, 0.4]), np.array([1.0, 0.5]), (0.0, 10.0)) is False


def test_settings_are_clipped_to_the_nearest_offered_value():
    s = ev.sanitize("a", 0.93, 120, 5)
    assert s.tau == 0.95 and s.n == 100                              # 0,93 liegt näher an 0,95 als an 0,9; 120 näher an 100 als an 200
    s = ev.sanitize("unbekannt", 0.9, 10 ** 6, -4)
    assert s.welt == C.DEFAULT_WELT and s.n == 1000 and s.seed == 0


def test_smallest_and_largest_n_run_in_every_world_without_error():
    for welt in (C.WELT_A, C.WELT_B, C.WELT_C8, C.WELT_C11):
        for n in (min(C.N_OPTIONS[welt]), max(C.N_OPTIONS[welt])):
            for tau in (min(C.TAUS), max(C.TAUS)):
                r = ev.fit_cover(welt, n, 1, tau)
                assert all(0.0 <= v <= 1.0 for rec in r["methods"].values() for v in rec["groups"] + [rec["total"]])


def test_hand_analysis_at_the_extreme_levels():
    for tau in (0.01, 0.99):
        h = A.hand_analysis(C.HAND_G0, C.HAND_G1, tau)
        assert all(0 <= c <= 5 for c in h["ols_cov"])
        assert all(q["cov_lo"] >= 1 for q in h["qr"])                 # mindestens der kleinste Wert liegt auf oder unter dem Quantil
