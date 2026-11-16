"""Welten: Container-Generator (Nachbau mit exaktem Zufallsstrom), Kundeneffekte, Entwurfsmatrizen, Testmenge, Gitter und Gruppen von Rohskala A und B."""

import numpy as np
import pytest

import que_constants as C
import que_scenario as S


def test_customer_effects_equal_numpy_default_rng_of_the_neighbour_demo():
    """Die fest eingetragenen Kundeneffekte sind die von `vwz_generator.kunden_effekte`: numpy.random.default_rng(12345).normal(0, 0,3, 12)."""
    ref = np.random.default_rng(C.UK_SEED).normal(0.0, C.UK_SD, C.K_KUNDEN)
    assert np.array(C.UK) == pytest.approx(ref, abs=1e-12)


def test_log_median_and_s_by_hand_for_each_art():
    """Handrechnung je Art (Kunde 0 mit Effekt −0,42714751…, Kundeneffekt je Art: 1 / 0,3 / 0,3 / 1)."""
    f = S.Features(np.array([0, 0, 1, 2, 3, 3]), np.zeros(6, int), np.array([0, 1, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 1]), np.array([0.0, 0.0, 4.0, 2.0, 0.0, 0.0]))
    mu, s = S.log_median_and_s(f)
    u0 = C.UK[0]
    expected = [
        np.log(3.5) + u0,                       # Import ohne Zollhalt, Werktag: Median 3,5
        np.log(3.5 * np.exp(0.7)) + u0,         # Import mit Zollhalt
        np.log(0.6 + 0.85 * 4.0) + 0.3 * u0,    # Export, Vorlauf 4: Median 4,0
        np.log(0.3 + 0.8 * 2.0) + 0.3 * u0,     # Umschlag, Vorlauf 2: Median 1,9
        np.log(7.0) + u0,                       # Leer, Werktag
        np.log(7.0 * np.exp(0.3)) + u0,         # Leer, Wochenende
    ]
    assert mu == pytest.approx(expected, abs=1e-12)
    assert s == pytest.approx([0.55, 0.55, 0.25, 0.35, 0.90, 0.90])


def test_holdout_set_has_the_documented_shares_and_size():
    f, mu, s = S.holdout_set()
    assert f.n == C.TEST_N == 20_000
    shares = [float(np.mean(f.art == a)) for a in range(4)]
    assert shares == pytest.approx(C.ART_P, abs=0.01)                              # Anteile 35 / 30 / 25 / 10 % (Toleranz 1 Prozentpunkt bei 20 000)
    assert float(np.mean(f.we)) == pytest.approx(2 / 7, abs=0.01)
    assert set(np.unique(f.zoll[f.art != 0])) <= {0}                               # Zollhalt nur bei Import
    assert float(np.mean(f.zoll[f.art == 0])) == pytest.approx(C.P_ZOLL, abs=0.02)
    assert min(len(i) for i in S.cell_groups(f)) > 400                             # jede der acht Zellen Art × Wochenende ist gefüllt
    assert S.holdout_set()[0] is f                                                 # die Testmenge wird nur einmal gerechnet


def test_vorlauf_ranges_by_art():
    f = S.draw_features(5000, 3)
    assert float(f.vorlauf[f.art == 1].min()) >= 1.0 and float(f.vorlauf[f.art == 1].max()) <= 10.0
    assert float(f.vorlauf[f.art == 2].min()) >= 0.5 and float(f.vorlauf[f.art == 2].max()) <= 6.0
    assert np.all(f.vorlauf[(f.art == 0) | (f.art == 3)] == 0.0)


def test_design_matrices_by_hand():
    f = S.Features(np.array([0, 1, 2, 3]), np.zeros(4, int), np.array([1, 0, 0, 0]), np.array([1, 1, 0, 1]), np.array([0.0, 5.0, 3.0, 0.0]))
    X8 = S.design_c8(f)
    assert X8.shape == (4, 8)
    assert X8[0].tolist() == [1, 0, 0, 0, 1, 1, 0, 0]       # Import mit Zollhalt am Wochenende
    assert X8[1].tolist() == [1, 1, 0, 0, 0, 1, 5, 0]       # Export, Vorlauf 5
    assert X8[2].tolist() == [1, 0, 1, 0, 0, 0, 0, 3]       # Umschlag, Vorlauf 3
    X11 = S.design_c11(f)
    assert X11.shape == (4, 11)
    assert X11[1, 8:].tolist() == [1, 0, 0] and X11[3, 8:].tolist() == [0, 0, 1] and X11[0, 8:].tolist() == [0, 0, 0]
    assert S.design(C.WELT_C8, f).shape[1] == C.N_PARAM[C.WELT_C8] and S.design(C.WELT_C11, f).shape[1] == C.N_PARAM[C.WELT_C11]
    with pytest.raises(ValueError):
        S.design(C.WELT_A, f)


def test_world_ab_grid_groups_and_functions_by_hand():
    xg = S.grid_ab()
    assert len(xg) == 20 and xg[0] == 0.25 and xg[-1] == 9.75
    assert [len(g) for g in S.groups_ab()] == [7, 6, 7]
    assert sorted(np.concatenate(S.groups_ab()).tolist()) == list(range(20))
    assert float(S.mu_ab(3.0)) == 7.0
    assert float(S.sd_ab(C.WELT_A, 10.0)) == pytest.approx(4.2) and float(S.sd_ab(C.WELT_B, 10.0)) == pytest.approx(4.2)
    assert float(S.sd_ab(C.WELT_B, 5.0)) == pytest.approx(1.2) and float(S.sd_ab(C.WELT_A, 0.0)) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        S.sd_ab(C.WELT_C8, 1.0)


def test_draw_ab_moments_and_prefix_property():
    X, y, x = S.draw_ab(C.WELT_A, 20000, 7)
    assert float(x.min()) >= 0 and float(x.max()) <= 10 and float(x.mean()) == pytest.approx(5.0, abs=0.1)
    resid = (y - S.mu_ab(x)) / S.sd_ab(C.WELT_A, x)
    assert float(resid.mean()) == pytest.approx(0.0, abs=0.03) and float(resid.std()) == pytest.approx(1.0, abs=0.03)
    X2, y2, x2 = S.draw_ab(C.WELT_A, 50, 7)
    assert np.array_equal(x2, x[:50]) and np.array_equal(y2, y[:50])      # die ersten Beobachtungen hängen nicht von n ab
    with pytest.raises(ValueError):
        S.draw_ab(C.WELT_A, 0, 1)


def test_repetition_seeds_are_distinct_and_container_models_share_samples():
    seeds = {S.rep_seed(w, n, i) for w in (C.WELT_A, C.WELT_B, C.WELT_C8) for n in (30, 300) for i in range(50)}
    assert len(seeds) == 3 * 2 * 50
    assert S.rep_seed(C.WELT_C8, 300, 5) == S.rep_seed(C.WELT_C11, 300, 5)
    f8 = S.draw_history(300, S.rep_seed(C.WELT_C8, 300, 5))[0]
    f11 = S.draw_history(300, S.rep_seed(C.WELT_C11, 300, 5))[0]
    assert np.array_equal(f8.art, f11.art)
