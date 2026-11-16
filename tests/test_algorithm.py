"""Kern gegen Handrechnung (Brüche) auf Mini-Instanzen: Band „Mittel + z·s“, Pinball-Verlust, Quantilregression (LP), Überdeckung, Handbeispiel.

Gerechnet wird in `fractions`, wo es exakt geht; nur Normalquantile z und Φ sind Gleitkomma. Der Pinball-Verlust ρ_τ(r) = τ·r für r >= 0 und (τ − 1)·r für r < 0 hat bei
ganzzahligem n·τ eine Strecke von Minimierern: die Tests vergleichen darum den Verlust (und die Streckenenden), nicht die Koeffizienten des Linearprogramms."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import que_algorithm as A
import que_constants as C


# --- Handbeispiel: zwei Gruppen, je fünf Beobachtungen, tau = 4/5 ---------------------------------------------------------------------------------

def test_hand_means_sse_and_variance(hand_groups):
    g0, g1 = hand_groups
    m0, m1 = sum(g0) / 5, sum(g1) / 5
    assert (m0, m1) == (4, 8)
    sse = sum((v - m0) ** 2 for v in g0) + sum((v - m1) ** 2 for v in g1)
    assert sse == 10 + 160                       # 4 + 1 + 0 + 1 + 4 und 64 + 16 + 0 + 16 + 64
    assert sse / (10 - 2) == F(85, 4)            # n − p = 8 Freiheitsgrade
    h = A.hand_analysis(C.HAND_G0, C.HAND_G1, 0.8)
    assert h["means"] == [4, 8] and h["sse"] == 170 and h["s2"] == F(85, 4)


def test_hand_band_covers_five_of_five_and_three_of_five():
    h = A.hand_analysis(C.HAND_G0, C.HAND_G1, 0.8)
    assert h["z"] == pytest.approx(0.841621, abs=1e-6)
    assert h["s"] == pytest.approx(math.sqrt(21.25), abs=1e-12)
    assert h["upper"] == pytest.approx([4 + 3.8797, 8 + 3.8797], abs=1e-3)       # 7,88 und 11,88
    assert h["ols_cov"] == [5, 3]                # Gruppe 0: alle fünf (z·s >= 2), Gruppe 1: 0, 4, 8 (die 12 liegt darüber)
    # exakt über Quadrate, ohne Wurzel: z·s >= 2 <=> z² >= 16/85 und z·s < 4 <=> z² < 64/85
    z2 = F(h["z"]) ** 2
    assert z2 >= F(16, 85) and z2 < F(64, 85)
    assert sum(h["ols_cov"]) == 8 and F(sum(h["ols_cov"]), 10) == F(4, 5)        # im Mittel 0,8, je Gruppe 1,0 und 0,6


def test_hand_pinball_minimum_per_group_and_flat_segments(hand_groups):
    g0, g1 = hand_groups
    t = F(4, 5)
    cand0 = sorted(set(g0) | {F(11, 2)})
    cand1 = sorted(set(g1))
    assert min(A.pinball_exact(g0, c, t) for c in cand0) == 2
    assert min(A.pinball_exact(g1, c, t) for c in cand1) == 8
    assert A.pinball_exact(g0, F(5), t) == A.pinball_exact(g0, F(6), t) == A.pinball_exact(g0, F(11, 2), t) == 2      # flach auf [5, 6]
    assert A.pinball_exact(g1, F(12), t) == A.pinball_exact(g1, F(16), t) == 8                                          # flach auf [12, 16]
    assert A.pinball_exact(g0, F(4), t) > 2 and A.pinball_exact(g1, F(8), t) > 8
    assert A.quantile_interval(g0, t) == (5, 6) and A.quantile_interval(g1, t) == (12, 16)


def test_hand_quantile_regression_coverage_is_a_staircase():
    h = A.hand_analysis(C.HAND_G0, C.HAND_G1, 0.8)
    assert [q["cov_lo"] for q in h["qr"]] == [4, 4]          # untere Streckenenden (5 und 12): 4 von 5 in jeder Gruppe
    assert [q["cov_hi"] for q in h["qr"]] == [5, 5]          # obere Enden (6 und 16): 5 von 5 (Treppe mit Stufen 1/5)
    assert [q["flat"] for q in h["qr"]] == [True, True]
    assert h["qr_loss"] == 10                                  # Summe 2 + 8


def test_hand_lp_reaches_the_pinball_sum_ten_with_a_group_dummy():
    X = np.column_stack([np.ones(10), np.r_[np.zeros(5), np.ones(5)]])
    y = np.array(C.HAND_G0 + C.HAND_G1, float)
    b, obj = A.qr_fit(X, y, 0.8)
    assert obj == pytest.approx(10.0, abs=1e-9)
    assert A.pinball(y - X @ b, 0.8) == pytest.approx(10.0, abs=1e-9)
    # welches Ende der Strecke der Löser liefert, bleibt offen: Gruppenwerte liegen jedenfalls auf den Strecken [5, 6] und [12, 16]
    assert 5 - 1e-9 <= b[0] <= 6 + 1e-9 and 12 - 1e-9 <= b[0] + b[1] <= 16 + 1e-9


@pytest.mark.parametrize("tau,expected", [(0.1, [0, 1]), (0.5, [3, 3]), (0.9, [5, 4]), (0.99, [5, 5])])
def test_hand_counts_for_other_levels(tau, expected):
    """Von Hand mit s = 4,61: tau 0,1 (z = −1,28, z·s = −5,91): Grenzen −1,91 und 2,09, gedeckt 0 und nur die 0 in Gruppe 1. tau 0,5 (z = 0): Grenzen 4 und 8, gedeckt 2, 3, 4
    und 0, 4, 8. tau 0,9 (z·s = 5,91): Grenzen 9,91 und 13,91, gedeckt alle fünf und 0, 4, 8, 12. tau 0,99 (z·s = 10,72): Grenzen 14,7 und 18,7, alles."""
    assert A.hand_analysis(C.HAND_G0, C.HAND_G1, tau)["ols_cov"] == expected


# --- Sieben Punkte: Aufzählung aller Geraden durch zwei Punkte --------------------------------------------------------------------------------

X7 = [F(v) for v in (1, 2, 3, 4, 5, 6, 7)]
Y7 = [F(1), F(7, 2), F(2), F(6), F(4), F(9), F(11, 2)]


def _enum_min(t):
    """Kleinster Pinball-Verlust über alle Geraden durch zwei der sieben Punkte (in Brüchen)."""
    best = None
    for i in range(7):
        for j in range(i + 1, 7):
            sl = (Y7[j] - Y7[i]) / (X7[j] - X7[i])
            ic = Y7[i] - sl * X7[i]
            loss = sum((t * (y - ic - sl * x) if y - ic - sl * x >= 0 else (t - 1) * (y - ic - sl * x)) for x, y in zip(X7, Y7))
            if best is None or loss < best:
                best = loss
    return best


@pytest.mark.parametrize("t,exact", [(F(1, 4), F(41, 16)), (F(1, 2), F(37, 8)), (F(9, 10), F(51, 40))])
def test_seven_points_lp_matches_enumeration_in_fractions(t, exact):
    assert _enum_min(t) == exact
    X = np.column_stack([np.ones(7), [float(v) for v in X7]])
    y = np.array([float(v) for v in Y7])
    b, obj = A.qr_fit(X, y, float(t))
    assert obj == pytest.approx(float(exact), abs=1e-9)
    assert A.pinball(y - X @ b, float(t)) == pytest.approx(float(exact), abs=1e-9)


def test_seven_points_high_level_covers_everything_in_training():
    """n = 7, tau = 0,9: n·(1 − tau) = 0,7 < 1, also liegen alle sieben Punkte auf oder unter der Geraden (2 genau darauf): im Training Überdeckung 1."""
    X = np.column_stack([np.ones(7), [float(v) for v in X7]])
    y = np.array([float(v) for v in Y7])
    b, _ = A.qr_fit(X, y, 0.9)
    r = y - X @ b
    assert int(np.sum(r <= C.RES_ZERO)) == 7 and int(np.sum(np.abs(r) <= C.RES_ZERO)) == 2


# --- Nur eine Konstante, n = 8 ---------------------------------------------------------------------------------------------------------------

YY = [F(3), F(1), F(4), F(3, 2), F(9), F(2), F(6), F(5)]


@pytest.mark.parametrize("t,lo,hi,loss", [(F(1, 4), F(3, 2), F(2), F(43, 8)), (F(1, 2), F(3), F(4), F(33, 4)), (F(3, 4), F(5), F(6), F(57, 8))])
def test_constant_only_minimum_is_a_segment(t, lo, hi, loss):
    assert A.quantile_interval(YY, t) == (lo, hi)
    for c in (lo, (lo + hi) / 2, hi):
        assert A.pinball_exact(YY, c, t) == loss
    assert A.pinball_exact(YY, lo - F(1, 10), t) > loss and A.pinball_exact(YY, hi + F(1, 10), t) > loss
    assert sum(1 for v in YY if v <= lo) == 8 * t
    X = np.ones((8, 1))
    _, obj = A.qr_fit(X, np.array([float(v) for v in YY]), float(t))
    assert obj == pytest.approx(float(loss), abs=1e-9)


def test_quantile_interval_is_a_point_if_n_tau_is_not_an_integer():
    assert A.quantile_interval([F(v) for v in (1, 2, 3, 4)], F(1, 3)) == (2, 2)       # n·tau = 4/3 -> x_(2)
    assert A.quantile_interval([F(7)], F(1, 2)) == (7, 7)


# --- Welt A: Grenzwert von „Vorhersage ± z·s“ ----------------------------------------------------------------------------------------------------

def test_limit_variance_of_world_a_is_463_over_75():
    s2 = F(4, 100) + F(16, 100) * 5 + F(16, 100) * F(100, 3)      # E[(0,2 + 0,4 x)²] mit E[x] = 5, E[x²] = 100/3 bei x ~ Gleich(0, 10)
    assert s2 == F(463, 75) and float(s2) == pytest.approx(6.1733333, abs=1e-7)


def test_limit_coverage_at_the_edge_of_world_a():
    """Stelle x = 10 (sd 4,2), tau = 0,9: Φ(1,2816 · 2,4846/4,2) = Φ(0,758) = 0,776."""
    c = A.ols_limit_coverage_a(0.9, np.array([10.0]), lambda x: 0.2 + 0.4 * np.asarray(x))
    assert float(c[0]) == pytest.approx(0.776, abs=5e-4)


# --- Einheiten ----------------------------------------------------------------------------------------------------------------------------------

def test_pinball_by_hand():
    assert A.pinball([2.0, -1.0], 0.8) == pytest.approx(0.8 * 2 + (0.8 - 1) * (-1))      # 1,6 + 0,2
    assert A.pinball([], 0.5) == 0.0


def test_cover_normal_by_hand():
    assert float(A.cover_normal(10.0, 10.0, 3.0)) == pytest.approx(0.5)
    assert float(A.cover_normal(13.0, 10.0, 3.0)) == pytest.approx(0.8413447, abs=1e-6)      # Φ(1)


def test_cover_lognormal_raw_is_zero_for_nonpositive_bands():
    out = A.cover_lognormal_raw(np.array([-1.0, 0.0, 1e-13, 4.0]), np.log(4.0) * np.ones(4), np.ones(4))
    assert out[0] == out[1] == out[2] == 0.0 and out[3] == pytest.approx(0.5)


def test_group_means_marks_empty_groups_as_nan():
    out = A.group_means(np.array([0.2, 0.4, 0.6]), [np.array([0, 1]), np.array([], dtype=int), np.array([2])])
    assert out[0] == pytest.approx(0.3) and out[2] == pytest.approx(0.6) and math.isnan(out[1])


def test_ols_on_a_line_and_sigma():
    X = np.column_stack([np.ones(5), np.arange(5.0)])
    b, r = A.ols_fit(X, 1 + 2 * np.arange(5.0))
    assert b == pytest.approx([1.0, 2.0]) and np.allclose(r, 0)
    b, r = A.ols_fit(X, np.array([1.0, 3.0, 5.0, 7.0, 10.0]))
    assert A.sigma_hat(r, 2) == pytest.approx(math.sqrt(float(np.sum(r ** 2)) / 3))


def test_sigma_by_group_uses_the_mean_square_and_a_fallback():
    r = np.array([1.0, -1.0, 2.0, -2.0, 3.0])
    out = A.sigma_by_group(r, np.array([0, 0, 1, 1, 2]), 3, fallback=9.0)
    assert out[0] == pytest.approx(1.0) and out[1] == pytest.approx(2.0) and out[2] == 9.0          # Gruppe 2 hat nur eine Beobachtung


def test_z_of_values():
    assert A.z_of(0.5) == 0.0
    assert A.z_of(0.9) == pytest.approx(1.2815516, abs=1e-6)
    assert A.z_of(0.1) == pytest.approx(-A.z_of(0.9))
