"""Orakel-Tests: Kern und Auswertung gegen einen anderen Rechenweg (ORACLE-PLAYBOOK §5, DEMO-PLAYBOOK §12).

* Linearprogramm der Quantilregression gegen die Aufzählung aller Geraden durch zwei Datenpunkte (der Minimierer liegt an einer Ecke) und gegen scikit-learn
  (`QuantileRegressor`, nur Test; `pytest.importorskip`). **Verglichen wird der Verlust, nicht die Koeffizienten**: bei ganzzahligem n·τ ist das Minimum eine Strecke,
  und Löser wählen verschiedene Enden.
* Überdeckungsformel gegen Monte Carlo (Rohskala A, 400 000 Ziehungen, vier Standardfehler) und gegen die realisierten Verweilzeiten des Generators je Art.
* Grenzwert Φ(z·s/sd) von „Vorhersage ± z·s“ gegen Quadratur (s² = 463/75) und gegen eine Anpassung mit n = 100 000.
* Die Auswertung `fit_cover` gegen einen unabhängigen Weg (Normalgleichung statt lstsq, scipy.stats.norm statt ndtr, eigene Gruppenzählung).
* Optimalitätsbedingung des Achsenabschnitts #{r < 0} <= n·τ <= #{r <= 0} an Zufallsinstanzen mit Gleichständen.
Plattform: Verluste mit Toleranz 1e-9, Zählungen exakt, Monte-Carlo-Abstände als Vielfache des Standardfehlers."""

import itertools
import math

import numpy as np
import pytest
from scipy import integrate, stats

import que_algorithm as A
import que_constants as C
import que_evaluation as ev
import que_rng as R
import que_scenario as S


def _enum_line_loss(x, y, tau):
    """Kleinster Pinball-Verlust über alle Geraden durch zwei Datenpunkte mit verschiedenem x (Gleitkomma, anderer Weg als das Linearprogramm)."""
    best = np.inf
    for i, j in itertools.combinations(range(len(x)), 2):
        if x[i] == x[j]:
            continue
        sl = (y[j] - y[i]) / (x[j] - x[i])
        ic = y[i] - sl * x[i]
        r = y - ic - sl * x
        best = min(best, float(np.sum(np.where(r >= 0, tau * r, (tau - 1) * r))))
    return best


def test_lp_loss_equals_the_enumeration_of_lines_through_two_points():
    rng = np.random.default_rng(606)          # nur Prüfdaten; der Prüfling benutzt keinen numpy-Zufall
    for k in range(40):
        n = int(rng.integers(4, 10))
        x = np.sort(rng.choice(np.arange(0, 12), size=n, replace=False)).astype(float)       # verschiedene x, ganzzahlig
        y = np.round(rng.normal(0, 3, n) + 0.7 * x).astype(float)                              # ganzzahlige y: Gleichstände
        for tau in (0.25, 0.5, 0.8, 0.9):
            X = np.column_stack([np.ones(n), x])
            _, obj = A.qr_fit(X, y, tau)
            assert obj == pytest.approx(_enum_line_loss(x, y, tau), abs=1e-9), (k, tau)


def test_lp_loss_equals_scikit_learn():
    sk = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(607)
    for k in range(12):
        n = int(rng.integers(20, 60))
        x = rng.uniform(0, 10, n)
        y = 1 + 2 * x + (0.2 + 0.4 * x) * rng.normal(size=n)
        for tau in (0.1, 0.5, 0.9):
            X = np.column_stack([np.ones(n), x])
            b, obj = A.qr_fit(X, y, tau)
            ref = sk.QuantileRegressor(quantile=tau, alpha=0.0, solver="highs").fit(x.reshape(-1, 1), y)
            loss_ref = A.pinball(y - ref.predict(x.reshape(-1, 1)), tau)
            assert obj == pytest.approx(loss_ref, rel=1e-7, abs=1e-7), (k, tau)


@pytest.mark.parametrize("x0,tau", [(1.0, 0.9), (5.0, 0.9), (9.0, 0.5), (9.0, 0.1)])
def test_coverage_formula_equals_monte_carlo_in_world_a(x0, tau):
    mu, sd = float(S.mu_ab(x0)), float(S.sd_ab(C.WELT_A, x0))
    q = mu + 0.7 * sd * A.z_of(tau) + 0.3 * sd                           # ein Band, das nicht das wahre Quantil ist
    formula = float(A.cover_normal(q, mu, sd))
    n = 400_000
    draws = mu + sd * R.normal_array(2024 + int(10 * x0), n)
    mc = float(np.mean(draws <= q))
    assert abs(formula - mc) <= 4 * math.sqrt(formula * (1 - formula) / n)


def test_true_quantile_covers_tau_in_both_worlds_by_monte_carlo():
    n = 200_000
    z = R.normal_array(77, n)
    for welt in (C.WELT_A, C.WELT_B):
        x = np.array([0.5, 4.0, 9.5])
        mu, sd = S.mu_ab(x), S.sd_ab(welt, x)
        for tau in (0.1, 0.9):
            q = mu + sd * A.z_of(tau)
            assert np.allclose(A.cover_normal(q, mu, sd), tau)             # exakt
            for i in range(3):
                assert abs(float(np.mean(mu[i] + sd[i] * z <= q[i])) - tau) <= 4 * math.sqrt(tau * (1 - tau) / n)


def test_true_quantile_covers_tau_for_every_container_art():
    """Das wahre Quantil exp(mu + s·z_τ) deckt in den realisierten Verweilzeiten der Testmenge je Art τ (Toleranz 4,5 Standardfehler)."""
    f, mu, s = S.holdout_set()
    z = R.normal_array(S.sub_seed(C.TEST_SEED, 3), f.n)
    dwell = np.exp(mu + s * z)
    for tau in (0.1, 0.5, 0.9):
        q = np.exp(mu + s * A.z_of(tau))
        for art in range(4):
            idx = np.where(f.art == art)[0]
            share = float(np.mean(dwell[idx] <= q[idx]))
            assert abs(share - tau) <= 4.5 * math.sqrt(tau * (1 - tau) / len(idx)), (art, tau, share)


def test_limit_variance_equals_quadrature_and_limit_coverage_equals_a_huge_fit():
    quad = integrate.quad(lambda x: (0.2 + 0.4 * x) ** 2 / 10.0, 0, 10)[0]
    assert quad == pytest.approx(463 / 75, abs=1e-9)
    X, y, x = S.draw_ab(C.WELT_A, 100_000, 31)
    b, r = A.ols_fit(X, y)
    s = A.sigma_hat(r, 2)
    assert s == pytest.approx(math.sqrt(463 / 75), abs=0.02)
    xg = S.grid_ab()
    for tau in (0.1, 0.9):
        c = A.cover_normal(b[0] + b[1] * xg + A.z_of(tau) * s, S.mu_ab(xg), S.sd_ab(C.WELT_A, xg))
        lim = A.ols_limit_coverage_a(tau, xg, lambda v: S.sd_ab(C.WELT_A, v))
        for g in S.groups_ab():
            assert float(np.mean(c[g])) == pytest.approx(float(np.mean(lim[g])), abs=0.004)
    # und der gerundete Wert aus den Gruppen: unten / Mitte / oben bei tau = 0,9
    lim = A.ols_limit_coverage_a(0.9, xg, lambda v: S.sd_ab(C.WELT_A, v))
    assert [round(float(np.mean(lim[g])), 3) for g in S.groups_ab()] == [0.996, 0.926, 0.820]


def test_fit_cover_equals_an_independent_computation_in_world_a():
    for seed, n, tau in ((1, 30, 0.9), (2, 100, 0.5), (3, 200, 0.1)):
        X, y, x = S.draw_ab(C.WELT_A, n, seed)
        b = np.linalg.solve(X.T @ X, X.T @ y)                          # Normalgleichung statt lstsq
        s = math.sqrt(float(np.sum((y - X @ b) ** 2)) / (n - 2))
        xg = np.linspace(0.25, 9.75, 20)
        q = b[0] + b[1] * xg + stats.norm.ppf(tau) * s
        cov = stats.norm.cdf((q - (1 + 2 * xg)) / (0.2 + 0.4 * xg))
        got = ev.fit_cover(C.WELT_A, n, seed, tau)["methods"]["ols1"]
        assert got["groups"] == pytest.approx([cov[:7].mean(), cov[7:13].mean(), cov[13:].mean()], abs=1e-9)       # 7 / 6 / 7 Gitterpunkte
        assert got["total"] == pytest.approx(float(cov.mean()), abs=1e-9)


def test_fit_cover_equals_an_independent_computation_for_the_containers():
    seed, n, tau = 5, 600, 0.9
    f, mu_h, s_h, d = S.draw_history(n, seed)
    ft, mu_t, s_t = S.holdout_set()
    # unabhängig: die Entwurfsmatrix Spalte für Spalte, Normalgleichung, eigene Art-Zählung
    X = np.zeros((n, 8))
    X[:, 0] = 1
    X[:, 1], X[:, 2], X[:, 3] = f.art == 1, f.art == 2, f.art == 3
    X[:, 4], X[:, 5] = f.zoll, f.we
    X[:, 6] = np.where(f.art == 1, f.vorlauf, 0.0)
    X[:, 7] = np.where(f.art == 2, f.vorlauf, 0.0)
    Xt = np.zeros((ft.n, 8))
    Xt[:, 0] = 1
    Xt[:, 1], Xt[:, 2], Xt[:, 3] = ft.art == 1, ft.art == 2, ft.art == 3
    Xt[:, 4], Xt[:, 5] = ft.zoll, ft.we
    Xt[:, 6] = np.where(ft.art == 1, ft.vorlauf, 0.0)
    Xt[:, 7] = np.where(ft.art == 2, ft.vorlauf, 0.0)
    ly = np.log(d)
    b = np.linalg.solve(X.T @ X, X.T @ ly)
    r = ly - X @ b
    s1 = math.sqrt(float(r @ r) / (n - 8))
    sa = np.array([math.sqrt(float(np.mean(r[f.art == a] ** 2))) for a in range(4)])
    z = float(stats.norm.ppf(tau))
    got = ev.fit_cover(C.WELT_C8, n, seed, tau)["methods"]
    for name, band in (("ols1", Xt @ b + z * s1), ("olsg", Xt @ b + z * sa[ft.art])):
        cov = stats.norm.cdf((band - mu_t) / s_t)
        assert got[name]["total"] == pytest.approx(float(cov.mean()), abs=1e-9)
        assert got[name]["groups"] == pytest.approx([float(cov[ft.art == a].mean()) for a in range(4)], abs=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_optimality_condition_of_the_intercept_with_ties(seed):
    """#{r < 0} <= n·τ <= #{r <= 0} (eigene Herleitung aus dem Teilgradienten); ganzzahlige y erzwingen Residuum 0 und Gleichstände."""
    rng = np.random.default_rng(900 + seed)
    for _ in range(10):
        n = int(rng.integers(5, 61))
        p = int(rng.integers(1, 4))
        tau = float(rng.choice([0.1, 0.25, 0.5, 0.8, 0.9, 0.95]))
        X = np.column_stack([np.ones(n)] + [rng.integers(0, 6, n).astype(float) for _ in range(p)])
        y = np.round(rng.normal(size=n) * 3 + X[:, 1] * 0.5).astype(float)
        b, _ = A.qr_fit(X, y, tau)
        r = y - X @ b
        assert int(np.sum(r < -C.RES_ZERO)) <= n * tau + 1e-9
        assert int(np.sum(r <= C.RES_ZERO)) >= n * tau - 1e-9
