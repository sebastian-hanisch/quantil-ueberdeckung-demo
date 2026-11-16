"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für das Verfahren gelten müssen, ohne dass man das richtige Ergebnis kennt.

Je Eigenschaft 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed (Prüfdaten aus numpy; der Prüfling benutzt den SplitMix64). Jede Eigenschaft ist eine belegte Aussage über das
Verfahren: Verschiebung und Skalierung der Zielgröße, Zeilenvertauschung, Äquivarianz der Überdeckung gegen Verschieben von x, z = 0 beim Median, Überdeckung der wahren
Quantilgeraden gleich τ, Monotonie der Überdeckung in τ (Mittel über Wiederholungen), Optimalitätsbedingung auch mit Gleichständen. Verluste mit Toleranz 1e-9."""

import numpy as np
import pytest

import que_algorithm as A
import que_constants as C
import que_scenario as S


def _instance(rng, n=None, p=2):
    n = int(rng.integers(8, 40)) if n is None else n
    x = rng.uniform(0, 10, n)
    X = np.column_stack([np.ones(n)] + [x] + [rng.uniform(0, 5, n) for _ in range(p - 2)])
    y = 1 + 2 * x + (0.2 + 0.4 * x) * rng.normal(size=n)
    return X, y


def test_shifting_all_y_shifts_the_intercept_and_keeps_the_loss():
    rng = np.random.default_rng(11)
    for _ in range(25):
        X, y = _instance(rng)
        tau = float(rng.choice([0.1, 0.5, 0.9]))
        c = float(rng.uniform(-50, 50))
        _, l0 = A.qr_fit(X, y, tau)
        b1, l1 = A.qr_fit(X, y + c, tau)
        assert l1 == pytest.approx(l0, abs=1e-8)
        assert A.pinball(y + c - X @ b1, tau) == pytest.approx(l0, abs=1e-8)


def test_scaling_y_by_a_positive_factor_scales_the_loss():
    rng = np.random.default_rng(12)
    for _ in range(25):
        X, y = _instance(rng)
        tau = float(rng.choice([0.1, 0.5, 0.9]))
        lam = float(rng.uniform(0.2, 7.0))
        _, l0 = A.qr_fit(X, y, tau)
        _, l1 = A.qr_fit(X, lam * y, tau)
        assert l1 == pytest.approx(lam * l0, rel=1e-8)


def test_permuting_the_rows_does_not_change_the_loss():
    rng = np.random.default_rng(13)
    for _ in range(25):
        X, y = _instance(rng)
        perm = rng.permutation(len(y))
        tau = float(rng.choice([0.25, 0.8, 0.95]))
        assert A.qr_fit(X[perm], y[perm], tau)[1] == pytest.approx(A.qr_fit(X, y, tau)[1], abs=1e-9)


def test_median_band_is_ols_because_z_is_zero():
    """tau = 0,5: z = 0, also ist das Band „Vorhersage ± z·s“ die Vorhersage selbst, unabhängig von s."""
    assert A.z_of(0.5) == 0.0
    rng = np.random.default_rng(14)
    for _ in range(20):
        X, y = _instance(rng)
        b, r = A.ols_fit(X, y)
        assert np.array_equal(X @ b + A.z_of(0.5) * A.sigma_hat(r, 2), X @ b)


def test_shifting_x_leaves_the_coverage_unchanged_for_both_methods():
    """Äquivarianz: x → x + 5 (Anpassung neu, Auswertung an den verschobenen Gitterpunkten) ändert weder das Band noch die Überdeckung."""
    xg = S.grid_ab()
    for seed in range(20):
        X, y, x = S.draw_ab(C.WELT_A, 60, seed)
        Xs = np.column_stack([np.ones(60), x + 5.0])
        for tau in (0.5, 0.9):
            z = A.z_of(tau)
            b, r = A.ols_fit(X, y)
            bs, rs = A.ols_fit(Xs, y)
            s, ss = A.sigma_hat(r, 2), A.sigma_hat(rs, 2)
            assert ss == pytest.approx(s, rel=1e-9)
            q1 = b[0] + b[1] * xg + z * s
            q2 = bs[0] + bs[1] * (xg + 5.0) + z * ss
            assert q2 == pytest.approx(q1, abs=1e-8)
            bq, lq = A.qr_fit(X, y, tau)
            bqs, lqs = A.qr_fit(Xs, y, tau)
            assert lqs == pytest.approx(lq, abs=1e-8)


def test_coverage_of_the_true_quantile_line_is_tau_in_every_group():
    xg = S.grid_ab()
    for welt in (C.WELT_A, C.WELT_B):
        for tau in (0.1, 0.5, 0.9, 0.99):
            q = S.mu_ab(xg) + A.z_of(tau) * S.sd_ab(welt, xg)
            c = A.cover_normal(q, S.mu_ab(xg), S.sd_ab(welt, xg))
            assert np.allclose(c, tau, atol=1e-12)
            assert np.allclose(A.group_means(c, S.groups_ab()), tau, atol=1e-12)


def test_coverage_is_nondecreasing_in_tau_on_average():
    """Mittel über 30 Wiederholungen je Stufe: die Gesamtüberdeckung wächst mit τ (beide Verfahren, Rohskala A, n = 100)."""
    taus = (0.1, 0.5, 0.9, 0.99)
    xg = S.grid_ab()
    mu, sd = S.mu_ab(xg), S.sd_ab(C.WELT_A, xg)
    means = {"ols": [], "qr": []}
    for tau in taus:
        o, q = [], []
        for rep in range(30):
            X, y, _ = S.draw_ab(C.WELT_A, 100, 500 + rep)
            b, r = A.ols_fit(X, y)
            s = A.sigma_hat(r, 2)
            o.append(float(np.mean(A.cover_normal(b[0] + b[1] * xg + A.z_of(tau) * s, mu, sd))))
            bq, _ = A.qr_fit(X, y, tau)
            q.append(float(np.mean(A.cover_normal(bq[0] + bq[1] * xg, mu, sd))))
        means["ols"].append(np.mean(o))
        means["qr"].append(np.mean(q))
    for k in ("ols", "qr"):
        assert all(a < b for a, b in zip(means[k], means[k][1:])), (k, means[k])


def test_optimality_condition_holds_with_integer_y_and_ties():
    rng = np.random.default_rng(15)
    for _ in range(50):
        n = int(rng.integers(5, 30))
        X = np.column_stack([np.ones(n), rng.integers(0, 4, n).astype(float)])
        y = np.round(rng.normal(size=n) * 2).astype(float)
        tau = float(rng.choice([0.1, 0.25, 0.5, 0.8, 0.9]))
        b, _ = A.qr_fit(X, y, tau)
        r = y - X @ b
        assert int(np.sum(r < -C.RES_ZERO)) <= n * tau + 1e-9 <= int(np.sum(r <= C.RES_ZERO)) + 2e-9


def test_same_zufalls_seed_gives_the_same_sample_and_different_seeds_differ():
    a = S.draw_ab(C.WELT_A, 50, 4)
    b = S.draw_ab(C.WELT_A, 50, 4)
    c = S.draw_ab(C.WELT_A, 50, 5)
    assert np.array_equal(a[1], b[1]) and not np.array_equal(a[1], c[1])
    fa, _, _, da = S.draw_history(100, 4)
    fb, _, _, db = S.draw_history(100, 4)
    assert np.array_equal(da, db) and np.array_equal(fa.art, fb.art)


def test_first_containers_do_not_depend_on_n():
    """Die ersten k Container einer Stichprobe hängen nicht von n ab (Zeile für Zeile aus einem Strom): keine Zukunftsinformation, Wiederholungen bleiben vergleichbar."""
    f30 = S.draw_features(30, 9)
    f300 = S.draw_features(300, 9)
    assert np.array_equal(f30.art, f300.art[:30]) and np.array_equal(f30.vorlauf, f300.vorlauf[:30]) and np.array_equal(f30.we, f300.we[:30])
