"""Kern der Demo (nur numpy, scipy und Standardbibliothek): Kleinste Quadrate mit Band „Vorhersage ± z·s“, lineare Quantilregression als Linearprogramm,
exakte Überdeckung aus der wahren bedingten Verteilungsfunktion, Gruppenmittel, Kreuzen der Quantilgeraden und das Handbeispiel in Brüchen.

Konvention: Quantilstufe `tau`, z = Φ⁻¹(tau). Pinball-Verlust ρ_τ(r) = τ·r für r >= 0 und (τ − 1)·r für r < 0 mit Residuum r = y − q (Standard-Prüffunktion, hier nur als
Definition benutzt). **Überdeckung** eines angepassten Quantilwertes q̂(x) ist P(y <= q̂(x) | x) unter der wahren bedingten Verteilung, gemittelt über die Punkte einer Gruppe;
Soll = tau. Zwei Verfahren: (1) Kleinste Quadrate, s² = SSE/(n − p), Band ŷ + z·s; (2) lineare Quantilregression, min Σ τ·u + (1 − τ)·v mit X·b + u − v = y, u, v >= 0
(ein Linearprogramm, HiGHS über die dünne Matrix). Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b)."""

import math
from fractions import Fraction
from statistics import NormalDist

import numpy as np
from scipy import optimize, sparse
from scipy.special import ndtr


def z_of(tau: float) -> float:
    """Standardnormalquantil z_τ = Φ⁻¹(τ)."""
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau = {tau} muss in (0, 1) liegen")
    return NormalDist().inv_cdf(tau)


# --- Anpassung ---------------------------------------------------------------------------------------------------------------------------------

def ols_fit(X: np.ndarray, y: np.ndarray):
    """Kleinste Quadrate: Koeffizienten und Residuen y − X·b (numpy.linalg.lstsq; bei nicht eindeutiger Lösung die kürzeste)."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b, y - X @ b


def sigma_hat(resid: np.ndarray, p: int) -> float:
    """s = sqrt(SSE/(n − p)): die eine gemeinsame Reststreuung. Braucht n > p."""
    n = len(resid)
    if n <= p:
        raise ValueError(f"n = {n} <= p = {p}: keine Freiheitsgrade für s")
    return float(math.sqrt(float(np.sum(resid ** 2)) / (n - p)))


def sigma_by_group(resid: np.ndarray, group_of: np.ndarray, k: int, fallback: float) -> np.ndarray:
    """Eine Reststreuung je Gruppe: Wurzel des mittleren Residuenquadrats (Teiler: Zahl der Beobachtungen der Gruppe); Gruppen mit weniger als zwei Beobachtungen bekommen `fallback`."""
    out = np.full(k, float(fallback))
    for g in range(k):
        m = group_of == g
        if int(np.sum(m)) >= 2:
            out[g] = float(math.sqrt(float(np.mean(resid[m] ** 2))))
    return out


def pinball(r, tau: float) -> float:
    """Summe des Pinball-Verlusts über die Residuen r = y − q."""
    r = np.asarray(r, float)
    return float(np.sum(np.where(r >= 0, tau * r, (tau - 1.0) * r)))


def qr_fit(X: np.ndarray, y: np.ndarray, tau: float):
    """Lineare Quantilregression als Linearprogramm; gibt (Koeffizienten, Zielwert) zurück. Bei ganzzahligem n·tau ist das Minimum bei Konstanten und Gruppenmitteln eine Strecke (mit stetigen Merkmalen in der Regel eindeutig): welches Ende der Löser
    liefert, ist eine Eigenschaft des Lösers (darum vergleichen Tests den Verlust, nicht die Koeffizienten). Braucht n >= p."""
    n, p = X.shape
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau = {tau} muss in (0, 1) liegen")
    if n < p:
        raise ValueError(f"n = {n} < p = {p}: die Quantilgerade ist nicht eindeutig")
    c = np.concatenate([np.zeros(p), tau * np.ones(n), (1.0 - tau) * np.ones(n)])
    A = sparse.hstack([sparse.csr_matrix(X), sparse.identity(n), -sparse.identity(n)], format="csr")
    res = optimize.linprog(c, A_eq=A, b_eq=np.asarray(y, float), bounds=[(None, None)] * p + [(0, None)] * (2 * n), method="highs")
    if res.status != 0:
        raise RuntimeError(f"Linearprogramm nicht gelöst: {res.message}")
    return res.x[:p], float(res.fun)


# --- Überdeckung (exakt aus der wahren bedingten Verteilung) ---------------------------------------------------------------------------------

def cover_normal(q, mu, sd) -> np.ndarray:
    """P(y <= q | x) bei y ~ Normal(mu, sd²)."""
    return ndtr((np.asarray(q, float) - np.asarray(mu, float)) / np.asarray(sd, float))


def cover_lognormal_log(logq, mu, s) -> np.ndarray:
    """P(T <= exp(logq)) bei log T ~ Normal(mu, s²): das Band liegt auf der Log-Skala."""
    return ndtr((np.asarray(logq, float) - np.asarray(mu, float)) / np.asarray(s, float))


def cover_lognormal_raw(q, mu, s) -> np.ndarray:
    """P(T <= q) bei log T ~ Normal(mu, s²) und einem Band q in Tagen; ein Band bei oder unter 0 deckt nichts (Überdeckung 0, nicht log(0))."""
    q = np.asarray(q, float)
    out = np.zeros(q.shape)
    pos = q > 1e-12
    out[pos] = ndtr((np.log(q[pos]) - np.asarray(mu, float)[pos]) / np.asarray(s, float)[pos])
    return out


def group_means(c: np.ndarray, groups) -> list[float]:
    """Mittel der Überdeckung je Gruppe; eine Gruppe ohne Punkt gibt nan zurück (nie 0: Überdeckung ist dort nicht definiert)."""
    return [float(np.mean(c[g])) if len(g) else float("nan") for g in groups]


def crossing(b_low: np.ndarray, b_high: np.ndarray, xs, tol: float = 1e-9) -> bool:
    """Liegt die Gerade der niedrigen Stufe an einer der Stellen xs über der der hohen Stufe? (Koeffizienten (Achsenabschnitt, Steigung).)"""
    xs = np.asarray(xs, float)
    return bool(np.any(b_low[0] + b_low[1] * xs > b_high[0] + b_high[1] * xs + tol))


def ols_limit_coverage_a(tau: float, xs: np.ndarray, sd_of_x) -> np.ndarray:
    """Grenzwert (n → ∞) der Überdeckung von „Vorhersage ± z·s“ bei richtigem Mittel und der gemeinsamen Reststreuung s² = 463/75 (Rohskala A):
    an der Stelle x deckt das Band mit Φ(z_τ·s/sd(x)). Eigene Herleitung: s² ist der Erwartungswert von sd(x)² über x ~ Gleich(0, 10)."""
    s = math.sqrt(463.0 / 75.0)
    return ndtr(z_of(tau) * s / np.asarray(sd_of_x(xs), float))


# --- Handbeispiel in Brüchen ---------------------------------------------------------------------------------------------------------------------

def quantile_interval(vals, tau: Fraction):
    """Menge der Minimierer des Pinball-Verlusts für eine Konstante: (unteres, oberes) Ende. Bei ganzzahligem n·τ eine Strecke [x_(nτ), x_(nτ+1)], sonst der Punkt x_(⌈nτ⌉)."""
    if not 0 < tau < 1:
        raise ValueError("tau muss in (0, 1) liegen")
    v = sorted(vals)
    n = len(v)
    if n == 0:
        raise ValueError("keine Beobachtung")
    k = n * tau
    if k.denominator == 1:
        return v[int(k) - 1], v[int(k)]
    j = math.ceil(k)
    return v[j - 1], v[j - 1]


def pinball_exact(vals, c, tau: Fraction):
    """Pinball-Verlust in Brüchen für die Konstante c."""
    return sum((tau * (v - c) if v >= c else (tau - 1) * (v - c)) for v in vals)


def hand_analysis(g0, g1, tau: float) -> dict:
    """Zwei Gruppen in Brüchen: Gruppenmittel, SSE, s² = SSE/(n − 2), das Band „Mittel + z·s“ mit EINEM s, wie viele Punkte es je Gruppe deckt (im Training),
    und die Quantilregression mit Gruppenvariable (je Gruppe der Pinball-Minimierer): untere und obere Streckenenden, gedeckte Punkte an beiden Enden, Verlust.
    Gedeckt heißt y <= Grenze. `tau` wird als Bruch mit Nenner höchstens 1000 gelesen."""
    t = Fraction(tau).limit_denominator(1000)
    groups = [[Fraction(v) for v in g0], [Fraction(v) for v in g1]]
    means = [sum(g) / len(g) for g in groups]
    sse = sum((v - m) ** 2 for g, m in zip(groups, means) for v in g)
    n = sum(len(g) for g in groups)
    s2 = sse / (n - 2)
    z = z_of(float(t))
    zs = z * math.sqrt(float(s2))
    ub = [float(m) + zs for m in means]
    ols_cov = [sum(1 for v in g if float(v) <= u) for g, u in zip(groups, ub)]
    qr = []
    for g in groups:
        lo, hi = quantile_interval(g, t)
        qr.append({
            "lo": lo, "hi": hi, "flat": lo != hi,
            "cov_lo": sum(1 for v in g if v <= lo), "cov_hi": sum(1 for v in g if v <= hi),
            "loss": pinball_exact(g, lo, t),
        })
    return {
        "tau": t, "z": z, "means": means, "sse": sse, "s2": s2, "s": math.sqrt(float(s2)), "zs": zs, "upper": ub,
        "n_group": [len(g) for g in groups], "ols_cov": ols_cov, "qr": qr, "qr_loss": sum(q["loss"] for q in qr),
    }
