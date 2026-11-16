"""Gegenprobe der Optimalitätsbedingung des Achsenabschnitts und des Linearprogramms an Zufallsinstanzen (ORACLE-PLAYBOOK §6): größer als die Fassung in `tests/`.

Je Instanz: n = 5 bis 60, p = 2 bis 4 Spalten (Achsenabschnitt und ganzzahlige Merkmale), ganzzahlige y (erzwingen Gleichstände und Residuen gleich 0), Stufe aus
{0,1; 0,25; 0,5; 0,8; 0,9; 0,95}. Geprüft wird (1) #{r < 0} <= n·τ <= #{r <= 0} für das Residuum r = y − X·b des Linearprogramms (eigene Herleitung aus dem Teilgradienten) und
(2) bei p = 2 und verschiedenen x: der Verlust des Linearprogramms gleich dem kleinsten Verlust über alle Geraden durch zwei Datenpunkte (Aufzählung).

Aufruf (Repo-Ordner, Python des Test-venv):  python tools/que_gegenprobe.py [Zahl der Instanzen]   (Standard 300; die README nennt diese Messung)."""

import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import que_algorithm as A  # noqa: E402
import que_constants as C  # noqa: E402


def run(n_inst: int = 300, seed: int = 1501) -> dict:
    rng = np.random.default_rng(seed)           # nur Prüfdaten, der Prüfling benutzt SplitMix64
    viol = tied = enum_checked = enum_bad = 0
    for _ in range(n_inst):
        n = int(rng.integers(5, 61))
        p = int(rng.integers(2, 5))
        tau = float(rng.choice([0.1, 0.25, 0.5, 0.8, 0.9, 0.95]))
        X = np.column_stack([np.ones(n)] + [rng.integers(0, 6, n).astype(float) for _ in range(p - 1)])
        y = np.round(rng.normal(size=n) * 3 + X[:, 1] * 0.5).astype(float)
        b, obj = A.qr_fit(X, y, tau)
        r = y - X @ b
        neg, nonpos = int(np.sum(r < -C.RES_ZERO)), int(np.sum(r <= C.RES_ZERO))
        if not (neg <= n * tau + 1e-9 and nonpos >= n * tau - 1e-9):
            viol += 1
        if nonpos > neg:
            tied += 1
        if p == 2 and len(set(X[:, 1].tolist())) >= 2 and n <= 40:
            x = X[:, 1]
            best = np.inf
            for i, j in itertools.combinations(range(n), 2):
                if x[i] == x[j]:
                    continue
                sl = (y[j] - y[i]) / (x[j] - x[i])
                ic = y[i] - sl * x[i]
                rr = y - ic - sl * x
                best = min(best, float(np.sum(np.where(rr >= 0, tau * rr, (tau - 1) * rr))))
            enum_checked += 1
            if abs(obj - best) > 1e-9:
                enum_bad += 1
    return {"instances": n_inst, "violations": viol, "with_zero_residual": tied, "enum_checked": enum_checked, "enum_bad": enum_bad}


if __name__ == "__main__":
    n_inst = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    t0 = time.time()
    res = run(n_inst)
    print(f"{res['instances']} Instanzen, Verletzungen der Optimalitätsbedingung: {res['violations']}; Instanzen mit mindestens einem Punkt auf der Geraden: {res['with_zero_residual']}")
    print(f"Aufzählung aller Geraden durch zwei Punkte: {res['enum_checked']} Instanzen verglichen, Abweichungen des Verlusts: {res['enum_bad']}")
    print(f"GESAMT: {res['instances'] + res['enum_checked']} Prüfungen, {res['violations'] + res['enum_bad']} Fehler; Laufzeit {time.time() - t0:.1f} s")
