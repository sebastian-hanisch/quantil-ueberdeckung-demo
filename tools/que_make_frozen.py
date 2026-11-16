"""Rechnet die vorgerechnete Messung der Demo neu und schreibt sie nach `que_frozen.json` (die App liest nur diese Datei).

Aufruf (Repo-Ordner, Python des Test-venv):
    python tools/que_make_frozen.py                 alles neu rechnen und die Datei schreiben (einige Minuten, ein Prozess)
    python tools/que_make_frozen.py --only a        nur die Zellen einer Welt (a, b, c8, c11) neu rechnen und in die Datei einsetzen
    python tools/que_make_frozen.py --check         Struktur der Datei gegen die Konstanten prüfen und zwei kleine Zellen vollständig neu rechnen (Band)

Zähleinheit der Messung: die Wiederholung (eine frische Trainingsstichprobe). Die Überdeckung ist exakt aus der wahren bedingten Verteilungsfunktion gerechnet (siehe
que_evaluation.py). Zufallsquelle: der ganzzahlige SplitMix64 (`que_rng`), Zufalls-Seeds aus `que_scenario.rep_seed`."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import que_constants as C  # noqa: E402
import que_evaluation as ev  # noqa: E402

TARGET = ROOT / C.FROZEN_FILE
VERSION = 1
SPOT_CHECKS = (("a", 30, 0.9), ("c8", 300, 0.9))     # vollständig neu gerechnete Zellen der Prüfung (schnell)
SPOT_TOL = 0.003                                     # Band: LP-Ecken und Gleitkomma-Summen unterscheiden sich je Plattform in der letzten Stelle, nicht in der dritten


def _flatten(cell: dict) -> dict:
    """Alle Zahlen einer Zelle mit Pfad, zum Vergleich."""
    flat = {}
    for m, e in cell["methods"].items():
        for k, v in e.items():
            if isinstance(v, list):
                for i, x in enumerate(v):
                    flat[f"{m}.{k}[{i}]"] = x
            else:
                flat[f"{m}.{k}"] = v
    flat["cross"] = cell["cross"]
    flat["reps"] = cell["reps"]
    return flat


def build(only=None, existing=None) -> dict:
    data = existing or {"meta": {}, "cells": {}}
    t_all = time.time()
    for welt in C.WELTEN:
        if welt == C.WELT_HAND or (only and welt != only):
            continue
        for n in C.N_OPTIONS[welt]:
            t0 = time.time()
            data["cells"].update(ev.compute_cell(welt, n, C.TAUS))
            print(f"  {welt} n = {n}: {C.REPS[(welt, n)]} Wiederholungen, {len(C.TAUS)} Stufen, {time.time() - t0:.0f} s", flush=True)
    data["meta"] = {
        "version": VERSION,
        "taus": list(C.TAUS),
        "n_options": {w: list(v) for w, v in C.N_OPTIONS.items()},
        "reps": {f"{w}|{n}": r for (w, n), r in C.REPS.items()},
        "seed_base": C.SEED_BASE,
        "gap_under": C.UNDER_GAP, "gap_group_under": C.GROUP_UNDER_GAP,
        "text": "Zähleinheit: Wiederholung (eine frische Trainingsstichprobe). Überdeckung exakt aus der wahren bedingten Verteilungsfunktion. Erzeugt von tools/que_make_frozen.py.",
    }
    print(f"fertig in {time.time() - t_all:.0f} s", flush=True)
    return data


def structure_problems(data: dict) -> list[str]:
    """Stimmen Raster, Zahl der Wiederholungen und Verfahren der Datei mit den Konstanten überein?"""
    problems = []
    cells = data.get("cells", {})
    expected = {}
    for welt in C.WELTEN:
        if welt == C.WELT_HAND:
            continue
        for n in C.N_OPTIONS[welt]:
            for t in C.TAUS:
                expected[ev.cell_key(welt, n, t)] = (welt, n)
    for key in sorted(set(expected) - set(cells)):
        problems.append(f"Zelle fehlt: {key}")
    for key in sorted(set(cells) - set(expected)):
        problems.append(f"Zelle ohne Konstante: {key}")
    for key, (welt, n) in expected.items():
        cell = cells.get(key)
        if cell is None:
            continue
        if cell["reps"] != C.REPS[(welt, n)]:
            problems.append(f"{key}: {cell['reps']} Wiederholungen statt {C.REPS[(welt, n)]}")
        if set(cell["methods"]) != set(C.METHODS_BY_WELT[welt]):
            problems.append(f"{key}: Verfahren {sorted(cell['methods'])} statt {sorted(C.METHODS_BY_WELT[welt])}")
    return problems


def spot_problems(data: dict) -> list[str]:
    """Zwei kleine Zellen vollständig neu rechnen und mit der Datei vergleichen (Band SPOT_TOL)."""
    problems = []
    for welt, n, tau in SPOT_CHECKS:
        new = ev.compute_cell(welt, n, [tau])[ev.cell_key(welt, n, tau)]
        old = data["cells"][ev.cell_key(welt, n, tau)]
        fn, fo = _flatten(new), _flatten(old)
        for k in fo:
            a, b = fn[k], fo[k]
            if a is None or b is None:
                if a != b:
                    problems.append(f"{welt}|{n}|{tau} {k}: {a} gegen {b}")
            elif abs(a - b) > SPOT_TOL and k != "reps":
                problems.append(f"{welt}|{n}|{tau} {k}: neu {a} gegen Datei {b}")
    return problems


def check() -> list[str]:
    if not TARGET.exists():
        return [f"{TARGET.name} fehlt"]
    data = json.loads(TARGET.read_text(encoding="utf-8"))
    return structure_problems(data) + spot_problems(data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", choices=[w for w in C.WELTEN if w != C.WELT_HAND])
    args = ap.parse_args(argv)
    if args.check:
        problems = check()
        print("\n".join(problems) if problems else "que_frozen.json passt zu den Konstanten und zur neu gerechneten Stichprobe")
        return 1 if problems else 0
    existing = json.loads(TARGET.read_text(encoding="utf-8")) if (args.only and TARGET.exists()) else None
    data = build(args.only, existing)
    TARGET.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("geschrieben:", TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
