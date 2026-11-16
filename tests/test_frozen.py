"""Die vorgerechnete Messung `que_frozen.json` passt zur Suite: Raster, Wiederholungen und Verfahren stimmen mit den Konstanten überein, zwei kleine Zellen
werden vollständig neu gerechnet und als Band verglichen (`tools/que_make_frozen.py --check`), die Kennzahlen sind in sich stimmig."""

import json
import sys
from pathlib import Path

import pytest

import que_constants as C
import que_evaluation as ev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import que_make_frozen as MF  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return json.loads((ROOT / C.FROZEN_FILE).read_text(encoding="utf-8"))


def test_grid_reps_and_methods_match_the_constants(data):
    assert MF.structure_problems(data) == []


def test_two_small_cells_recompute_within_the_band(data):
    assert MF.spot_problems(data) == []


def test_every_cell_is_consistent_in_itself(data):
    for key, cell in data["cells"].items():
        welt, n, tau = key.split("|")
        tau = float(tau)
        for m, e in cell["methods"].items():
            assert all(0.0 <= v <= 1.0 for v in e["groups"] + [e["total"], e["p5"], e["p50"], e["p95"], e["min_group"], e["share_under"], e["share_group_under"]]), (key, m)
            assert e["p5"] <= e["p50"] <= e["p95"], (key, m)
            assert e["min_group"] <= min(e["groups"]) + 1e-4, (key, m)                          # Mittel des kleinsten Werts je Wiederholung <= kleinstes Gruppenmittel
            assert e["min_group"] <= e["total"] + 1e-4
            assert len(e["groups"]) == (4 if welt in ("c8", "c11") else 3)
            if welt in ("c8", "c11"):
                assert len(e["cells"]) == 8 and e["max_cell_dev"] == pytest.approx(max(abs(c - tau) for c in e["cells"]), abs=2e-4)
        if welt == C.WELT_A and tau > 0.5:
            assert 0.0 <= cell["cross"] <= 1.0
        else:
            assert cell["cross"] is None


def test_frozen_cell_lookup_and_missing_cells(data):
    c = ev.frozen_cell(C.WELT_A, 200, 0.9)
    assert c is not None and c["reps"] == 400
    assert ev.frozen_cell(C.WELT_HAND, 10, 0.8) is None
    assert ev.frozen_cell(C.WELT_A, 77, 0.9) is None
    assert ev.cell_key("a", 200, 0.9) == "a|200|0.90"


def test_one_cell_matches_a_direct_recomputation_of_a_few_repetitions():
    """aggregate() gegen eine eigene Mittelung: 6 Wiederholungen, Rohskala A, n = 50, τ = 0,9."""
    import numpy as np
    from que_scenario import rep_seed

    res = [ev.fit_cover(C.WELT_A, 50, rep_seed(C.WELT_A, 50, i), 0.9, methods=("ols1", "qr"), cross=True) for i in range(6)]
    agg = ev.aggregate(res, 0.9, ("ols1", "qr"))
    g = np.array([r["methods"]["qr"]["groups"] for r in res])
    t = np.array([r["methods"]["qr"]["total"] for r in res])
    assert agg["methods"]["qr"]["groups"] == pytest.approx(g.mean(axis=0), abs=1e-4)
    assert agg["methods"]["qr"]["total"] == pytest.approx(float(t.mean()), abs=1e-4)
    assert agg["methods"]["qr"]["share_under"] == pytest.approx(float(np.mean(t < 0.9 - C.UNDER_GAP)), abs=6e-5)
    assert agg["methods"]["qr"]["min_group"] == pytest.approx(float(g.min(axis=1).mean()), abs=1e-4)
    assert agg["cross"] == pytest.approx(float(np.mean([r["cross"] for r in res])), abs=6e-5)
    assert agg["reps"] == 6
