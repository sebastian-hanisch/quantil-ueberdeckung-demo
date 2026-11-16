"""Claims-Tests: jede Zahl der Befunde B1 bis B11 der README mit ihren Voraussetzungen (DEMO-PLAYBOOK §4, §11a).

Voraussetzung überall: Welt, n, τ und die Zahl der Wiederholungen wie in der README genannt; Zähleinheit Wiederholung (eine frische Trainingsstichprobe).
Quelle der Zahlen: B1, B2, B3 und B11 sind in Brüchen bzw. aus der Rechnung exakt gerechnet; B4 bis B10 stehen in `que_frozen.json` (vorgerechnet; zwei Zellen rechnet
`tests/test_frozen.py` vollständig neu, die übrigen sind **gegen die Datei** geprüft, nicht gegen eine frische Rechnung). Jede README-Zahl wird hier aus den Daten formatiert
(`ev.de`, `ev.pct`) **und** im README-Text gesucht (`_readme_has`)."""

import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

import que_algorithm as A
import que_constants as C
import que_evaluation as ev
import que_scenario as S
from que_evaluation import de, pct

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8").replace("\\|", "|").replace("\\*", "*")


def _has(*tokens):
    for t in tokens:
        assert t in README, t


def _cell(welt, n, tau):
    c = ev.frozen_cell(welt, n, tau)
    assert c is not None, (welt, n, tau)
    return c


def _g(entry):
    """„a / b / c“ der Gruppenwerte wie in der README."""
    return " / ".join(de(v) for v in entry["groups"])


# --- B1 -------------------------------------------------------------------------------------------------------------------------------------

def test_b1_hand_example_exact_numbers():
    g0, g1 = [F(v) for v in C.HAND_G0], [F(v) for v in C.HAND_G1]
    assert (sum(g0) / 5, sum(g1) / 5) == (4, 8)
    assert sum((v - 4) ** 2 for v in g0) + sum((v - 8) ** 2 for v in g1) == 170
    assert F(170, 8) == F(85, 4)
    h = ev.hand(0.8)
    assert de(h["s"], 3) == "4,610" and de(h["z"], 4) == "0,8416" and de(h["zs"], 3) == "3,880"
    assert [de(u, 2) for u in h["upper"]] == ["7,88", "11,88"]
    assert h["ols_cov"] == [5, 3] and sum(h["ols_cov"]) == 8
    assert F(h["z"]) ** 2 < F(64, 85)                                              # die 12 liegt über der Grenze: knapp
    assert h["qr_loss"] == 10 and [q["loss"] for q in h["qr"]] == [2, 8]
    assert [(q["lo"], q["hi"]) for q in h["qr"]] == [(5, 6), (12, 16)]
    assert [q["cov_lo"] for q in h["qr"]] == [4, 4] and [q["cov_hi"] for q in h["qr"]] == [5, 5]
    _has("**s² = 170/8 = 85/4**", "s = 4,610", "z(0,8) = 0,8416", "z·s = 3,880", "7,88 und 11,88", "**5 von 5**", "**3 von 5**", "zusammen 8 von 10 = 0,8", "(Summe **10**)",
         "[5, 6]", "[12, 16]", "4 von 5 in jeder Gruppe", "5 von 5: **die Überdeckung im Training ist eine Treppe mit Stufen 1/5", "z² < 64/85", "SSE = 10 + 160 = 170", "3,88 gegen 4")


# --- B2 -------------------------------------------------------------------------------------------------------------------------------------

def _enum_min(x, y, t):
    """Kleinster Pinball-Verlust über alle Geraden durch zwei Punkte, in Brüchen."""
    best = None
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            sl = (y[j] - y[i]) / (x[j] - x[i])
            ic = y[i] - sl * x[i]
            loss = sum((t * (v - ic - sl * u) if v - ic - sl * u >= 0 else (t - 1) * (v - ic - sl * u)) for u, v in zip(x, y))
            best = loss if best is None or loss < best else best
    return best


def test_b2_seven_points_and_eight_points_exact():
    x7 = [F(v) for v in (1, 2, 3, 4, 5, 6, 7)]
    y7 = [F(1), F(7, 2), F(2), F(6), F(4), F(9), F(11, 2)]
    for t, exact in ((F(1, 4), F(41, 16)), (F(1, 2), F(37, 8)), (F(9, 10), F(51, 40))):
        assert _enum_min(x7, y7, t) == exact
    assert 7 * (1 - F(9, 10)) == F(7, 10) and F(7, 10) < 1
    X = np.column_stack([np.ones(7), [float(v) for v in x7]])
    b, _ = A.qr_fit(X, np.array([float(v) for v in y7]), 0.9)
    r = np.array([float(v) for v in y7]) - X @ b
    assert int(np.sum(r <= C.RES_ZERO)) == 7 and int(np.sum(np.abs(r) <= C.RES_ZERO)) == 2
    yy = [F(3), F(1), F(4), F(3, 2), F(9), F(2), F(6), F(5)]
    for t, loss, seg in ((F(1, 4), F(43, 8), (F(3, 2), F(2))), (F(1, 2), F(33, 4), (F(3), F(4))), (F(3, 4), F(57, 8), (F(5), F(6)))):
        assert A.quantile_interval(yy, t) == seg and A.pinball_exact(yy, seg[0], t) == loss == A.pinball_exact(yy, seg[1], t)
    _has("**41/16**", "**37/8**", "**51/40**", "0,7 < 1", "zwei genau darauf", "**43/8**", "**33/4**", "**57/8**", "[3/2, 2], [3, 4], [5, 6]")


# --- B3 -------------------------------------------------------------------------------------------------------------------------------------

def test_b3_world_a_limit():
    assert F(4, 100) + F(16, 100) * 5 + F(16, 100) * F(100, 3) == F(463, 75) and de(463 / 75, 4) == "6,1733"
    assert de((463 / 75) ** 0.5, 3) == "2,485"
    xg = S.grid_ab()
    lim = A.ols_limit_coverage_a(0.9, xg, lambda v: S.sd_ab(C.WELT_A, v))
    groups = [float(np.mean(lim[g])) for g in S.groups_ab()]
    assert " / ".join(de(v) for v in groups) == "0,996 / 0,926 / 0,820" and de(float(np.mean(lim))) == "0,914"
    assert de(float(A.ols_limit_coverage_a(0.9, np.array([10.0]), lambda v: S.sd_ab(C.WELT_A, v))[0]), 3) == "0,776"
    # Messung bei n = 1 000 (150 Wiederholungen) liegt in der oberen Gruppe bei 0,819 gegen 0,820 im Grenzwert: der Fehler ist systematisch
    assert de(_cell(C.WELT_A, 1000, 0.9)["methods"]["ols1"]["groups"][2]) == "0,819" and abs(_cell(C.WELT_A, 1000, 0.9)["methods"]["ols1"]["groups"][2] - groups[2]) < 0.002
    _has("**463/75 = 6,1733**", "s = 2,485", "**0,996 / 0,926 / 0,820**", "gesamt 0,914", "Φ(0,758) = 0,776", "sd 4,2", "bei n = 1 000 liegt die Messung oben bei 0,819 gegen 0,820 im Grenzwert")


# --- Absatz „Warum dieses Problem“ ---------------------------------------------------------------------------------------------------------------

def test_warum_paragraph_numbers_for_the_containers():
    m = _cell(C.WELT_C8, 3000, 0.9)["methods"]
    assert de(m["ols1"]["total"]) == "0,918" and de(m["ols1"]["groups"][3]) == "0,775" and de(m["ols1"]["groups"][1]) == "0,992"
    assert (de(min(m["qr"]["groups"])), de(max(m["qr"]["groups"]))) == ("0,896", "0,900")
    assert (de(min(m["olsg"]["groups"])), de(max(m["olsg"]["groups"]))) == ("0,898", "0,900")
    assert C.S_BASE == (0.55, 0.25, 0.35, 0.90)                                  # „log-Streuung 0,90 gegen 0,25 bei Export“
    _has("gesamt 0,918 (Soll 0,9;", "nur 0,775, Export dagegen 0,992", "(0,896 bis 0,900)", "(0,898 bis 0,900)", "0,90 gegen 0,25 bei Export")


def test_app_limits_table_numbers_match_the_measurement():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    q30 = _cell(C.WELT_A, 30, 0.9)["methods"]["qr"]
    assert f"{de(q30['p5'])} bis {de(q30['p95'])}" == "0,762 bis 0,951"
    assert (de(_cell(C.WELT_A, 30, 0.99)["methods"]["qr"]["total"]), de(_cell(C.WELT_A, 1000, 0.99)["methods"]["qr"]["total"])) == ("0,937", "0,989")
    b = _cell(C.WELT_B, 1000, 0.9)["methods"]["qr"]["groups"]
    assert (de(b[2]), de(b[1])) == ("0,869", "0,946")
    for phrase in ("5. bis 95. Perzentil 0,762 bis 0,951", "bei n = 30 im Mittel bei 0,937 und erreicht erst bei n = 1 000 0,989", "0,869 statt 0,9, bei 0,946 in der mittleren Gruppe"):
        assert phrase in app, phrase


# --- B4 -------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_b4_world_a_n200(tau):
    c = _cell(C.WELT_A, 200, tau)
    assert c["reps"] == 400
    o, q = c["methods"]["ols1"], c["methods"]["qr"]
    bold = tau == 0.9
    wrap = (lambda s: f"**{s}**") if bold else (lambda s: s)
    row = f"| {'**0,9**' if bold else de(tau, 1)} | {wrap(_g(o))}, gesamt {de(o['total'])} | {wrap(_g(q))}, gesamt {de(q['total'])} |"
    _has(row)
    # Aussage: bei tau = 0,5 trifft beides in jeder Gruppe (größte Abweichung unter 0,01), bei 0,9 nur die Quantilregression
    dev = lambda e: max(abs(v - tau) for v in e["groups"])
    if tau == 0.5:
        assert dev(o) < 0.01 and dev(q) < 0.01
    if tau == 0.9:
        assert dev(o) > 0.09 and dev(q) < 0.005


def test_b4_text_mentions_median_case():
    _has("Beim Median fällt das Band auf die Vorhersage zusammen (z = 0)", "Preset „Ohne Effekt“")
    assert A.z_of(0.5) == 0.0


# --- B5 -------------------------------------------------------------------------------------------------------------------------------------

def test_b5_amount_table_tau09():
    expected = {
        30: ("0,872", "0,762 bis 0,951", "33,0 %", "0,899"), 50: ("0,886", "0,812 bis 0,951", "20,2 %", "0,906"), 100: ("0,890", "0,828 bis 0,939", "10,8 %", "0,908"),
        200: ("0,898", "0,864 bis 0,928", "2,0 %", "0,912"), 1000: ("0,899", "0,882 bis 0,915", "0,0 %", "0,913"),
    }
    for n, (tot, band, under, ols) in expected.items():
        c = _cell(C.WELT_A, n, 0.9)
        q, o = c["methods"]["qr"], c["methods"]["ols1"]
        assert (de(q["total"]), f"{de(q['p5'])} bis {de(q['p95'])}", pct(q["share_under"]), de(o["total"])) == (tot, band, under, ols), n
        assert c["reps"] == (150 if n == 1000 else 400)
        _has(f"| {ev.de_int(n)} | {tot} | {band} | {under} | {ols} |")


def test_b5_high_level_tau099():
    qs = [de(_cell(C.WELT_A, n, 0.99)["methods"]["qr"]["total"]) for n in (30, 50, 100, 200, 1000)]
    assert qs == ["0,937", "0,963", "0,980", "0,986", "0,989"]
    c30 = _cell(C.WELT_A, 30, 0.99)
    assert pct(c30["methods"]["qr"]["share_under"]) == "42,2 %" and pct(_cell(C.WELT_A, 200, 0.99)["methods"]["qr"]["share_under"]) == "0,0 %"
    assert pct(_cell(C.WELT_A, 1000, 0.99)["methods"]["qr"]["share_under"]) == "0,0 %"
    o, q = c30["methods"]["ols1"], c30["methods"]["qr"]
    assert (de(o["total"]), de(q["total"]), de(o["min_group"]), de(q["min_group"])) == ("0,971", "0,937", "0,930", "0,891")
    assert abs(o["total"] - 0.99) < abs(q["total"] - 0.99)                       # „näher am Soll“ gilt hier
    _has("0,937 / 0,963 / 0,980 / 0,986 / 0,989", "42,2 % bei n = 30", "(0,971 bei n = 30)", "(0,937)", "(0,930 gegen 0,891)")


# --- B6 -------------------------------------------------------------------------------------------------------------------------------------

def test_b6_groups_at_small_n():
    expected = {30: ("0,799 / 0,816", "46,0 % / 35,2 %"), 100: ("0,812 / 0,863", "38,0 % / 9,0 %"), 200: ("0,818 / 0,880", "26,5 % / 0,8 %")}
    for n, (mins, shares) in expected.items():
        c = _cell(C.WELT_A, n, 0.9)
        o, q = c["methods"]["ols1"], c["methods"]["qr"]
        assert f"{de(o['min_group'])} / {de(q['min_group'])}" == mins and f"{pct(o['share_group_under'])} / {pct(q['share_group_under'])}" == shares, n
        _has(f"| {n} | {mins} | {shares} |")
    c = _cell(C.WELT_A, 30, 0.9)
    assert _g(c["methods"]["ols1"]) == "0,992 / 0,907 / 0,799" and _g(c["methods"]["qr"]) == "0,872 / 0,880 / 0,866"
    _has("Kleinste Quadrate ± z·s 0,992 / 0,907 / 0,799, Quantilregression 0,872 / 0,880 / 0,866")


# --- B7 -------------------------------------------------------------------------------------------------------------------------------------

def test_b7_crossing():
    got = " / ".join(pct(_cell(C.WELT_A, n, 0.9)["cross"]) for n in (30, 50, 100, 200, 1000))
    assert got == "13,8 % / 12,0 % / 3,5 % / 1,0 % / 0,0 %"
    _has("**13,8 % / 12,0 % / 3,5 % / 1,0 % / 0,0 %**")


# --- B8 -------------------------------------------------------------------------------------------------------------------------------------

def test_b8_container_table_tau09_n3000():
    c = _cell(C.WELT_C8, 3000, 0.9)
    assert c["reps"] == 40
    rows = {
        "ols1": "| Kleinste Quadrate (log) ± z·s, ein s | 0,918 | 0,859 | 0,992 | 0,967 | **0,775** |",
        "olsg": "| Kleinste Quadrate (log) ± z·s_Art | 0,898 | 0,898 | 0,898 | 0,899 | 0,900 |",
        "qr": "| Quantilregression (log) | 0,898 | 0,900 | 0,896 | 0,898 | 0,898 |",
        "ols1_roh": "| Kleinste Quadrate roh ± z·s, ein s | 0,959 | 0,934 | 0,996 | 0,998 | 0,833 |",
        "qr_roh": "| Quantilregression roh | 0,899 | 0,900 | 0,898 | 0,897 | 0,898 |",
    }
    for m, line in rows.items():
        e = c["methods"][m]
        vals = [de(e["total"])] + [de(v) for v in e["groups"]]
        body = line.strip("|").split("|")[1:]
        assert [b.strip().strip("*") for b in body] == vals, m
        _has(line)


def test_b8_tau01_and_n300_and_the_group_wise_s_is_as_good():
    c = _cell(C.WELT_C8, 3000, 0.1)
    o = c["methods"]["ols1"]
    assert de(o["total"]) == "0,084" and _g(o) == "0,142 / 0,009 / 0,035 / 0,237"
    r = c["methods"]["ols1_roh"]
    assert r["groups"][1] == 0.0 and r["groups"][2] == 0.0 and de(r["groups"][1]) == "0,000"
    c3 = _cell(C.WELT_C8, 300, 0.9)
    assert c3["reps"] == 100
    q, g = c3["methods"]["qr"], c3["methods"]["olsg"]
    assert (de(q["min_group"]), de(g["min_group"]), pct(q["share_group_under"]), pct(g["share_group_under"])) == ("0,838", "0,857", "18,0 %", "8,0 %")
    # „so gut wie“: bei n = 3 000 ist die größte Abweichung einer Art vom Soll bei beiden unter 0,005
    c0 = _cell(C.WELT_C8, 3000, 0.9)["methods"]
    assert max(abs(v - 0.9) for v in c0["olsg"]["groups"]) < 0.005 and max(abs(v - 0.9) for v in c0["qr"]["groups"]) < 0.005
    _has("ein s gesamt 0,084 (je Art 0,142 / 0,009 / 0,035 / 0,237)", "**0,000**", "kleinster Artwert einer Wiederholung im Mittel 0,838 gegen 0,857", "18,0 % gegen 8,0 %")


# --- B9 -------------------------------------------------------------------------------------------------------------------------------------

def test_b9_cells_art_times_weekend():
    expected = {(0.9, "c8"): ("0,051", "0,045"), (0.9, "c11"): ("0,004", "0,007"), (0.5, "c8"): ("0,131", "0,089"), (0.5, "c11"): ("0,010", "0,009")}
    for (tau, welt), (g, q) in expected.items():
        m = _cell(welt, 3000, tau)["methods"]
        assert (de(m["olsg"]["max_cell_dev"]), de(m["qr"]["max_cell_dev"])) == (g, q), (tau, welt)
        assert m["olsg"]["max_cell_dev"] == pytest.approx(max(abs(v - tau) for v in m["olsg"]["cells"]), abs=2e-4)
    _has("| 0,9 | 8 Parameter (ein Wochenend-Koeffizient) | 0,051 | 0,045 |", "| 0,9 | 11 Parameter (je Art) | 0,004 | 0,007 |", "| 0,5 | 8 Parameter | 0,131 | 0,089 |", "| 0,5 | 11 Parameter | 0,010 | 0,009 |",
         "(bei τ = 0,9 0,045 gegen 0,051 bei s je Art)")
    m = _cell(C.WELT_C11, 300, 0.9)["methods"]
    assert (de(m["olsg"]["max_cell_dev"]), de(m["qr"]["max_cell_dev"])) == ("0,018", "0,029")
    _has("0,018 (s je Art) gegen 0,029 (Quantilregression)")
    # Aussage: bei 11 Parametern liegt keine Zelle mehr als 0,01 daneben (n = 3 000), bei 8 Parametern mehr als 0,04
    assert _cell(C.WELT_C11, 3000, 0.9)["methods"]["qr"]["max_cell_dev"] < 0.01 and _cell(C.WELT_C8, 3000, 0.9)["methods"]["qr"]["max_cell_dev"] > 0.04


# --- B10 ------------------------------------------------------------------------------------------------------------------------------------

def test_b10_false_class():
    c = _cell(C.WELT_B, 1000, 0.9)
    assert c["reps"] == 100
    o, q = c["methods"]["ols1"], c["methods"]["qr"]
    assert _g(o) == "1,000 / 0,971 / 0,806" and de(o["total"]) == "0,923" and _g(q) == "0,893 / 0,946 / 0,869" and de(q["total"]) == "0,900"
    assert (de(q["min_group"]), de(o["min_group"])) == ("0,866", "0,806")
    assert q["groups"][2] < 0.9 - ev.C.TOL_OK + 0.001 and q["groups"][1] > 0.9 + 0.04            # obere Gruppe daneben, mittlere zu viel
    _has("1,000 / 0,971 / 0,806, gesamt 0,923", "**Quantilregression: 0,893 / 0,946 / 0,869**", "gesamt 0,900", "0,866 (± z·s: 0,806)", "0,869", "0,946")


# --- B11 ------------------------------------------------------------------------------------------------------------------------------------

def test_b11_gegenprobe_300_of_300():
    sys.path.insert(0, str(ROOT / "tools"))
    import que_gegenprobe as G

    res = G.run(300)
    assert res["violations"] == 0 and res["with_zero_residual"] == 300 and res["enum_bad"] == 0 and res["enum_checked"] == 81
    _has("**300 von 300**", "(alle 300 mit mindestens einem Punkt auf der Geraden)", "in den 81 Instanzen", "0 Abweichungen")


# --- Korrekturen, Hinweise und Zahlen im Text ----------------------------------------------------------------------------------------------------

def test_corrections_section_numbers():
    c = _cell(C.WELT_A, 30, 0.9)
    assert pct(c["methods"]["ols1"]["share_group_under"]) == "46,0 %" and pct(c["methods"]["qr"]["share_group_under"]) == "35,2 %"
    assert pct(_cell(C.WELT_A, 50, 0.9)["cross"]) == "12,0 %" and pct(c["cross"]) == "13,8 %"
    assert [len(g) for g in S.groups_ab()] == [7, 6, 7]
    m3 = _cell(C.WELT_C11, 3000, 0.9)["methods"]
    assert de(m3["qr"]["max_cell_dev"]) == "0,007" and de(m3["olsg"]["max_cell_dev"]) == "0,004"
    q200 = _cell(C.WELT_A, 200, 0.9)["methods"]["qr"]
    assert max(abs(v - 0.9) for v in q200["groups"]) < 0.0035 and de(_cell(C.WELT_A, 200, 0.9)["methods"]["ols1"]["total"]) == "0,912"
    assert max(abs(v - 0.9) for v in c["methods"]["qr"]["groups"]) == pytest.approx(0.034, abs=0.0006)
    assert max(abs(v - 0.9) for v in _cell(C.WELT_B, 1000, 0.9)["methods"]["qr"]["groups"]) == pytest.approx(0.046, abs=0.0006)
    _has("**7 / 6 / 7**", "46,0 % gegen 42,5 %", "unverändert 35,2 %", "13,8 % gegen 17,5 % (n = 30), 12,0 % gegen 11,3 % (n = 50)", "0,897 / 0,899 / 0,898 gegen 0,891 / 0,894 / 0,892",
         "0,996 / 0,924 / 0,818 gegen 0,996 / 0,921 / 0,814", "0,872 gegen 0,877", "33,0 % gegen 30,7 %", "**0,007**", "(0,034)", "(0,046)", "gesamt 0,912 gegen 0,898 bei n = 200", "0,003")


def test_every_offered_n_has_the_documented_repetitions_and_the_runtime_numbers_are_plausible():
    assert [C.REPS[(C.WELT_A, n)] for n in C.N_OPTIONS[C.WELT_A]] == [400, 400, 400, 400, 150]
    assert [C.REPS[(C.WELT_C8, n)] for n in C.N_OPTIONS[C.WELT_C8]] == [100, 40] and [C.REPS[(C.WELT_B, n)] for n in C.N_OPTIONS[C.WELT_B]] == [100, 100]
    _has("40 bis 400 frische Stichproben", "(282 s Rechenzeit für alle Zellen)", "40 Wiederholungen, Test auf 20 000 festen Testcontainern")
    assert C.TEST_N == 20_000 and ev.C.N_GRID == 20


def test_readme_and_app_state_the_same_formulas_and_sources():
    """Die zwei Fassungen desselben Textes (App-Expander und README) nennen dieselben Gleichungen, Namen und Quellen; ein Test vergleicht sie, statt sie von Hand synchron zu halten."""
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for phrase in ("**kein** Konfidenzintervall der Steigung", r"\tfrac{463}{75}", "Jackknife", "Davidson und MacKinnon 1993", "Koenker und Bassett 1978", "arXiv 1903.04684", "arXiv 1905.03222",
                   "MacKinnon 2011", "eigene Herleitung"):
        assert phrase in app, phrase
    for phrase in ("**kein** Konfidenzintervall der Steigung", "463/75", "Jackknife", "Davidson und MacKinnon 1993", "Koenker und Bassett 1978", "arXiv 1903.04684", "arXiv 1905.03222", "MacKinnon 2011", "eigene Herleitung"):
        assert phrase in README, phrase


def test_test_count_in_readme_matches_the_suite():
    import re
    import subprocess

    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, stdin=subprocess.DEVNULL)
    m = re.search(r"(\d+) tests? collected", out.stdout)
    assert m, out.stdout[-300:]
    assert f"{m.group(1)} Tests (`pytest tests`)" in README, m.group(1)
