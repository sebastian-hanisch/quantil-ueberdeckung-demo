"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zellen (Zusammenfassungen einer Wiederholungsmessung), keine Zufalls-Seeds. Damit prüft der Test die Grenzen
(TOL_OK = 0,03, TOL_FAIL = 0,05, MEAN_GAP = 0,02) und jede genannte Zahl direkt. Zusätzlich: je Preset die erwartete Verdikt-Zeile an der vorgerechneten Messung
(ein Preset ohne Effekt meldet keine Abweichung)."""

import pytest

import que_constants as C
import que_evaluation as ev


def _entry(groups, total=None, **kw):
    e = {"groups": groups, "total": sum(groups) / len(groups) if total is None else total, "p5": 0.8, "p50": 0.9, "p95": 0.95,
         "share_under": 0.1, "min_group": min(groups), "share_group_under": 0.2}
    e.update(kw)
    return e


def _cell(ols1, qr, olsg=None, reps=50, cross=None):
    m = {"ols1": ols1, "qr": qr}
    if olsg is not None:
        m["olsg"] = olsg
    return {"reps": reps, "methods": m, "cross": cross}


# --- Kernfrage 1 ------------------------------------------------------------------------------------------------------------------------------

def test_coverage_both_hit_is_info_and_says_so():
    cell = _cell(_entry([0.51, 0.50, 0.49], 0.50), _entry([0.50, 0.50, 0.50], 0.50))
    kind, text = ev.verdict_coverage(C.WELT_A, 0.5, 200, cell)
    assert kind == "info"
    assert "beide Verfahren treffen in jeder Gruppe" in text and "größte Abweichung vom Soll 0,010" in text and "Mittel über 50 Wiederholungen" in text
    assert "Hier ist keine Abweichung zu sehen" in text


def test_coverage_threshold_is_inclusive_at_the_ok_limit():
    ok = _cell(_entry([0.93, 0.90, 0.90]), _entry([0.90, 0.90, 0.87]))                 # genau 0,03 Abweichung: noch getroffen
    assert ev.verdict_coverage(C.WELT_A, 0.9, 100, ok)[0] == "info"
    assert "beide Verfahren treffen" in ev.verdict_coverage(C.WELT_A, 0.9, 100, ok)[1]


def test_coverage_ols_misses_and_quantile_regression_hits_is_a_warning_with_all_numbers():
    cell = _cell(_entry([0.996, 0.924, 0.818], 0.912), _entry([0.897, 0.899, 0.898], 0.898))
    kind, text = ev.verdict_coverage(C.WELT_A, 0.9, 200, cell)
    assert kind == "warning"
    assert "0,996 / 0,924 / 0,818 (gesamt 0,912)" in text and "0,897 / 0,899 / 0,898 (gesamt 0,898)" in text
    assert "die Gesamtüberdeckung liegt nahe am Soll" in text                         # |0,912 − 0,9| <= 0,03
    assert "die größte Abweichung einer Gruppe vom Soll ist 0,003" in text
    assert "Grenzwert für große n (Rechnung): 0,996 / 0,926 / 0,820" in text and "systematisch" in text
    assert "kein Gewinn" not in text                                                  # ohne Verfahren mit s je Gruppe keine Aussage dazu


def test_coverage_quantile_regression_also_misses_is_named():
    cell = _cell(_entry([1.0, 0.986, 0.930], 0.971), _entry([0.929, 0.948, 0.934], 0.937))
    kind, text = ev.verdict_coverage(C.WELT_A, 0.99, 30, cell)
    assert kind == "warning" and "Auch die Quantilregression trifft nicht" in text and "die größte Abweichung einer Gruppe vom Soll ist 0,061" in text


def test_coverage_neither_mean_nor_groups_when_total_is_far_off():
    cell = _cell(_entry([0.95, 0.99, 0.99], 0.98), _entry([0.90, 0.90, 0.90]))          # Gesamtüberdeckung 0,08 über dem Soll
    kind, text = ev.verdict_coverage(C.WELT_B, 0.9, 200, cell)
    assert kind == "warning" and "weder im Mittel noch in den Gruppen" in text


def test_coverage_simple_route_is_named_only_if_the_group_wise_s_hits():
    ols = _entry([0.859, 0.992, 0.967, 0.775], 0.918)
    qr = _entry([0.900, 0.896, 0.898, 0.898], 0.898)
    hit = _entry([0.898, 0.898, 0.899, 0.900], 0.898)
    kind, text = ev.verdict_coverage(C.WELT_C8, 0.9, 3000, _cell(ols, qr, hit, reps=40))
    assert kind == "warning" and "Mit einem s je Art (0,898 / 0,898 / 0,899 / 0,900)" in text and "kein Gewinn, sondern ein Umweg" in text
    assert "Import / Export / Umschlag / Leer" in text and "Grenzwert" not in text           # kein Grenzwert in der Container-Welt
    miss = _entry([0.85, 0.90, 0.90, 0.95], 0.90)
    _, text2 = ev.verdict_coverage(C.WELT_C8, 0.9, 3000, _cell(ols, qr, miss, reps=40))
    assert "kein Gewinn" not in text2


def test_coverage_in_between_is_info_without_claims():
    cell = _cell(_entry([0.90, 0.90, 0.85]), _entry([0.90, 0.90, 0.86]))            # 0,05 und 0,04: weder getroffen (> 0,03) noch klar daneben (> 0,05 verlangt)
    kind, text = ev.verdict_coverage(C.WELT_A, 0.9, 100, cell)
    assert kind == "info" and "weder klar daneben noch klar getroffen" in text


def test_coverage_missing_cell_says_so():
    kind, text = ev.verdict_coverage(C.WELT_A, 0.9, 77, None)
    assert kind == "info" and "keine vorgerechnete Messung" in text and "n = 77" in text


def test_coverage_limit_sentence_is_omitted_at_the_median():
    cell = _cell(_entry([0.60, 0.50, 0.50]), _entry([0.50, 0.50, 0.50]))
    _, text = ev.verdict_coverage(C.WELT_A, 0.5, 200, cell)
    assert "Grenzwert" not in text


# --- Handbeispiel ----------------------------------------------------------------------------------------------------------------------------

def test_hand_verdict_at_four_fifths_names_five_of_five_and_three_of_five():
    kind, text = ev.verdict_hand(ev.hand(0.8))
    assert kind == "warning" and text.startswith("Im Mittel stimmt es, je Gruppe nicht.")
    assert "deckt im Training 5 von 5 Punkten in Gruppe 0 und 3 von 5 in Gruppe 1, zusammen 8 von 10 (0,8)" in text
    assert "s = 4,61 (Grenzen 7,88 und 11,88)" in text
    assert "deckt 4 von 5 und 4 von 5" in text and "auf [5; 6] und [12; 16] flach" in text


def test_hand_verdict_is_info_when_both_groups_are_covered_alike():
    kind, text = ev.verdict_hand(ev.hand(0.5))                # 3 von 5 und 3 von 5
    assert kind == "info" and "beide Gruppen nicht auseinander" in text and "3 von 5 Punkten in Gruppe 0 und 3 von 5 in Gruppe 1" in text


# --- Kernfrage 2 und 3 -------------------------------------------------------------------------------------------------------------------------

def test_amount_verdict_names_both_ends_and_warns_only_if_the_mean_is_off():
    c30 = _cell(_entry([0.992, 0.907, 0.799], 0.899), _entry([0.872, 0.880, 0.866], 0.872, p5=0.762, p95=0.951, share_under=0.33, share_group_under=0.352), reps=400, cross=0.138)
    c1000 = _cell(_entry([0.996, 0.925, 0.819], 0.913), _entry([0.899, 0.899, 0.899], 0.899, share_under=0.0), reps=150, cross=0.0)
    kind, text = ev.verdict_amount(C.WELT_A, 0.9, [(30, c30), (1000, c1000)])
    assert kind == "warning"                                  # |0,872 − 0,9| = 0,028 > 0,02
    assert "0,872 bei n = 30" in text and "0,762 bis 0,951" in text and "33,0 %" in text and "0,899 bei n = 1 000 (0,0 %)" in text
    assert "35,2 %" in text and "(Kleinste Quadrate ± z·s: 20,0 %)" in text
    assert "Damit liegt Kleinste Quadrate ± z·s im Mittel näher am Soll" in text          # |0,899 − 0,9| gegen 0,028: Unterschied über 0,01
    assert "13,8 %" in text and "0,0 % bei n = 1 000" in text and "nur diese zwei Randstellen geprüft" in text
    ok = _cell(_entry([0.9, 0.9, 0.9]), _entry([0.89, 0.9, 0.9], 0.895), reps=40)           # |0,895 − 0,9| = 0,005
    assert ev.verdict_amount(C.WELT_B, 0.9, [(200, ok), (1000, ok)])[0] == "info"


def test_amount_verdict_without_crossing_sentence_outside_world_a_and_for_the_median():
    c = _cell(_entry([0.1, 0.1, 0.1]), _entry([0.1, 0.1, 0.1]), reps=100, cross=None)
    _, text = ev.verdict_amount(C.WELT_B, 0.1, [(200, c), (1000, c)])
    assert "Randstellen" not in text and "unter τ (Kleinste Quadrate" not in text             # bei tau = 0,1 keine Aussage über Gruppen unter τ − 0,10 (Zahl wäre unter 0)
    c9 = _cell(_entry([0.9, 0.9, 0.9]), _entry([0.9, 0.9, 0.9]), reps=100, cross=0.2)
    assert "Randstellen" in ev.verdict_amount(C.WELT_A, 0.9, [(200, c9), (1000, c9)])[1]


def test_amount_verdict_for_the_hand_example_has_no_series():
    kind, text = ev.verdict_amount(C.WELT_HAND, 0.8, [])
    assert kind == "info" and "genau zehn Punkte" in text


def test_simple_verdict_names_the_cells_and_the_fix_only_if_the_numbers_show_it():
    g8 = _entry([0.898, 0.898, 0.899, 0.900], 0.898, max_cell_dev=0.051)
    q8 = _entry([0.900, 0.896, 0.898, 0.898], 0.898, max_cell_dev=0.045)
    g11 = _entry([0.897, 0.898, 0.899, 0.900], 0.898, max_cell_dev=0.004)
    q11 = _entry([0.899, 0.896, 0.898, 0.899], 0.898, max_cell_dev=0.007)
    c8 = {"reps": 40, "methods": {"ols1": g8, "olsg": g8, "qr": q8}, "cross": None}
    c11 = {"reps": 40, "methods": {"ols1": g11, "olsg": g11, "qr": q11}, "cross": None}
    kind, text = ev.verdict_simple(c8, c11, 0.9, 3000)
    assert kind == "info"
    assert "bis 0,051 (s je Art) bzw. 0,045 (Quantilregression)" in text and "bis 0,004 bzw. 0,007" in text
    assert "behebt die Abweichung nicht: erst das fehlende Wochenend-Glied je Art senkt sie (Quantilregression von 0,045 auf 0,007)" in text
    q8_small = dict(q8, max_cell_dev=0.02)
    c8b = {"reps": 40, "methods": {"ols1": g8, "olsg": g8, "qr": q8_small}, "cross": None}
    _, text2 = ev.verdict_simple(c8b, c11, 0.9, 3000)
    assert "behebt die Abweichung nicht" not in text2 and "weicht keine Zelle" in text2
    assert ev.verdict_simple(None, c11, 0.9, 3000)[0] == "info"


# --- je Preset die erwartete Verdikt-Zeile (an der vorgerechneten Messung) ----------------------------------------------------------------------

def _preset_cell(name):
    p = C.PRESETS[name]
    s = ev.sanitize(p["welt"], p["tau"], p["n"], p["seed"])
    return s, ev.frozen_cell(s.welt, s.n, s.tau)


def test_preset_without_effect_reports_no_deviation():
    s, cell = _preset_cell("Ohne Effekt (τ = 0,5)")
    kind, text = ev.verdict_coverage(s.welt, s.tau, s.n, cell)
    assert kind == "info" and "Hier ist keine Abweichung zu sehen" in text
    rows = [(n, ev.frozen_cell(s.welt, n, s.tau)) for n in C.N_OPTIONS[s.welt]]
    assert ev.verdict_amount(s.welt, s.tau, rows)[0] == "info"


@pytest.mark.parametrize("name", [n for n in C.PRESETS if n not in ("Ohne Effekt (τ = 0,5)", "Von Hand (zehn Punkte)")])
def test_presets_with_an_effect_report_a_warning_in_question_one(name):
    s, cell = _preset_cell(name)
    kind, text = ev.verdict_coverage(s.welt, s.tau, s.n, cell)
    assert kind == "warning", (name, text)


def test_preset_hand_is_a_warning_with_the_count():
    kind, text = ev.verdict_hand(ev.hand(C.PRESETS["Von Hand (zehn Punkte)"]["tau"]))
    assert kind == "warning" and "5 von 5" in text and "3 von 5" in text


def test_preset_container_with_group_wise_s_names_the_simple_route():
    for name in ("Container, 8 Parameter", "Container, Wochenende je Art"):
        s, cell = _preset_cell(name)
        assert "kein Gewinn, sondern ein Umweg" in ev.verdict_coverage(s.welt, s.tau, s.n, cell)[1], name


def test_preset_false_class_names_the_upper_group():
    s, cell = _preset_cell("Falsche Klasse (Rohskala B)")
    _, text = ev.verdict_coverage(s.welt, s.tau, s.n, cell)
    assert "0,869" in text and "Quantilregression 0,893 / 0,946 / 0,869" in text


# --- Formatierung -----------------------------------------------------------------------------------------------------------------------------

def test_number_formatting():
    assert ev.de(0.3066, 3) == "0,307" and ev.de(-0.5, 1) == "−0,5" and ev.de(0.25, 3, sign=True) == "+0,250"
    assert ev.pct(0.307) == "30,7 %" and ev.de_int(3000) == "3 000" and ev.tau_text(0.9) == "0,9" and ev.tau_text(0.95) == "0,95" and ev.tau_text(0.99) == "0,99"
