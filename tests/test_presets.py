"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jede Einstellung hat eine vorgerechnete Messung; Permalink begrenzt auf gültige Werte."""

import pytest

import que_constants as C
import que_evaluation as ev
import que_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_offered_by_the_controls():
    for name, p in C.PRESETS.items():
        assert p["welt"] in C.WELTEN and p["tau"] in C.TAUS and p["n"] in C.N_OPTIONS[p["welt"]] and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS.values()))
    assert ev.Settings(**first) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_preset_outside_the_hand_example_has_a_frozen_cell_for_all_three_questions():
    for name, p in C.PRESETS.items():
        if p["welt"] == C.WELT_HAND:
            continue
        assert ev.frozen_cell(p["welt"], p["n"], p["tau"]) is not None, name
        assert all(ev.frozen_cell(p["welt"], n, p["tau"]) is not None for n in C.N_OPTIONS[p["welt"]]), name


def test_every_offered_combination_has_a_frozen_cell():
    for welt in C.WELTEN:
        if welt == C.WELT_HAND:
            continue
        for n in C.N_OPTIONS[welt]:
            for tau in C.TAUS:
                assert ev.frozen_cell(welt, n, tau) is not None, (welt, n, tau)


@pytest.mark.parametrize("welt,tau,n,expected", [
    ("c8", 0.93, 5000, ("c8", 0.95, 3000)),
    ("a", 0.2, 120, ("a", 0.1, 100)),
    ("hand", 0.9, 3, ("hand", 0.9, 10)),
    ("nichts", 0.9, 200, (C.DEFAULT_WELT, 0.9, 200)),
])
def test_sanitize_clips_to_offered_values(welt, tau, n, expected):
    s = ev.sanitize(welt, tau, n, 5)
    assert (s.welt, s.tau, s.n) == expected
