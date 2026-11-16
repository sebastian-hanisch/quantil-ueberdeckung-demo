"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle vier Regler (Welt, Quantilstufe, Trainingsmenge, Zufalls-Seed) werden immer gezeichnet; damit gibt es keinen Regler, der in einer Welt zum Blindgänger wird, und
keinen Zustand eines nicht gezeichneten Reglers. Die Trainingsmenge hat je Welt eigene Auswahlwerte (nur für diese gibt es vorgerechnete Messungen): wechselt die Welt,
setzt `normalize_state` die Menge vor dem Zeichnen auf den nächsten gültigen Wert."""

import time
from dataclasses import dataclass

import streamlit as st

import que_constants as C
import que_evaluation as ev
import que_rng as R


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object


SETTING_SPECS = {
    "welt_select": SettingSpec("welt", str, C.DEFAULT_WELT),
    "tau_slider": SettingSpec("tau", float, C.DEFAULT_TAU),
    "n_select": SettingSpec("n", int, C.DEFAULT_N),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED),
}
PRESET_KEYS = {"welt": "welt_select", "tau": "tau_slider", "n": "n_select", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def normalize_state():
    """Welt, Stufe, Menge und Zufalls-Seed im Session State auf gültige Werte bringen (Menge je nach Welt); wird vor dem Zeichnen der Regler aufgerufen."""
    s = ev.sanitize(st.session_state.get("welt_select", C.DEFAULT_WELT), float(st.session_state.get("tau_slider", C.DEFAULT_TAU)),
                    int(st.session_state.get("n_select", C.DEFAULT_N)), int(st.session_state.get("seed_input", C.DEFAULT_SEED)))
    st.session_state["welt_select"], st.session_state["tau_slider"], st.session_state["n_select"], st.session_state["seed_input"] = s.welt, s.tau, s.n, s.seed


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen (ungültige oder unlesbare Werte werden verworfen, die übrigen auf gültige Werte begrenzt)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                st.session_state[state_key] = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    """Neuer Zufalls-Seed aus der Uhrzeit, durch SplitMix64 gemischt (kein Python-`random`)."""
    st.session_state["seed_input"] = int(R.splitmix_array(time.time_ns() & R.MASK, 1)[0]) % (C.SEED_MAX + 1)
