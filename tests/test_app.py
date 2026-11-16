"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Ergänzungen dieser Demo (der Anfang der Datei bleibt die Vorlage) ------------------------------------------------------------------------

def test_every_world_every_n_and_every_level_renders_without_exception():
    import que_constants as C

    at = _fresh_app()
    for welt in C.WELTEN:
        at.sidebar.selectbox(key="welt_select").set_value(welt).run(timeout=TIMEOUT)
        assert not at.exception, (welt, [str(e) for e in at.exception])
        for n in C.N_OPTIONS[welt]:
            at.sidebar.selectbox(key="n_select").set_value(n).run(timeout=TIMEOUT)
            assert not at.exception, (welt, n, [str(e) for e in at.exception])
        for tau in C.TAUS:
            at.sidebar.select_slider(key="tau_slider").set_value(tau).run(timeout=TIMEOUT)
            assert not at.exception, (welt, tau, [str(e) for e in at.exception])


def test_the_three_questions_each_show_a_verdict_and_the_default_names_the_group_miss():
    at = _fresh_app()
    boxes = [w.value for w in list(at.warning) + list(at.info) + list(at.success)]
    assert any("Rohskala A, τ = 0,9, n = 200" in b and "in den Gruppen nicht" in b for b in boxes)
    assert any("Gesamtüberdeckung der Quantilregression 0,872 bei n = 30" in b for b in boxes)
    assert any("Container, τ = 0,9, n = 3 000" in b for b in boxes)


def test_the_zufalls_seed_changes_the_sample_but_not_the_frozen_verdict():
    at = _fresh_app()
    first = [m.value for m in at.metric]
    at.sidebar.slider(key="seed_input").set_value(12345).run(timeout=TIMEOUT)
    assert not at.exception
    second = [m.value for m in at.metric]
    assert first != second
    texts = [w.value for w in at.warning]
    assert any("Mittel über 400 Wiederholungen" in b for b in texts)


def test_permalink_with_invalid_values_is_clipped():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["welt"] = "gibtsnicht"
    at.query_params["tau"] = "0.93"
    at.query_params["n"] = "7"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.selectbox(key="welt_select").value == "a" and at.sidebar.select_slider(key="tau_slider").value == 0.95 and at.sidebar.selectbox(key="n_select").value == 30
