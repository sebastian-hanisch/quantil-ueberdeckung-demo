"""Quantile statt Mittelwert – trifft das 90-%-Band in jeder Gruppe? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Band „Vorhersage ± z·s“ mit einer gemeinsamen Reststreuung s gegen die lineare Quantilregression: Wo liegt das Band, und trifft es in jeder Gruppe? Die Überdeckung
wird exakt aus der wahren bedingten Verteilungsfunktion gerechnet; Messungen über Wiederholungen sind vorgerechnet (que_frozen.json, tools/que_make_frozen.py).
Alle Formeln und Befunde sind eigene Nachrechnung in diesen Modellen, bei diesen Parametern, Zähleinheit Wiederholung (eine frische Trainingsstichprobe).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import que_constants as C
import que_evaluation as ev
from que_evaluation import de, tau_text
from que_layout import dunkles_design, kopf_css
from que_presets import (
    init_session_state_defaults,
    load_permalink_settings,
    normalize_state,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from que_visualization import build_amount, build_bars, build_container, build_hand, build_scatter

st.set_page_config(page_title="Quantile statt Mittelwert – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _sample(welt, n, seed, tau):
    return ev.fit_cover(welt, n, seed, tau, cross=False, plot=True)


def _table(rows):
    return pd.DataFrame(rows)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in que_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📊 Quantile statt Mittelwert – trifft das 90-%-Band in jeder Gruppe?")
st.markdown(
    """
Wer „den Puffer, der 90 % der Container abdeckt“ braucht, fragt nach dem **Rand der Verteilung eines einzelnen Containers**, nicht nach dem Mittelwert. Die naheliegende Brücke vom Mittelwertmodell ist
**Vorhersage ± z·s** mit **einer** gemeinsamen Reststreuung s; sie nimmt die konstante Streuung wieder an. Die Demo misst **exakt, wo das Band trifft** und wo nicht, und stellt die **lineare Quantilregression**
dagegen. Zähleinheit überall: die **Wiederholung** (eine frische Trainingsstichprobe); alle Zahlen gelten in diesen Modellen, bei diesen Parametern.

Weiter unten: die **❓ Kernfragen**, zwei **🔬 Experimente**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Band", expanded=True):
    st.markdown(
        """
1. **Anpassen:** Kleinste Quadrate liefern die Vorhersage ŷ(x) und die Reststreuung s = √(SSE/(n − p)). Das Band zur Stufe τ ist **ŷ(x) + z·s** mit z = Φ⁻¹(τ).
2. **Quantilregression:** statt des Mittelwerts wird direkt das τ-Quantil angepasst (Linearprogramm über den Pinball-Verlust); der Verlust selbst ist Thema von [`newsvendor-direkt-demo`](https://github.com/sebastian-hanisch/newsvendor-direkt-demo), hier geht es nur darum, ob das Band trifft.
3. **Überdeckung messen:** Die Überdeckung an einer Stelle ist der Anteil der künftigen Beobachtungen, die unter dem Band liegen, **exakt** aus der wahren Verteilung gerechnet und über die Punkte einer Gruppe gemittelt. Soll ist τ. Im Mittel kann es stimmen und je Gruppe nicht.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()
normalize_state()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    welt = st.selectbox("Welt", C.WELTEN, format_func=lambda w: C.WELT_LABELS[w], key="welt_select",
                        help="Handbeispiel: zehn Punkte in Brüchen. Container: Verweilzeit-Container (erfunden), Gruppen sind die vier Arten. Rohskala A und B: y = 1 + 2x + sd(x)·ε mit wachsender Streuung, Gruppen sind die Drittel von x.")
    tau = st.select_slider("Quantilstufe τ", options=C.TAUS, format_func=tau_text, key="tau_slider",
                           help="Soll der Überdeckung: Anteil der künftigen Beobachtungen unter dem Band.")
    n = st.selectbox("Trainingsmenge n", C.N_OPTIONS[welt], format_func=ev.de_int, key="n_select",
                     help="Anzahl der Beobachtungen, an die angepasst wird. Nur diese Mengen sind vorgerechnet; das Handbeispiel hat genau zehn Punkte.")
    seed = st.slider("Zufalls-Seed der Stichprobe", 0, C.SEED_MAX, key="seed_input", step=1,
                     help="Wählt die eine Stichprobe, die oben im Diagramm und im linken Balkendiagramm steht; die vorgerechneten Wiederholungen sind davon unabhängig.")
    st.button("🎲 Neue Stichprobe ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"welt_select": welt, "tau_slider": tau, "n_select": int(n), "seed_input": int(seed)})
s = ev.sanitize(welt, tau, int(n), int(seed))
cell = ev.frozen_cell(s.welt, s.n, s.tau)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown(f"## 🎯 Ergebnis: {C.WELT_SHORT[s.welt]}, τ = {tau_text(s.tau)}, n = {ev.de_int(s.n)}")
if s.welt == C.WELT_HAND:
    h = ev.hand(s.tau)
    methods = ev.hand_methods(h)
    st.plotly_chart(build_hand(h), width="stretch", key="main_chart")
    st.caption("Grau: die zehn Beobachtungen. Orange: obere Grenze „Gruppenmittel + z·s“ mit einem gemeinsamen s. Grün: Minimierer des Pinball-Verlusts je Gruppe (bei Gruppen eine Strecke, wenn n·τ ganzzahlig ist). Gedeckt heißt y ≤ Grenze; die Zahlen gelten im Training.")
else:
    res = _sample(s.welt, s.n, s.seed, s.tau)
    methods = res["methods"]
    if s.welt in C.IS_CONTAINER:
        st.plotly_chart(build_container(res["plot"], s.welt, s.tau), width="stretch", key="main_chart")
        st.caption("Punkte: Verweilzeiten der ersten Container der Stichprobe (erfundene Daten, Kunde und Rauschmerkmal gehen nicht in die Anpassung ein). Waagerechte Striche: obere Grenze des Bandes für einen Bezugscontainer der Art (Werktag, ohne Zollhalt, mittlerer Vorlauf).")
    else:
        st.plotly_chart(build_scatter(res["plot"], s.welt, s.tau), width="stretch", key="main_chart")
        st.caption("Gestrichelt: das wahre τ-Quantil. Orange: Vorhersage + z·s. Grün: Quantilregression. Die gepunkteten Linien trennen die drei Drittel von x, in denen die Überdeckung getrennt gemittelt wird.")
m1, m2, m3 = st.columns(3)
o, q = methods["ols1"], methods["qr"]
unit = "im Training" if s.welt == C.WELT_HAND else "diese Stichprobe"
m1.metric(f"Soll (Quantilstufe τ)", tau_text(s.tau))
m2.metric(f"Kleinste Quadrate ± z·s: gesamt ({unit})", de(o["total"]), delta=f"{de(o['total'] - s.tau, 3, sign=True)} gegen Soll", delta_color="off", delta_arrow="off")
m3.metric(f"Quantilregression: gesamt ({unit})", de(q["total"]), delta=f"{de(q['total'] - s.tau, 3, sign=True)} gegen Soll", delta_color="off", delta_arrow="off")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**1. Trifft das Band in jeder Gruppe, oder nur im Mittel?**")
if s.welt == C.WELT_HAND:
    kind, text = ev.verdict_hand(h)
else:
    kind, text = ev.verdict_coverage(s.welt, s.tau, s.n, cell)
getattr(st, kind)(text)

st.markdown("**2. Was kostet das an Daten, und wann trifft auch die Quantilregression nicht?**")
amount_rows = [(nn, ev.frozen_cell(s.welt, nn, s.tau)) for nn in C.N_OPTIONS[s.welt]] if s.welt != C.WELT_HAND else []
kind, text = ev.verdict_amount(s.welt, s.tau, amount_rows)
getattr(st, kind)(text)

st.markdown("**3. Wann reicht der einfache Weg (ein s je Gruppe)?**")
n_c = s.n if s.welt in C.IS_CONTAINER else 3000
kind, text = ev.verdict_simple(ev.frozen_cell(C.WELT_C8, n_c, s.tau), ev.frozen_cell(C.WELT_C11, n_c, s.tau), s.tau, n_c)
getattr(st, kind)(text)
st.caption("Frage 3 rechnet immer im Container-Vehikel (bei einer anderen Welt in der Einstellung mit n = 3 000) und gilt nur dort: Kleinste Quadrate ± z·s mit einem s je Art ist dort so gut wie die Quantilregression, weil die Streuung der bekannten Art folgt.")

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wo liegt das Band? Überdeckung je Gruppe")
st.caption("Links: die eine Stichprobe der Einstellung. Rechts: der Mittelwert über die vorgerechneten Wiederholungen (jede Wiederholung ist eine frische Trainingsstichprobe, Zufalls-Seeds fest). Die gestrichelte Linie ist das Soll τ.")
c_left, c_right = st.columns(2)
live_methods = {m: methods[m] for m in C.LIVE_METHODS[s.welt] if m in methods}
with c_left:
    st.markdown("**Diese Stichprobe**" if s.welt != C.WELT_HAND else "**Handbeispiel (Anteil der im Training gedeckten Punkte)**")
    st.plotly_chart(build_bars(live_methods, s.welt, s.tau) if s.welt != C.WELT_HAND else build_bars(live_methods, C.WELT_HAND, s.tau), width="stretch", key="bars_sample")
with c_right:
    if cell is not None:
        st.markdown(f"**Mittel über {cell['reps']} Wiederholungen (vorgerechnet)**")
        st.plotly_chart(build_bars({m: cell["methods"][m] for m in C.LIVE_METHODS[s.welt]}, s.welt, s.tau), width="stretch", key="bars_frozen")
    else:
        st.info("Das Handbeispiel hat nur diese zehn Punkte; es gibt keine vorgerechneten Wiederholungen.")
if cell is not None:
    st.markdown("**Vorgerechnete Messung, alle Verfahren**")
    st.dataframe(_table(ev.frozen_table_rows(cell, s.welt, s.tau)), hide_index=True, width="stretch")
    st.caption("Alle Zahlen sind Überdeckungen (Anteil unter dem Band). Die Spalten mit Prozentzahlen zählen Wiederholungen. Gruppen: "
               + (" / ".join(C.ART_LABELS) if s.welt in C.IS_CONTAINER else "Drittel von x (7 / 6 / 7 Gitterpunkte)") + ".")
    if s.welt in C.IS_CONTAINER:
        c8_n, c11_n = ev.frozen_cell(C.WELT_C8, s.n, s.tau), ev.frozen_cell(C.WELT_C11, s.n, s.tau)
        st.markdown("**Zellen Art × Wochenende** (die Quantilregression repariert kein falsches Mittelwertmodell)")
        st.dataframe(_table(ev.cells_table_rows(c8_n, c11_n, s.tau)), hide_index=True, width="stretch")
        st.caption("Im Generator wirkt das Wochenende nur bei Import und Leer. Mit einem gemeinsamen Wochenend-Koeffizienten (8 Parameter) bleiben einzelne Zellen daneben, mit einem je Art (11 Parameter) nicht.")
st.caption("Das Verdikt zu diesem Experiment steht oben unter Kernfrage 1.")

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wann trifft auch die Quantilregression nicht?")
st.caption("Gesamtüberdeckung über die Trainingsmenge, vorgerechnet für die Welt und die Stufe der Einstellung. Das Band ist das 5. bis 95. Perzentil der Gesamtüberdeckung der Quantilregression über die Wiederholungen.")
if amount_rows:
    st.plotly_chart(build_amount(amount_rows, s.tau), width="stretch", key="amount_chart")
    st.dataframe(_table(ev.amount_table_rows(amount_rows, s.welt, s.tau)), hide_index=True, width="stretch")
    st.caption("Das Verdikt zu diesem Experiment steht oben unter Kernfrage 2.")
else:
    st.info("Für das Handbeispiel gibt es keine Messung über die Trainingsmenge.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wo gemessen |
|---|---|---|
| **Eine gemeinsame Reststreuung s** (das Band „Vorhersage ± z·s“) | Bei Streuung, die mit dem Merkmal wächst, deckt das Band unten fast alles und oben zu wenig; im Mittel über die Gruppen kann es trotzdem stimmen. | Experiment 1, Rohskala A, Container |
| **Genug Daten für die Quantilregression** | Bei kleinem n streut die Gesamtüberdeckung stark von Stichprobe zu Stichprobe (Rohskala A, τ = 0,9, n = 30: 5. bis 95. Perzentil 0,762 bis 0,951); bei τ = 0,99 liegt sie bei n = 30 im Mittel bei 0,937 und erreicht erst bei n = 1 000 0,989. | Experiment 2 |
| **Richtige Modellform der Quantilfunktion** | Wächst die Streuung quadratisch und wird eine Gerade angepasst, trifft auch die Quantilregression die obere Gruppe nicht (Rohskala B, n = 1 000, τ = 0,9: 0,869 statt 0,9, bei 0,946 in der mittleren Gruppe). | Rohskala B |
| **Die Quantilregression repariert kein falsches Mittelwertmodell** | Fehlt ein Wochenend-Koeffizient je Art, bleiben einzelne Zellen Art × Wochenende bei beiden Verfahren daneben. | Container, 8 gegen 11 Parameter |
| **Eine Gruppe mit bekannter Streuung** (Containerart) | Dann reicht Kleinste Quadrate ± z·s mit einem s je Art: kein Gewinn durch die Quantilregression. Das gilt nur, wenn die Gruppe die Streuung erklärt. | Experiment 1, Container |
| **Bedingte Überdeckung ist verteilungsfrei nicht zu garantieren** | Konforme Standardverfahren garantieren nur die Überdeckung im Mittel über die Merkmalswerte; bedingt auf einen einzelnen Punkt ist verteilungsfrei keine Garantie möglich (Foygel Barber u. a.), für vorab bekannte Gruppen gibt es gruppenweise Varianten; sie sind hier nicht nachgebaut. | Verweis, nicht gemessen |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Konvention.** Quantilstufe $\tau$ (nicht $\alpha$), $z_\tau = \Phi^{-1}(\tau)$. Residuum $r = y - \hat q$ (Beobachtung minus Band). Beobachtungen $(x_i, y_i)$, $i = 1, \dots, n$, Entwurfsmatrix $X$ mit $p$ Spalten.

| Symbol | Bedeutung |
|---|---|
| $\tau$ | Quantilstufe, zugleich Soll der Überdeckung |
| $z_\tau$ | Standardnormalquantil zur Stufe $\tau$ |
| $\hat q(x)$ | angepasstes Band an der Stelle $x$ |
| $s$ | gemeinsame Reststreuung der Kleinsten Quadrate, $s^2 = \mathrm{SSE}/(n-p)$ |
| $n - p$ | Freiheitsgrade der Reststreuung |
| $c(x)$ | Überdeckung an der Stelle $x$ |

**Band mit einer Reststreuung.** Kleinste Quadrate $\hat\beta = \arg\min \sum_i (y_i - x_i^\top \beta)^2$, dann

$$\hat q_{\mathrm{OLS}}(x) = x^\top \hat\beta + z_\tau \, s, \qquad s^2 = \frac{\sum_i (y_i - x_i^\top \hat\beta)^2}{n - p}.$$

Das ist **kein** Konfidenzintervall der Steigung: es sagt etwas über den Rand der Verteilung eines einzelnen Containers, nicht über die Genauigkeit eines Koeffizienten. (Standardfehler der Koeffizienten, auch die Familie „HC“: [`streuung-wls-demo`](https://github.com/sebastian-hanisch/streuung-wls-demo); Namensfalle HC3 nach MacKinnon 2011, Abschn. 2: bei MacKinnon und White 1985 ist HC3 ein Jackknife-Verfahren, heute heißt so die Form von Davidson und MacKinnon 1993.)

**Pinball-Verlust.**

$$\rho_\tau(r) = \begin{cases} \tau\, r & r \ge 0 \\ (\tau - 1)\, r & r < 0 \end{cases}$$

**Lineare Quantilregression** als Linearprogramm (HiGHS):

$$\min_{b,\,u,\,v}\ \sum_{i=1}^n \bigl(\tau\, u_i + (1-\tau)\, v_i\bigr) \quad \text{mit} \quad X b + u - v = y,\quad u, v \ge 0.$$

**Überdeckung** an der Stelle $x$ für ein Band $\hat q$ bei wahrer bedingter Verteilungsfunktion $F(\cdot \mid x)$:

$$c(x) = F\bigl(\hat q(x) \mid x\bigr).$$

In der Rohskala-Welt ist $y \mid x$ normal mit Mittel $1 + 2x$ und Streuung $\mathrm{sd}(x)$, also $c(x) = \Phi\bigl((\hat q(x) - 1 - 2x)/\mathrm{sd}(x)\bigr)$; im Container-Vehikel ist $\log T \mid x$ normal mit Mittel $\mu(x)$ und Streuung $s(x)$ des Generators, also $c(x) = \Phi\bigl((\log \hat q(x) - \mu(x))/s(x)\bigr)$ für $\hat q(x) > 0$ und $0$ sonst. Die Gruppenüberdeckung ist das Mittel von $c$ über die Punkte der Gruppe, die Gesamtüberdeckung das Mittel über alle Punkte.

**Grenzwert von „Vorhersage ± z·s“ in Rohskala A** (eigene Herleitung): Ist das Mittel richtig angepasst, strebt $s^2$ gegen $E[\mathrm{sd}(x)^2]$ mit $\mathrm{sd}(x) = 0{,}2 + 0{,}4x$ und $x \sim \mathrm{Gleich}(0, 10)$:

$$s^2 \to 0{,}04 + 0{,}16 \cdot 5 + 0{,}16 \cdot \tfrac{100}{3} = \tfrac{463}{75}, \qquad c(x) \to \Phi\!\left(\frac{z_\tau\, s}{\mathrm{sd}(x)}\right).$$

**Optimalitätsbedingung des Achsenabschnitts** (eigene Herleitung aus dem Teilgradienten, im Test an Zufallsinstanzen bestätigt): Für das Minimum des Pinball-Verlusts gilt, mit $r_i = y_i - x_i^\top \hat b$,

$$\#\{i : r_i < 0\} \;\le\; n\,\tau \;\le\; \#\{i : r_i \le 0\}.$$

Folge: Ist $n(1-\tau) < 1$, liegen alle Punkte auf oder unter der Geraden; eine Stufe nahe 1 mit zu wenig Daten deckt im Training alles und sagt über den Rand nichts. Ist $n\tau$ ganzzahlig, ist die Menge der Minimierer bei einer Konstanten oder bei Gruppenmitteln eine Strecke (Handbeispiel), mit stetigen Merkmalen in der Regel ein Punkt; darum vergleichen Tests den Verlust, nicht die Koeffizienten.

**Literatur** (nur, soweit gelesen; ein Wortlaut wird nicht übernommen). Koenker und Bassett 1978, *Regression Quantiles*, Econometrica 46(1), 33–50: Abstract gelesen (Regressionsquantile als Verallgemeinerung der Stichprobenquantile auf das lineare Modell). Koenker 2005, *Quantile Regression*, Cambridge: nur die Kapitelliste. Foygel Barber, Candès, Ramdas und Tibshirani, *The limits of distribution-free conditional predictive inference*, arXiv 1903.04684: Abstract gelesen; exakte **bedingte** Überdeckung ist ohne Verteilungsannahme nicht zu garantieren. Romano, Patterson und Candès, *Conformalized Quantile Regression*, arXiv 1905.03222: nur als Verweis, hier nicht nachgebaut.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
