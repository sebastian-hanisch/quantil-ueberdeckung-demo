"""Konstanten: Welten, Regler-Grenzen, feste Annahmen, Farben und Presets.

Die Container-Welt (Arten, Streuungen, Kundeneffekte) übernimmt die Parameter des synthetischen Generators der Nachbar-Demo `verweilzeit-demo`
(`vwz_constants.py`, `vwz_generator.log_median_and_s`): kopiert und mit Präfix umbenannt, nicht importiert. Alle Merkmale, Verteilungen und Größenordnungen
sind **erfunden** (synthetisch, kein Echtdatenbezug)."""

# --- Welten ----------------------------------------------------------------------------------------------------------------------------
WELT_HAND, WELT_A, WELT_B, WELT_C8, WELT_C11 = "hand", "a", "b", "c8", "c11"
WELTEN = (WELT_HAND, WELT_C8, WELT_C11, WELT_A, WELT_B)
WELT_LABELS = {
    WELT_HAND: "Handbeispiel (zehn Punkte, zwei Gruppen)",
    WELT_C8: "Container (8 Parameter, ein Wochenend-Koeffizient)",
    WELT_C11: "Container (11 Parameter, Wochenende je Art)",
    WELT_A: "Rohskala A (Streuung wächst linear mit x)",
    WELT_B: "Rohskala B (Streuung quadratisch, Gerade angepasst)",
}
WELT_SHORT = {WELT_HAND: "Handbeispiel", WELT_C8: "Container, 8 Parameter", WELT_C11: "Container, 11 Parameter", WELT_A: "Rohskala A", WELT_B: "Rohskala B"}
WELT_GRUPPEN = {WELT_HAND: "Gruppe", WELT_C8: "Art", WELT_C11: "Art", WELT_A: "Drittel von x", WELT_B: "Drittel von x"}
IS_CONTAINER = (WELT_C8, WELT_C11)

# Quantilstufen und Trainingsmengen, die der Regler anbietet; für alle (Welt, n, tau) dieser Kombinationen gibt es eine vorgerechnete Messung.
TAUS = (0.1, 0.5, 0.8, 0.9, 0.95, 0.99)
N_OPTIONS = {WELT_HAND: (10,), WELT_C8: (300, 3000), WELT_C11: (300, 3000), WELT_A: (30, 50, 100, 200, 1000), WELT_B: (200, 1000)}
# Wiederholungen der vorgerechneten Messung (Zähleinheit: eine frische Trainingsstichprobe); bei kleinen Mengen mehr, weil sie schneller zu rechnen sind.
REPS = {
    (WELT_A, 30): 400, (WELT_A, 50): 400, (WELT_A, 100): 400, (WELT_A, 200): 400, (WELT_A, 1000): 150,
    (WELT_B, 200): 100, (WELT_B, 1000): 100,
    (WELT_C8, 300): 100, (WELT_C8, 3000): 40, (WELT_C11, 300): 100, (WELT_C11, 3000): 40,
}
SEED_BASE = {WELT_A: 1_520_000_000, WELT_B: 1_540_000_000, WELT_C8: 1_530_000_000, WELT_C11: 1_530_000_000}   # die zwei Container-Modelle sehen dieselben Trainingsstichproben
TEST_SEED = 424242                    # feste Testmenge der Container-Welt
TEST_N = 20_000

# --- Welt A und B (Rohskala): y = 1 + 2x + sd(x)·eps, x ~ Gleich(0, 10), eps normal -----------------------------------------------------------
X_LO, X_HI = 0.0, 10.0
A_INTERCEPT, A_SLOPE = 1.0, 2.0
SD_A = (0.2, 0.4)                     # sd(x) = 0,2 + 0,4·x (Welt A)
SD_B = (0.2, 0.04)                    # sd(x) = 0,2 + 0,04·x² (Welt B)
N_GRID = 20                           # Testgitter x = 0,25 … 9,75 (20 Punkte); Gruppen unten / Mitte / oben mit 7 / 6 / 7 Punkten
GRID_LO, GRID_HI = 0.25, 9.75
GROUP_AB = ("unten (x < 3,3)", "Mitte", "oben (x ≥ 6,7)")
S2_A_LIMIT_NUM, S2_A_LIMIT_DEN = 463, 75   # s² = E[sd(x)²] = 463/75 für Welt A (Grenzwert der Reststreuung von Kleinste Quadrate)

# --- Container-Welt (aus dem Generator der verweilzeit-demo) ----------------------------------------------------------------------------------
ART_LABELS = ("Import", "Export", "Umschlag", "Leer")
ART_P = (0.35, 0.30, 0.25, 0.10)      # Anteile der Containerarten
K_KUNDEN = 12                         # Kunden mit je einem Effekt auf die log-Verweilzeit (gehen in die Anpassung NICHT ein)
U_SCALE = (1.0, 0.3, 0.3, 1.0)        # Kundeneffekt je Art
S_BASE = (0.55, 0.25, 0.35, 0.90)     # Streuung der log-Verweilzeit je Art
P_ZOLL = 0.25                         # Zollhalt bei Import
# Kundeneffekte: numpy.random.default_rng(12345).normal(0, 0,3, 12) wie in `vwz_generator.kunden_effekte`, hier fest eingetragen (ein Test vergleicht sie mit numpy)
UK = (-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927, -0.02260299210315629, -0.2222653956256827,
      -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.5858589189036569, 0.7042228963136555, 0.2905490717255771)
UK_SEED, UK_SD = 12345, 0.30
TAGE_HISTORIE = 7000.0                # Ankunftstag gleichverteilt über 7000 Tage (daraus das Wochenende)
N_PARAM = {WELT_C8: 8, WELT_C11: 11}
COEF_NAMES_8 = ("Achsenabschnitt", "Export", "Umschlag", "Leer", "Zollhalt", "Wochenende", "Vorlauf (Export)", "Vorlauf (Umschlag)")
COEF_NAMES_11 = COEF_NAMES_8 + ("Wochenende × Export", "Wochenende × Umschlag", "Wochenende × Leer")
VORLAUF_REF = (0.0, 5.5, 3.25, 0.0)   # Bezugscontainer je Art für die Abbildung: Werktag, ohne Zollhalt, mittlerer Vorlauf

# --- Auswertung ------------------------------------------------------------------------------------------------------------------------------
TOL_OK = 0.03                         # größte Abweichung einer Gruppe vom Soll, bis zu der „trifft“ gilt
TOL_FAIL = 0.05                       # darüber gilt „trifft nicht“
UNDER_GAP = 0.05                      # Gesamtüberdeckung „zu niedrig“: um mehr als so viel unter tau
GROUP_UNDER_GAP = 0.10                # eine Gruppe „deutlich daneben“: um mehr als so viel unter tau
MEAN_GAP = 0.02                       # Gesamtüberdeckung der Quantilregression bei der kleinsten Menge um mehr als so viel neben tau: Warnung
SCATTER_MAX = 400                     # höchstens so viele Punkte im Streudiagramm
CROSS_EDGES = (X_LO, X_HI)            # Randstellen, an denen das Kreuzen der Quantilgeraden geprüft wird
RES_ZERO = 1e-7                       # Residuum „auf der Geraden“ in den Tests der Optimalitätsbedingung

# --- Darstellung -----------------------------------------------------------------------------------------------------------------------------
COLOR_OLS1 = "#c77700"
COLOR_OLSG = "#7a3fb0"
COLOR_QR = "#2e7d4f"
COLOR_TRUE = "#5b6470"
COLOR_POINTS = "rgba(120,120,120,0.45)"
METHOD_LABELS = {
    "ols1": "Kleinste Quadrate ± z·s (ein s)",
    "olsg": "Kleinste Quadrate ± z·s je Art",
    "qr": "Quantilregression",
    "ols1_roh": "Kleinste Quadrate roh ± z·s (ein s)",
    "qr_roh": "Quantilregression roh",
}
METHOD_COLORS = {"ols1": COLOR_OLS1, "olsg": COLOR_OLSG, "qr": COLOR_QR, "ols1_roh": "#e3b778", "qr_roh": "#8fc4a3"}
METHODS_BY_WELT = {
    WELT_HAND: ("ols1", "qr"),
    WELT_A: ("ols1", "qr"),
    WELT_B: ("ols1", "qr"),
    WELT_C8: ("ols1", "olsg", "qr", "ols1_roh", "qr_roh"),
    WELT_C11: ("ols1", "olsg", "qr"),
}
LIVE_METHODS = {WELT_HAND: ("ols1", "qr"), WELT_A: ("ols1", "qr"), WELT_B: ("ols1", "qr"), WELT_C8: ("ols1", "olsg", "qr"), WELT_C11: ("ols1", "olsg", "qr")}
CHART_HEIGHT = 380
FROZEN_FILE = "que_frozen.json"

# --- Handbeispiel: zwei Gruppen mit je fünf Beobachtungen (Brüche gleich im Code), tau = 4/5 im Preset ---------------------------------------
HAND_G0 = (2, 3, 4, 5, 6)
HAND_G1 = (0, 4, 8, 12, 16)

# --- Regler und Presets ----------------------------------------------------------------------------------------------------------------------
SEED_MAX, DEFAULT_SEED = 999, 7
DEFAULT_WELT, DEFAULT_TAU, DEFAULT_N = WELT_A, 0.9, 200

PRESETS = {
    "Rohskala, Streuung wächst (Voreinstellung)": {"welt": WELT_A, "tau": 0.9, "n": 200, "seed": DEFAULT_SEED},
    "Von Hand (zehn Punkte)": {"welt": WELT_HAND, "tau": 0.8, "n": 10, "seed": DEFAULT_SEED},
    "Container, 8 Parameter": {"welt": WELT_C8, "tau": 0.9, "n": 3000, "seed": DEFAULT_SEED},
    "Container, Wochenende je Art": {"welt": WELT_C11, "tau": 0.9, "n": 3000, "seed": DEFAULT_SEED},
    "Wenig Daten (n = 30)": {"welt": WELT_A, "tau": 0.9, "n": 30, "seed": DEFAULT_SEED},
    "Hohe Stufe (τ = 0,99)": {"welt": WELT_A, "tau": 0.99, "n": 30, "seed": DEFAULT_SEED},
    "Falsche Klasse (Rohskala B)": {"welt": WELT_B, "tau": 0.9, "n": 1000, "seed": DEFAULT_SEED},
    "Ohne Effekt (τ = 0,5)": {"welt": WELT_A, "tau": 0.5, "n": 200, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Rohskala, Streuung wächst (Voreinstellung)": "Rohskala A, n = 200, τ = 0,9: die Streuung wächst mit x. Das Band „Vorhersage ± z·s“ trifft unten fast alles und oben deutlich zu wenig.",
    "Von Hand (zehn Punkte)": "Zwei Gruppen mit je fünf Punkten, τ = 0,8: das Band mit einem gemeinsamen s deckt 5 von 5 und 3 von 5, zusammen 8 von 10.",
    "Container, 8 Parameter": "Verweilzeit-Container, n = 3 000, τ = 0,9, Modell mit einem Wochenend-Koeffizienten: ein gemeinsames s trifft die Leer-Container nicht.",
    "Container, Wochenende je Art": "Wie oben, aber mit einem Wochenend-Koeffizienten je Art (11 Parameter): das Wochenende wirkt im Generator nur bei Import und Leer.",
    "Wenig Daten (n = 30)": "Rohskala A, nur 30 Beobachtungen: die Quantilregression trifft im Mittel, streut aber stark von Stichprobe zu Stichprobe.",
    "Hohe Stufe (τ = 0,99)": "Rohskala A, n = 30, τ = 0,99: oberhalb von τ = 0,9 braucht die Quantilregression sehr viel mehr Daten.",
    "Falsche Klasse (Rohskala B)": "Die Streuung wächst quadratisch, angepasst wird eine Gerade: auch die Quantilregression trifft die obere Gruppe nicht.",
    "Ohne Effekt (τ = 0,5)": "Beim Median fällt „Vorhersage ± z·s“ auf die Vorhersage zusammen (z = 0): beide Verfahren treffen in jeder Gruppe, hier ist nichts zu sehen.",
}
