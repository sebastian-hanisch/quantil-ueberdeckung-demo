"""Welten der Demo: Rohskala A und B (Streuung wächst mit x), Container-Verweilzeit und das Handbeispiel.

Alle Zufallszahlen kommen aus dem ganzzahligen SplitMix64 (`que_rng`), nie aus Python-`hash()`, `random` oder `numpy.random`.
Container-Welt: Nachbau des synthetischen Generators der Nachbar-Demo `verweilzeit-demo` (`vwz_generator.py`): dieselben Anteile, Streuungen, Kundeneffekte und
Mittelfunktionen (`log_median_and_s` ist kopiert und mit Präfix umbenannt), aber mit dem exakten Zufallsstrom dieser Demo statt `numpy.random.default_rng`; die Zahlen
der Nachbar-Demo sind deshalb nicht Zahl für Zahl dieselben. Das Rauschmerkmal des Generators geht in keine Anpassung ein und fehlt hier. Alles ist erfunden (synthetisch)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import que_constants as C
import que_rng as R

UK = np.array(C.UK)
ART_CUM = np.cumsum(C.ART_P)


def sub_seed(seed: int, k: int) -> int:
    """Unabhängiger Teilstrom k zu einem Zufalls-Seed: der erste Ausgabewert von SplitMix64 mit Zustand seed·16 + k (ganzzahlig, plattformfest)."""
    return int(R.splitmix_array((int(seed) * 16 + int(k)) & R.MASK, 1)[0])


def rep_seed(welt: str, n: int, rep: int) -> int:
    """Zufalls-Seed der Wiederholung `rep` einer vorgerechneten Messung. Die beiden Container-Modelle teilen die Stichproben (gleiche Basis)."""
    return C.SEED_BASE[welt] + 100_000 * int(n) + int(rep)


# --- Rohskala A und B --------------------------------------------------------------------------------------------------------------------------

def mu_ab(x):
    return C.A_INTERCEPT + C.A_SLOPE * np.asarray(x, float)


def sd_ab(welt: str, x):
    x = np.asarray(x, float)
    if welt == C.WELT_A:
        return C.SD_A[0] + C.SD_A[1] * x
    if welt == C.WELT_B:
        return C.SD_B[0] + C.SD_B[1] * x * x
    raise ValueError(f"Welt {welt} hat keine Streuungsfunktion in x")


def grid_ab() -> np.ndarray:
    return np.linspace(C.GRID_LO, C.GRID_HI, C.N_GRID)


def groups_ab() -> list[np.ndarray]:
    """Index der Gitterpunkte in den Dritteln unten / Mitte / oben (7 / 6 / 7 Punkte)."""
    xg = grid_ab()
    third = (C.X_HI - C.X_LO) / 3
    return [np.where(xg < C.X_LO + third)[0], np.where((xg >= C.X_LO + third) & (xg < C.X_LO + 2 * third))[0], np.where(xg >= C.X_LO + 2 * third)[0]]


def draw_ab(welt: str, n: int, seed: int):
    """Eine Stichprobe: Entwurfsmatrix (1, x), Zielgröße y, Merkmal x. Der Zufallsstrom von x und von eps ist je ein eigener Teilstrom des Zufalls-Seeds."""
    if n < 1:
        raise ValueError("n muss mindestens 1 sein")
    x = C.X_LO + (C.X_HI - C.X_LO) * R.uniform_array(sub_seed(seed, 1), n)
    eps = R.normal_array(sub_seed(seed, 2), n)
    y = mu_ab(x) + sd_ab(welt, x) * eps
    return np.column_stack([np.ones(n), x]), y, x


# --- Container-Welt ----------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Features:
    art: np.ndarray        # 0 Import, 1 Export, 2 Umschlag, 3 Leer
    kunde: np.ndarray
    zoll: np.ndarray       # 1 = Zollhalt (nur Import)
    we: np.ndarray         # 1 = Ankunft am Wochenende
    vorlauf: np.ndarray    # Tage bis zur geplanten Abfahrt (Export und Umschlag), sonst 0

    @property
    def n(self) -> int:
        return len(self.art)


def draw_features(n: int, seed: int) -> Features:
    """n Container: je Container fünf Gleichverteilte (Art, Kunde, Zollhalt, Ankunftstag, Vorlauf), Zeile für Zeile aus einem Strom (die ersten k Container hängen nicht von n ab)."""
    if n < 1:
        raise ValueError("n muss mindestens 1 sein")
    u = R.uniform_array(sub_seed(seed, 1), 5 * n).reshape(n, 5)
    art = np.minimum(np.searchsorted(ART_CUM, u[:, 0]), 3)
    kunde = np.minimum((u[:, 1] * C.K_KUNDEN).astype(int), C.K_KUNDEN - 1)
    zoll = ((u[:, 2] < C.P_ZOLL) & (art == 0)).astype(int)
    tday = u[:, 3] * C.TAGE_HISTORIE
    we = ((np.floor(tday).astype(int) % 7) >= 5).astype(int)
    vorlauf = np.where(art == 1, 1.0 + 9.0 * u[:, 4], np.where(art == 2, 0.5 + 5.5 * u[:, 4], 0.0))
    return Features(art, kunde, zoll, we, vorlauf)


def log_median_and_s(f: Features):
    """Bedingte Verteilung der log-Verweilzeit, N(mu, s²): der Orakel-Wert des Generators. Kopie von `vwz_generator.log_median_and_s` (mit der Struktur Features)."""
    art = f.art
    u = UK[f.kunde] * np.take(C.U_SCALE, art)
    med = np.select(
        [art == 0, art == 1, art == 2],
        [3.5 * np.exp(0.7 * f.zoll + 0.30 * f.we), 0.6 + 0.85 * f.vorlauf, 0.3 + 0.8 * f.vorlauf],
        default=7.0 * np.exp(0.30 * f.we),
    )
    return np.log(med) + u, np.take(C.S_BASE, art).astype(float)


def draw_history(n: int, seed: int):
    """Trainingsdaten: Merkmale, Orakel (mu, s) und die Verweilzeit in Tagen = exp(mu + s·z) mit z standardnormal (eigener Teilstrom)."""
    f = draw_features(n, seed)
    mu, s = log_median_and_s(f)
    z = R.normal_array(sub_seed(seed, 3), n)
    return f, mu, s, np.exp(mu + s * z)


@lru_cache(maxsize=1)
def holdout_set():
    """Feste Testmenge von 20 000 Containern (Merkmale und Orakel (mu, s)); die Überdeckung wird exakt aus (mu, s) gerechnet, nicht aus Zählungen von Testwerten."""
    f = draw_features(C.TEST_N, C.TEST_SEED)
    mu, s = log_median_and_s(f)
    return f, mu, s


def design_c8(f: Features) -> np.ndarray:
    """8 Parameter: Achsenabschnitt (Import), Export, Umschlag, Leer, Zollhalt, Wochenende, Vorlauf nur bei Export, Vorlauf nur bei Umschlag."""
    a = f.art
    return np.column_stack([np.ones(f.n), a == 1, a == 2, a == 3, f.zoll, f.we, f.vorlauf * (a == 1), f.vorlauf * (a == 2)]).astype(float)


def design_c11(f: Features) -> np.ndarray:
    """11 Parameter: die 8 und ein zusätzlicher Wochenend-Koeffizient je Art (Export, Umschlag, Leer als Abweichung von Import)."""
    a = f.art
    return np.column_stack([design_c8(f)] + [f.we * (a == k) for k in (1, 2, 3)]).astype(float)


def design(welt: str, f: Features) -> np.ndarray:
    if welt == C.WELT_C8:
        return design_c8(f)
    if welt == C.WELT_C11:
        return design_c11(f)
    raise ValueError(f"Welt {welt} ist keine Container-Welt")


def art_groups(f: Features) -> list[np.ndarray]:
    return [np.where(f.art == k)[0] for k in range(4)]


def cell_groups(f: Features) -> list[np.ndarray]:
    """Zellen Art × Wochenende (acht): Import/Werktag, Import/Wochenende, Export/Werktag, …"""
    return [np.where((f.art == a) & (f.we == w))[0] for a in range(4) for w in range(2)]


def reference_rows(welt: str):
    """Je Art ein Bezugscontainer (Werktag, ohne Zollhalt, mittlerer Vorlauf) als Zeile der Entwurfsmatrix, dazu sein (mu, s) ohne Kundeneffekt (Kunde mit Effekt 0 gibt es nicht: der Effekt wird weggelassen)."""
    f = Features(np.arange(4), np.zeros(4, int), np.zeros(4, int), np.zeros(4, int), np.array(C.VORLAUF_REF, float))
    return design(welt, f)


# --- Handbeispiel ------------------------------------------------------------------------------------------------------------------------------

def hand_design():
    """Zehn Punkte, zwei Gruppen: Entwurfsmatrix (1, Gruppe), Zielgröße y und Gruppenindex."""
    y = np.array(C.HAND_G0 + C.HAND_G1, float)
    g = np.r_[np.zeros(5), np.ones(5)]
    return np.column_stack([np.ones(10), g]), y, g.astype(int)
