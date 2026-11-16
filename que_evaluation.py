"""Auswertung: eine Stichprobe anpassen und exakt auswerten (live), Wiederholungen zusammenfassen (vorgerechnet und eingefroren in `que_frozen.json`) und die Verdikte.

Zähleinheit überall: die Wiederholung (eine frische Trainingsstichprobe, ein Satz angepasster Quantile). Überdeckung wird exakt aus der wahren bedingten
Verteilungsfunktion gerechnet (Rohskala: Normal auf einem 20-Punkte-Gitter; Container: lognormal mit (mu, s) des Generators auf 20 000 Testcontainern), nicht aus Zählungen
von Testwerten. Jedes Verdikt der App ist eine reine Funktion `verdict_*(...) -> (Art, Text)` (DEMO-PLAYBOOK §12)."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

import que_algorithm as A
import que_constants as C
import que_scenario as S

ROOT = Path(__file__).resolve().parent


# --- Anzeige -----------------------------------------------------------------------------------------------------------------------------------

def de(x: float, nd: int = 3, sign: bool = False) -> str:
    """Zahl mit Dezimalkomma und Minuszeichen (Anzeige); nan als „–“."""
    if x != x:
        return "–"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def pct(x: float, nd: int = 1) -> str:
    """Anteil als Prozentzahl mit Komma: 0,307 -> „30,7 %“."""
    return de(100.0 * x, nd) + " %"


def de_int(x: int) -> str:
    return f"{int(x):,}".replace(",", " ")


def group_names(welt: str) -> tuple[str, ...]:
    return C.ART_LABELS if welt in C.IS_CONTAINER else C.GROUP_AB


def tau_text(tau: float) -> str:
    return de(tau, 2 if round(tau * 100) % 10 else 1)


# --- Einstellungen -------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    welt: str = C.DEFAULT_WELT
    tau: float = C.DEFAULT_TAU
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED


def nearest(options, value):
    return min(options, key=lambda o: (abs(o - value), o))


def sanitize(welt: str, tau: float, n: int, seed: int) -> Settings:
    """Auf gültige Werte begrenzen: unbekannte Welt -> Voreinstellung, tau und n auf den nächsten angebotenen Wert (nur für diese gibt es vorgerechnete Messungen)."""
    if welt not in C.WELTEN:
        welt = C.DEFAULT_WELT
    return Settings(welt, float(nearest(C.TAUS, tau)), int(nearest(C.N_OPTIONS[welt], n)), int(min(max(seed, 0), C.SEED_MAX)))


# --- Eine Stichprobe anpassen und exakt auswerten --------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _holdout_groups():
    f, mu, s = S.holdout_set()
    return S.art_groups(f), S.cell_groups(f)


def _record(c: np.ndarray, groups, cells=None) -> dict:
    return {"total": float(np.mean(c)), "groups": A.group_means(c, groups), "cells": A.group_means(c, cells) if cells is not None else None}


def fit_cover(welt: str, n: int, seed: int, tau: float, methods=None, cross: bool = False, plot: bool = False) -> dict:
    """Eine Stichprobe (Welt, n, Zufalls-Seed) anpassen und für jedes Verfahren die exakte Überdeckung rechnen.

    Rückgabe: {"methods": {Name: {"total", "groups", "cells"}}, "cross": bool oder None, "plot": Daten für die Abbildung oder None}.
    Container: Band auf der Log-Skala (Namen ols1, olsg, qr) und, wenn verlangt, auf der Rohskala (ols1_roh, qr_roh)."""
    z = A.z_of(tau)
    methods = tuple(methods) if methods is not None else C.LIVE_METHODS[welt]
    if welt in (C.WELT_A, C.WELT_B):
        X, y, x = S.draw_ab(welt, n, seed)
        xg = S.grid_ab()
        Xg = np.column_stack([np.ones(len(xg)), xg])
        mu, sd = S.mu_ab(xg), S.sd_ab(welt, xg)
        groups = S.groups_ab()
        b, r = A.ols_fit(X, y)
        s = A.sigma_hat(r, 2)
        bq, _ = A.qr_fit(X, y, tau)
        preds = {"ols1": Xg @ b + z * s, "qr": Xg @ bq}
        out = {m: _record(A.cover_normal(preds[m], mu, sd), groups) for m in methods}
        crossed = None
        if cross and tau > 0.5:
            b_low, _ = A.qr_fit(X, y, 1.0 - tau)
            crossed = A.crossing(b_low, bq, C.CROSS_EDGES)
        info = {"x": x, "y": y, "b_ols": b, "b_qr": bq, "s": s, "xg": xg, "pred": preds} if plot else None
        return {"methods": out, "cross": crossed, "plot": info}
    if welt in C.IS_CONTAINER:
        f, mu_h, s_h, d = S.draw_history(n, seed)
        ly = np.log(d)
        X = S.design(welt, f)
        p = X.shape[1]
        ft, mu_t, s_t = S.holdout_set()
        Xt = S.design(welt, ft)
        art_groups, cell_groups = _holdout_groups()
        b, r = A.ols_fit(X, ly)
        s1 = A.sigma_hat(r, p)
        out = {}
        pred_log = {}
        if "ols1" in methods:
            pred_log["ols1"] = Xt @ b + z * s1
        if "olsg" in methods:
            sa = A.sigma_by_group(r, f.art, 4, s1)
            pred_log["olsg"] = Xt @ b + z * sa[ft.art]
        if "qr" in methods:
            bq, _ = A.qr_fit(X, ly, tau)
            pred_log["qr"] = Xt @ bq
        for m, q in pred_log.items():
            out[m] = _record(A.cover_lognormal_log(q, mu_t, s_t), art_groups, cell_groups)
        if "ols1_roh" in methods:
            bo, ro = A.ols_fit(X, d)
            so = A.sigma_hat(ro, p)
            out["ols1_roh"] = _record(A.cover_lognormal_raw(Xt @ bo + z * so, mu_t, s_t), art_groups, cell_groups)
        if "qr_roh" in methods:
            bqr, _ = A.qr_fit(X, d, tau)
            out["qr_roh"] = _record(A.cover_lognormal_raw(Xt @ bqr, mu_t, s_t), art_groups, cell_groups)
        out = {m: out[m] for m in methods if m in out}
        info = None
        if plot:
            Xr = S.reference_rows(welt)
            ref = {"ols1": Xr @ b + z * s1}
            if "olsg" in methods:
                ref["olsg"] = Xr @ b + z * np.array(sa)
            if "qr" in methods:
                ref["qr"] = Xr @ bq
            info = {"art": f.art[: C.SCATTER_MAX * 3], "dwell": d[: C.SCATTER_MAX * 3], "ref": ref, "s1": s1, "sa": np.array(sa) if "olsg" in methods else None}
        return {"methods": out, "cross": None, "plot": info}
    raise ValueError(f"Welt {welt} wird nicht über Stichproben ausgewertet")


# --- Wiederholungen zusammenfassen ----------------------------------------------------------------------------------------------------------

def aggregate(results: list[dict], tau: float, methods) -> dict:
    """Fasst die Ergebnisse von `fit_cover` über die Wiederholungen zusammen: Gruppenmittel, Gesamtüberdeckung (Mittel und 5./50./95. Perzentil), Anteil der Wiederholungen
    mit Gesamtüberdeckung um mehr als UNDER_GAP unter tau, Mittel des kleinsten Gruppenwerts, Anteil mit einer Gruppe um mehr als GROUP_UNDER_GAP unter tau, bei Container-Zellen
    Art × Wochenende deren Mittel und größte Abweichung von tau; dazu der Anteil der Wiederholungen mit Kreuzen."""
    out = {"reps": len(results), "methods": {}}
    for m in methods:
        G = np.array([r["methods"][m]["groups"] for r in results], float)
        T = np.array([r["methods"][m]["total"] for r in results], float)
        pc = np.percentile(T, [5, 50, 95])
        entry = {
            "groups": [round(float(v), 4) for v in G.mean(axis=0)],
            "total": round(float(T.mean()), 4),
            "p5": round(float(pc[0]), 4), "p50": round(float(pc[1]), 4), "p95": round(float(pc[2]), 4),
            "share_under": round(float(np.mean(T < tau - C.UNDER_GAP)), 4),
            "min_group": round(float(G.min(axis=1).mean()), 4),
            "share_group_under": round(float(np.mean(G.min(axis=1) < tau - C.GROUP_UNDER_GAP)), 4),
        }
        if results[0]["methods"][m]["cells"] is not None:
            Cm = np.array([r["methods"][m]["cells"] for r in results], float).mean(axis=0)
            entry["cells"] = [round(float(v), 4) for v in Cm]
            entry["max_cell_dev"] = round(float(np.max(np.abs(Cm - tau))), 4)
        out["methods"][m] = entry
    crosses = [r["cross"] for r in results if r["cross"] is not None]
    out["cross"] = round(float(np.mean(crosses)), 4) if crosses else None
    return out


def cell_key(welt: str, n: int, tau: float) -> str:
    return f"{welt}|{int(n)}|{tau:.2f}"


def compute_cell(welt: str, n: int, taus, reps: int | None = None) -> dict:
    """Wiederholungen für alle `taus` einer (Welt, n)-Zelle neu rechnen (langsam: wird von tools/que_make_frozen.py benutzt). Die Wiederholung i nimmt den Zufalls-Seed
    rep_seed(welt, n, i); alle taus sehen dieselben Stichproben. Container: die Rohskala-Verfahren nur im Modell mit 8 Parametern."""
    reps = C.REPS[(welt, n)] if reps is None else reps
    methods = C.METHODS_BY_WELT[welt]
    res = {float(t): [] for t in taus}
    for i in range(reps):
        seed = S.rep_seed(welt, n, i)
        for t in taus:
            res[float(t)].append(fit_cover(welt, n, seed, float(t), methods=methods, cross=welt == C.WELT_A))
    return {cell_key(welt, n, t): aggregate(r, t, methods) for t, r in res.items()}


@lru_cache(maxsize=1)
def frozen_data() -> dict:
    """Die vorgerechnete Messung (que_frozen.json); fehlt sie, ist das ein Fehler im Repo, keine stille Lücke."""
    return json.loads((ROOT / C.FROZEN_FILE).read_text(encoding="utf-8"))


def frozen_cell(welt: str, n: int, tau: float) -> dict | None:
    """Zelle der vorgerechneten Messung oder None (Handbeispiel und Kombinationen außerhalb des Rasters)."""
    return frozen_data()["cells"].get(cell_key(welt, n, tau))


# --- Handbeispiel --------------------------------------------------------------------------------------------------------------------------------

def hand(tau: float) -> dict:
    return A.hand_analysis(C.HAND_G0, C.HAND_G1, tau)




# --- Verdikte (reine Funktionen) -----------------------------------------------------------------------------------------------------------------

def _dev(entry: dict, tau: float) -> float:
    """Größte Abweichung einer Gruppe vom Soll, auf 6 Stellen gerundet (die Schwellen gelten inklusive: 0,93 gegen 0,9 ist genau 0,03)."""
    return round(max(abs(g - tau) for g in entry["groups"]), 6)


def _groups_text(entry: dict, welt: str) -> str:
    return " / ".join(de(v) for v in entry["groups"])


def verdict_hand(h: dict) -> tuple[str, str]:
    """Handbeispiel: Zählung im Training. Ein Band mit gemeinsamem s deckt die Gruppen verschieden, obwohl es im Mittel das Soll trifft."""
    t = float(h["tau"])
    c0, c1 = h["ols_cov"]
    n0, n1 = h["n_group"]
    total = c0 + c1
    q = h["qr"]
    qr_lo = f"{q[0]['cov_lo']} von {n0} und {q[1]['cov_lo']} von {n1}"
    flat = [i for i, e in enumerate(q) if e["flat"]]
    flat_txt = ""
    if flat:
        spans = " und ".join(f"[{de(float(q[i]['lo']), 0)}; {de(float(q[i]['hi']), 0)}]" for i in flat)
        flat_txt = f" Der Pinball-Verlust ist bei τ = {tau_text(t)} auf {spans} flach (n·τ ganzzahlig): die Überdeckung im Training ist eine Treppe, und am oberen Streckenende wären es mehr."
    text = (
        f"τ = {tau_text(t)}: das Band „Gruppenmittel + z·s“ mit einem gemeinsamen s = {de(h['s'], 2)} (Grenzen {de(h['upper'][0], 2)} und {de(h['upper'][1], 2)}) deckt im Training "
        f"{c0} von {n0} Punkten in Gruppe 0 und {c1} von {n1} in Gruppe 1, zusammen {total} von {n0 + n1} ({de(total / (n0 + n1), 1)}). "
        f"Die Quantilregression (je Gruppe der Pinball-Minimierer, am unteren Streckenende) deckt {qr_lo}.{flat_txt}"
    )
    if abs(c0 - c1) / max(n0, n1) >= 0.4:
        return "warning", "Im Mittel stimmt es, je Gruppe nicht. " + text
    return "info", "Hier fallen beide Gruppen nicht auseinander. " + text


def _range_text(entry: dict) -> str:
    return " bis ".join([de(min(entry["groups"])), de(max(entry["groups"]))])


def verdict_coverage(welt: str, tau: float, n: int, cell: dict | None) -> tuple[str, str]:
    """Kernfrage 1: Trifft das Band in jeder Gruppe? Die Aussage folgt der vorgerechneten Messung (Mittel über die Wiederholungen dieser Zelle).
    Gruppenwerte: Überdeckung gemittelt über die Gruppe; Soll = tau. Bewertet wird die größte Abweichung einer Gruppe vom Soll (TOL_OK, TOL_FAIL)."""
    if cell is None:
        return "info", f"Für {C.WELT_SHORT[welt]} mit n = {de_int(n)} und τ = {tau_text(tau)} gibt es keine vorgerechnete Messung."
    m = cell["methods"]
    reps = cell["reps"]
    head = f"{C.WELT_SHORT[welt]}, τ = {tau_text(tau)}, n = {de_int(n)} (Mittel über {reps} Wiederholungen; Gruppen {' / '.join(group_names(welt))}): "
    o, q = m["ols1"], m["qr"]
    g = m.get("olsg")
    d_o, d_q = _dev(o, tau), _dev(q, tau)
    d_g = _dev(g, tau) if g else None
    ols_txt = f"Kleinste Quadrate ± z·s mit einem s {_groups_text(o, welt)} (gesamt {de(o['total'])})"
    qr_txt = f"Quantilregression {_groups_text(q, welt)} (gesamt {de(q['total'])})"
    if d_o <= C.TOL_OK and d_q <= C.TOL_OK:
        return "info", head + f"beide Verfahren treffen in jeder Gruppe (größte Abweichung vom Soll {de(max(d_o, d_q))}): {ols_txt}; {qr_txt}. Hier ist keine Abweichung zu sehen."
    if d_o > C.TOL_FAIL:
        mean_ok = abs(o["total"] - tau) <= C.TOL_OK
        text = head + f"{ols_txt}: " + (f"die Gesamtüberdeckung liegt nahe am Soll (Abweichung höchstens {de(C.TOL_OK, 2)}), in den Gruppen nicht. " if mean_ok
                                         else "weder im Mittel noch in den Gruppen. ")
        if d_q <= C.TOL_FAIL:
            text += f"{qr_txt}: die größte Abweichung einer Gruppe vom Soll ist {de(d_q)}."
        else:
            text += f"Auch die Quantilregression trifft nicht: {qr_txt}; die größte Abweichung einer Gruppe vom Soll ist {de(d_q)}."
        if g is not None and d_g <= C.TOL_OK and d_q <= C.TOL_OK:
            text += f" Mit einem s je Art ({_groups_text(g, welt)}) trifft schon Kleinste Quadrate in jeder Art: hier ist die Quantilregression kein Gewinn, sondern ein Umweg (gilt, wenn die Streuung einer bekannten Gruppe folgt)."
        if welt == C.WELT_A and tau != 0.5:
            lim = A.ols_limit_coverage_a(tau, S.grid_ab(), lambda x: S.sd_ab(C.WELT_A, x))
            lg = [float(np.mean(lim[i])) for i in S.groups_ab()]
            if max(abs(v - tau) for v in lg) > C.TOL_FAIL:
                text += f" Grenzwert für große n (Rechnung): {' / '.join(de(v) for v in lg)}: der Fehler des Bandes ist systematisch, kein Stichprobenrauschen."
        return "warning", text
    if d_q > C.TOL_FAIL:
        return "warning", head + f"{qr_txt}: Die Quantilregression trifft in mindestens einer Gruppe nicht (größte Abweichung {de(d_q)}); {ols_txt}."
    return "info", head + f"{ols_txt}; {qr_txt}: die größte Abweichung einer Gruppe vom Soll ist {de(d_o)} bzw. {de(d_q)}, weder klar daneben noch klar getroffen."


def verdict_amount(welt: str, tau: float, cells: list[tuple[int, dict]]) -> tuple[str, str]:
    """Kernfrage 2: Was kostet es an Daten? `cells` = [(n, Zelle), …] aufsteigend in n für dieselbe Welt und dieselbe Stufe."""
    if not cells:
        return "info", "Für das Handbeispiel gibt es keine Messung über die Datenmenge: es hat genau zehn Punkte."
    (n0, c0), (n1, c1) = cells[0], cells[-1]
    q0, q1 = c0["methods"]["qr"], c1["methods"]["qr"]
    o0 = c0["methods"]["ols1"]
    text = (
        f"{C.WELT_SHORT[welt]}, τ = {tau_text(tau)}: Gesamtüberdeckung der Quantilregression {de(q0['total'])} bei n = {de_int(n0)} "
        f"(5. bis 95. Perzentil über {c0['reps']} Wiederholungen: {de(q0['p5'])} bis {de(q0['p95'])}; in {pct(q0['share_under'])} der Wiederholungen mehr als {de(C.UNDER_GAP, 2)} unter τ) und "
        f"{de(q1['total'])} bei n = {de_int(n1)} ({pct(q1['share_under'])})."
    )
    if tau >= 0.5:
        text += f" Eine Gruppe liegt bei n = {de_int(n0)} in {pct(q0['share_group_under'])} der Wiederholungen mehr als {de(C.GROUP_UNDER_GAP, 2)} unter τ (Kleinste Quadrate ± z·s: {pct(o0['share_group_under'])})."
    gap_o, gap_q = abs(o0["total"] - tau), abs(q0["total"] - tau)
    closer = " Damit liegt Kleinste Quadrate ± z·s im Mittel näher am Soll als die Quantilregression;" if gap_q - gap_o > 0.01 else (
        " Damit liegt die Quantilregression im Mittel näher am Soll;" if gap_o - gap_q > 0.01 else "")
    text += (f" Gesamt bei n = {de_int(n0)}: Kleinste Quadrate ± z·s {de(o0['total'])}, Quantilregression {de(q0['total'])}.{closer}"
             f" in den Gruppen {_groups_text(o0, welt)} gegen {_groups_text(q0, welt)}." if closer else
             f" Gesamt bei n = {de_int(n0)}: Kleinste Quadrate ± z·s {de(o0['total'])}, Quantilregression {de(q0['total'])}; in den Gruppen {_groups_text(o0, welt)} gegen {_groups_text(q0, welt)}.")
    if c0.get("cross") is not None and tau > 0.5:
        text += (f" Die Gerade der Stufe {tau_text(1 - tau)} liegt an einer der Randstellen x = 0 oder x = 10 über der Geraden der Stufe {tau_text(tau)} in {pct(c0['cross'])} der Wiederholungen bei n = {de_int(n0)}"
                 f" und in {pct(c1['cross'])} bei n = {de_int(n1)} (nur diese zwei Randstellen geprüft; einzelne Ziehung je Zelle mit Zufallsfehler).")
    kind = "warning" if abs(q0["total"] - tau) > C.MEAN_GAP else "info"
    return kind, text


def verdict_simple(c8: dict | None, c11: dict | None, tau: float, n: int) -> tuple[str, str]:
    """Kernfrage 3: Wann reicht der einfache Weg (ein s je Art)? Container, Zellen Art × Wochenende: Modell mit 8 gegen 11 Parametern."""
    if c8 is None or c11 is None:
        return "info", "Für diese Stufe oder Datenmenge liegt keine vorgerechnete Container-Messung vor."
    a8, a11 = c8["methods"], c11["methods"]
    g, q = a8["olsg"], a8["qr"]
    g11, q11 = a11["olsg"], a11["qr"]
    text = (
        f"Container, τ = {tau_text(tau)}, n = {de_int(n)} ({c8['reps']} Wiederholungen): Kleinste Quadrate ± z·s mit einem s je Art deckt je Art {_groups_text(g, C.WELT_C8)}, die Quantilregression {_groups_text(q, C.WELT_C8)}. "
        f"Die acht Zellen Art × Wochenende weichen mit einem gemeinsamen Wochenend-Koeffizienten (8 Parameter) bis {de(g['max_cell_dev'])} (s je Art) bzw. {de(q['max_cell_dev'])} (Quantilregression) vom Soll ab, "
        f"mit einem Wochenend-Koeffizienten je Art (11 Parameter) bis {de(g11['max_cell_dev'])} bzw. {de(q11['max_cell_dev'])}."
    )
    if q["max_cell_dev"] > C.TOL_OK and q["max_cell_dev"] > 2 * q11["max_cell_dev"]:
        text += f" Die Quantilregression allein behebt die Abweichung nicht: erst das fehlende Wochenend-Glied je Art senkt sie (Quantilregression von {de(q['max_cell_dev'])} auf {de(q11['max_cell_dev'])})."
    else:
        text += f" Bei dieser Stufe weicht keine Zelle um mehr als {de(C.TOL_OK, 2)} (Quantilregression, 8 Parameter) vom Soll ab oder das Modell mit 11 Parametern ändert wenig."
    return ("info" if max(_dev(g, tau), _dev(q, tau)) <= C.TOL_OK else "warning"), text


def verdict_sample(methods: dict, welt: str, tau: float) -> tuple[str, str]:
    """Hinweis zur einzelnen Stichprobe (Zufalls-Seed): die Gruppenwerte dieser einen Anpassung streuen um das Mittel der Wiederholungen."""
    names = " / ".join(group_names(welt))
    return "info", f"Diese eine Stichprobe (Gruppen {names}): " + "; ".join(
        f"{C.METHOD_LABELS[m]} {' / '.join(de(v) for v in rec['groups'])} (gesamt {de(rec['total'])})" for m, rec in methods.items()
    ) + f". Soll: {tau_text(tau)}."


# --- Tabellen (Zeilen aus Zeichenketten mit Dezimalkomma; die App macht daraus Datenrahmen) ------------------------------------------------------------

def hand_methods(h: dict) -> dict:
    """Handbeispiel im Format der Verfahren: Anteil der im Training gedeckten Punkte je Gruppe und gesamt (Quantilregression am unteren Streckenende)."""
    n0, n1 = h["n_group"]
    ols = [h["ols_cov"][0] / n0, h["ols_cov"][1] / n1]
    qr = [h["qr"][0]["cov_lo"] / n0, h["qr"][1]["cov_lo"] / n1]
    return {"ols1": {"groups": ols, "total": sum(h["ols_cov"]) / (n0 + n1), "cells": None},
            "qr": {"groups": qr, "total": (h["qr"][0]["cov_lo"] + h["qr"][1]["cov_lo"]) / (n0 + n1), "cells": None}}


def frozen_table_rows(cell: dict, welt: str, tau: float) -> list[dict]:
    """Eine Zeile je Verfahren: Gruppenmittel, gesamt, 5./95. Perzentil der Gesamtüberdeckung, Anteil mit Gesamtüberdeckung zu niedrig, kleinster Gruppenwert, Anteil mit einer Gruppe deutlich zu niedrig."""
    rows = []
    for m in C.METHODS_BY_WELT[welt]:
        e = cell["methods"][m]
        row = {"Verfahren": C.METHOD_LABELS[m]}
        for name, v in zip(group_names(welt), e["groups"]):
            row[name] = de(v)
        row["gesamt"] = de(e["total"])
        row["5. Perzentil gesamt"] = de(e["p5"])
        row["95. Perzentil gesamt"] = de(e["p95"])
        row[f"Anteil gesamt unter τ − {de(C.UNDER_GAP, 2)}"] = pct(e["share_under"])
        row["kleinster Gruppenwert (Mittel)"] = de(e["min_group"])
        row[f"Anteil mit Gruppe unter τ − {de(C.GROUP_UNDER_GAP, 1)}"] = pct(e["share_group_under"])
        rows.append(row)
    return rows


CELL_NAMES = [f"{a}, {w}" for a in C.ART_LABELS for w in ("Werktag", "Wochenende")]


def cells_table_rows(cell8: dict, cell11: dict, tau: float) -> list[dict]:
    """Zellen Art × Wochenende (Container): Überdeckung der drei Verfahren mit 8 und mit 11 Parametern, dazu die größte Abweichung vom Soll."""
    rows = []
    for i, name in enumerate(CELL_NAMES):
        row = {"Zelle": name}
        for label, cell in (("8 Parameter", cell8), ("11 Parameter", cell11)):
            for m in ("olsg", "qr"):
                row[f"{C.METHOD_LABELS[m]}, {label}"] = de(cell["methods"][m]["cells"][i])
        rows.append(row)
    last = {"Zelle": f"größte Abweichung von τ = {tau_text(tau)}"}
    for label, cell in (("8 Parameter", cell8), ("11 Parameter", cell11)):
        for m in ("olsg", "qr"):
            last[f"{C.METHOD_LABELS[m]}, {label}"] = de(cell["methods"][m]["max_cell_dev"])
    rows.append(last)
    return rows


def amount_table_rows(rows: list[tuple[int, dict]], welt: str, tau: float) -> list[dict]:
    """Eine Zeile je Trainingsmenge: Quantilregression gesamt, 5./50./95. Perzentil, Anteil zu niedrig, kleinster Gruppenwert beider Verfahren, Anteil mit einer Gruppe deutlich zu niedrig, Kreuzen."""
    out = []
    for n, c in rows:
        q, o = c["methods"]["qr"], c["methods"]["ols1"]
        row = {
            "n": de_int(n), "Wiederholungen": de_int(c["reps"]),
            "Quantilregression gesamt": de(q["total"]), "5. Perzentil": de(q["p5"]), "50. Perzentil": de(q["p50"]), "95. Perzentil": de(q["p95"]),
            f"Anteil gesamt unter τ − {de(C.UNDER_GAP, 2)}": pct(q["share_under"]),
            "kleinster Gruppenwert (Mittel), ± z·s": de(o["min_group"]), "kleinster Gruppenwert (Mittel), Quantilregression": de(q["min_group"]),
            f"Anteil mit Gruppe unter τ − {de(C.GROUP_UNDER_GAP, 1)}, ± z·s": pct(o["share_group_under"]), "… Quantilregression": pct(q["share_group_under"]),
        }
        if welt == C.WELT_A and tau > 0.5:
            row["Kreuzen an x = 0 oder 10"] = pct(c["cross"])
        out.append(row)
    return out
