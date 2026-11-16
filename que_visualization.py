"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import que_algorithm as A
import que_constants as C
import que_scenario as S
from que_evaluation import de, group_names, tau_text


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=C.CHART_HEIGHT, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      legend=dict(orientation="h", y=-0.25, x=0))
    return lock_axes(fig)


def build_scatter(plot: dict, welt: str, tau: float):
    """Rohskala: Punkte der Stichprobe, wahre τ-Quantilkurve, Band „Vorhersage ± z·s“ und Quantilgerade; senkrechte Linien trennen die drei Drittel."""
    z = A.z_of(tau)
    xs = np.linspace(C.X_LO, C.X_HI, 101)
    x, y = plot["x"][: C.SCATTER_MAX], plot["y"][: C.SCATTER_MAX]
    true_q = S.mu_ab(xs) + z * S.sd_ab(welt, xs)
    b, bq, s = plot["b_ols"], plot["b_qr"], plot["s"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=4, color=C.COLOR_POINTS), name=f"Stichprobe ({len(plot['x'])} Punkte, gezeigt {len(x)})", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=true_q, mode="lines", line=dict(color=C.COLOR_TRUE, dash="dash", width=2), name=f"wahres {tau_text(tau)}-Quantil",
                             hovertemplate="x = %{x:.1f}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=b[0] + b[1] * xs + z * s, mode="lines", line=dict(color=C.COLOR_OLS1, width=3), name=C.METHOD_LABELS["ols1"],
                             hovertemplate="x = %{x:.1f}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=bq[0] + bq[1] * xs, mode="lines", line=dict(color=C.COLOR_QR, width=3), name=C.METHOD_LABELS["qr"],
                             hovertemplate="x = %{x:.1f}: %{y:.2f}<extra></extra>"))
    for xv in (C.X_HI / 3, 2 * C.X_HI / 3):
        fig.add_vline(x=xv, line=dict(color="rgba(128,128,128,0.5)", dash="dot", width=1))
    fig.update_xaxes(title_text="Merkmal x", range=[C.X_LO, C.X_HI])
    fig.update_yaxes(title_text="Zielgröße y")
    return _base(fig)


def build_container(plot: dict, welt: str, tau: float):
    """Container: Verweilzeiten der ersten Stichprobenpunkte je Art (logarithmische Achse) und je Verfahren die obere Grenze für einen Bezugscontainer der Art."""
    art, dwell = plot["art"], plot["dwell"]
    jitter = ((np.arange(len(art)) * 0.6180339887) % 1.0 - 0.5) * 0.55
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=art + jitter, y=dwell, mode="markers", marker=dict(size=4, color=C.COLOR_POINTS), name=f"Stichprobe (gezeigt {len(art)})", hoverinfo="skip"))
    for m, logq in plot["ref"].items():
        fig.add_trace(go.Scatter(x=np.arange(4), y=np.exp(logq), mode="markers", marker=dict(symbol="line-ew", size=26, line=dict(width=4, color=C.METHOD_COLORS[m])),
                                 name=C.METHOD_LABELS[m], hovertemplate="%{y:.1f} Tage<extra></extra>"))
    fig.update_xaxes(tickmode="array", tickvals=[0, 1, 2, 3], ticktext=list(C.ART_LABELS), range=[-0.6, 3.6])
    fig.update_yaxes(title_text="Verweilzeit (Tage), logarithmische Achse", type="log")
    return _base(fig)


def build_hand(h: dict):
    """Handbeispiel: die zehn Punkte in zwei Gruppen, je Gruppe die obere Grenze „Mittel + z·s“ (orange) und die Minimierer-Strecke des Pinball-Verlusts (grün)."""
    fig = go.Figure()
    for gi, vals in enumerate((C.HAND_G0, C.HAND_G1)):
        fig.add_trace(go.Scatter(x=[gi] * len(vals), y=list(vals), mode="markers", marker=dict(size=10, color=C.COLOR_POINTS, line=dict(width=1, color="gray")),
                                 name="Beobachtungen" if gi == 0 else None, showlegend=gi == 0, hovertemplate="y = %{y}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[gi - 0.3, gi + 0.3], y=[h["upper"][gi]] * 2, mode="lines", line=dict(color=C.COLOR_OLS1, width=4),
                                 name=C.METHOD_LABELS["ols1"] if gi == 0 else None, showlegend=gi == 0, hovertemplate="%{y:.2f}<extra></extra>"))
        q = h["qr"][gi]
        fig.add_trace(go.Scatter(x=[gi + 0.15, gi + 0.15], y=[float(q["lo"]), float(q["hi"])], mode="lines+markers", line=dict(color=C.COLOR_QR, width=6), marker=dict(size=8),
                                 name=f"Quantilregression: Minimierer des Pinball-Verlusts (τ = {tau_text(float(h['tau']))})" if gi == 0 else None, showlegend=gi == 0,
                                 hovertemplate="%{y}<extra></extra>"))
    fig.update_xaxes(tickmode="array", tickvals=[0, 1], ticktext=["Gruppe 0", "Gruppe 1"], range=[-0.6, 1.6])
    fig.update_yaxes(title_text="y", range=[-1, 18])
    return _base(fig)


def build_bars(methods: dict, welt: str, tau: float, key_groups="groups", total_label="gesamt"):
    """Überdeckung je Gruppe und gesamt, je Verfahren ein Balken; die waagerechte Linie ist das Soll τ."""
    names = list(group_names(welt)) + [total_label]
    fig = go.Figure()
    for m, rec in methods.items():
        vals = list(rec[key_groups]) + [rec["total"]]
        fig.add_trace(go.Bar(x=names, y=vals, name=C.METHOD_LABELS[m], marker_color=C.METHOD_COLORS[m], text=[de(v) for v in vals], textposition="outside",
                             hovertemplate="%{x}: %{y:.3f}<extra>" + C.METHOD_LABELS[m] + "</extra>"))
    fig.add_hline(y=tau, line=dict(color="rgba(128,128,128,0.9)", dash="dash", width=2), annotation_text=f"Soll τ = {tau_text(tau)}", annotation_position="top left")
    fig.update_yaxes(title_text="Überdeckung P(y ≤ Band)", range=[0, 1.15], tickvals=[0, 0.25, 0.5, 0.75, 1.0])
    fig.update_layout(barmode="group")
    return _base(fig)


def build_amount(rows: list[tuple[int, dict]], tau: float):
    """Gesamtüberdeckung über die Trainingsmenge: Quantilregression mit 5.-bis-95.-Perzentil-Band über die Wiederholungen, Kleinste Quadrate ± z·s als Linie."""
    xs = [f"n = {n}" for n, _ in rows]
    q = [c["methods"]["qr"] for _, c in rows]
    o = [c["methods"]["ols1"] for _, c in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[e["p95"] for e in q] + [e["p5"] for e in q][::-1], fill="toself", fillcolor="rgba(46,125,79,0.18)", line=dict(width=0),
                             hoverinfo="skip", name="Quantilregression: 5. bis 95. Perzentil"))
    fig.add_trace(go.Scatter(x=xs, y=[e["total"] for e in q], mode="lines+markers", line=dict(color=C.COLOR_QR, width=3), name=C.METHOD_LABELS["qr"] + " (Mittel)",
                             hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[e["total"] for e in o], mode="lines+markers", line=dict(color=C.COLOR_OLS1, width=3), name=C.METHOD_LABELS["ols1"] + " (Mittel)",
                             hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=tau, line=dict(color="rgba(128,128,128,0.9)", dash="dash", width=2), annotation_text=f"Soll τ = {tau_text(tau)}", annotation_position="top left")
    lo = min(min(e["p5"] for e in q), min(e["total"] for e in o), tau)
    hi = max(max(e["p95"] for e in q), max(e["total"] for e in o), tau)
    fig.update_yaxes(title_text="Gesamtüberdeckung", range=[max(0.0, lo - 0.05), min(1.0, hi + 0.05)])
    fig.update_xaxes(type="category")
    return _base(fig)
