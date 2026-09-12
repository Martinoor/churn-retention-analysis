"""Run the full analysis and write figures + findings.

    python run_analysis.py --data data/train.parquet

Writes charts to figures/ and a numbers-filled summary to findings.md, so the
write-up never has to be updated by hand.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from churn_analysis import (  # noqa: E402
    Window, behaviour_panel, decay_curve, gains_curve, intervention_value,
    lift_table, load_events,
)

INK = "#1b1b1f"
MUTED = "#8a8f98"
CHURN_C = "#c1442e"
STAY_C = "#4a7ba7"
GRID = "#e3e5e8"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "semibold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

SIGNAL_LABELS = {
    "rate_roll_advert": "Adverts per 100 events",
    "rate_thumbs_down": "Thumbs-down per 100 events",
    "rate_thumbs_up": "Thumbs-up per 100 events",
    "rate_help": "Help visits per 100 events",
    "rate_error": "Errors per 100 events",
    "rate_downgrade": "Downgrade visits per 100 events",
    "rate_settings": "Settings visits per 100 events",
    "rate_add_to_playlist": "Playlist adds per 100 events",
    "days_since_last_seen": "Days since last seen",
    "activity_trend": "Activity trend (2nd half vs 1st)",
    "events_per_active_day": "Events per active day",
    "events": "Total events",
    "songs": "Songs played",
    "active_days": "Active days",
    "distinct_artists": "Distinct artists",
    "sessions": "Sessions",
}


def fig_decay(ev, outdir, days_before=28):
    dc = decay_curve(ev, days_before=days_before)
    wide = dc.pivot(index="days_to_event", columns="group", values="events")

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for label, colour in (("Retained", STAY_C), ("Cancelled", CHURN_C)):
        if label in wide:
            ax.plot(wide.index, wide[label], color=colour, lw=2.2, label=label)
            ax.annotate(label, (wide.index[-1], wide[label].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        color=colour, fontweight="semibold", va="center")

    ax.set_xlabel("Days before cancellation")
    ax.set_ylabel("Median events per day")
    ax.set_title("Disengagement is visible well before the cancellation click")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-days_before, 2)
    ax.margins(y=0.15)
    fig.savefig(Path(outdir) / "01_decay_curve.png")
    plt.close(fig)
    return wide


def fig_lift(panel, signals, outdir, top_n=8):
    lt = lift_table(panel, signals)
    if lt.empty:
        return lt
    show = lt.head(top_n).iloc[::-1]
    labels = [
        f"{SIGNAL_LABELS.get(s, s)}  ({d})"
        for s, d in zip(show["signal"], show["direction"])
    ]

    fig, ax = plt.subplots(figsize=(7.6, 0.46 * len(show) + 1.5))
    bars = ax.barh(labels, show["lift"], color=CHURN_C, height=0.62)
    ax.axvline(1.0, color=MUTED, lw=1.2, ls="--")
    ax.annotate("no signal", (1.0, -0.9), color=MUTED, fontsize=8,
                ha="center", annotation_clip=False)
    for b, v in zip(bars, show["lift"]):
        ax.annotate(f"{v:.1f}x", (b.get_width(), b.get_y() + b.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, fontweight="semibold")
    ax.set_xlabel("Cancellation rate vs. average user")
    ax.set_title("Users in the extreme fifth of each signal")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.12)
    fig.savefig(Path(outdir) / "02_leading_signals.png")
    plt.close(fig)
    return lt


def fig_gains(y, scores, outdir):
    g = gains_curve(y, scores)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(g["share_contacted"] * 100, g["share_of_churners_caught"] * 100,
            color=CHURN_C, lw=2.2, label="Model ranking")
    ax.plot([0, 100], [0, 100], color=MUTED, lw=1.2, ls="--", label="Contact at random")

    for q in (0.10, 0.20):
        row = g.iloc[max(0, int(q * len(g)) - 1)]
        ax.scatter([q * 100], [row["share_of_churners_caught"] * 100],
                   color=INK, zorder=5, s=28)
        ax.annotate(f"{row['share_of_churners_caught']:.0%} of churners\nfor {q:.0%} contacted",
                    (q * 100, row["share_of_churners_caught"] * 100),
                    xytext=(10, -18), textcoords="offset points", fontsize=8.5)

    ax.set_xlabel("Share of users contacted (%)")
    ax.set_ylabel("Share of churners reached (%)")
    ax.set_title("How far a retention budget goes")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.savefig(Path(outdir) / "03_gains_curve.png")
    plt.close(fig)
    return g


def fig_economics(gains, n_users, base_rate, outdir, contact_cost, value, save_rate):
    v = intervention_value(gains, n_users, base_rate, contact_cost, value, save_rate)
    best = v.loc[v["net_value"].idxmax()]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(v["share_contacted"] * 100, v["net_value"], color=CHURN_C, lw=2.2)
    ax.axhline(0, color=MUTED, lw=1.0)
    ax.scatter([best["share_contacted"] * 100], [best["net_value"]],
               color=INK, zorder=5, s=40)
    ax.annotate(f"optimum: contact top {best['share_contacted']:.0%}",
                (best["share_contacted"] * 100, best["net_value"]),
                xytext=(8, 10), textcoords="offset points", fontweight="semibold", fontsize=9)
    ax.set_xlabel("Share of users contacted (%)")
    ax.set_ylabel("Net value of campaign")
    ax.set_title(f"Break-even at €{contact_cost:.2f}/contact, €{value:.0f} subscriber value, "
                 f"{save_rate:.0%} save rate")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.savefig(Path(outdir) / "04_intervention_economics.png")
    plt.close(fig)
    return v, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.parquet")
    ap.add_argument("--anchor", default=None,
                    help="Anchor date YYYY-MM-DD (default: horizon before last event)")
    ap.add_argument("--lookback", type=int, default=28)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--contact-cost", type=float, default=0.50)
    ap.add_argument("--subscriber-value", type=float, default=60.0)
    ap.add_argument("--save-rate", type=float, default=0.25)
    args = ap.parse_args()

    Path(args.outdir).mkdir(exist_ok=True)
    ev = load_events(args.data)

    last = ev["time"].max().normalize()
    anchor = pd.Timestamp(args.anchor) if args.anchor else last - pd.Timedelta(days=args.horizon)
    window = Window(anchor=anchor, lookback_days=args.lookback, horizon_days=args.horizon)

    print(f"events {len(ev):,} | users {ev['userId'].nunique():,} | "
          f"anchor {anchor:%Y-%m-%d} | lookback {args.lookback}d | horizon {args.horizon}d")

    decay = fig_decay(ev, args.outdir)
    panel = behaviour_panel(ev, window)

    signals = [c for c in panel.columns if c.startswith("rate_")] + [
        "days_since_last_seen", "activity_trend", "events_per_active_day",
        "events", "songs", "active_days", "distinct_artists", "sessions",
    ]
    signals = [s for s in signals if s in panel.columns]
    lt = fig_lift(panel, signals, args.outdir)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X = panel[signals].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = panel["churned"].to_numpy()
    if y.sum() < 10:
        raise SystemExit(f"Only {y.sum()} churners in the horizon — widen --horizon.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_predict(
        HistGradientBoostingClassifier(max_iter=200, random_state=0),
        X, y, cv=cv, method="predict_proba",
    )[:, 1]

    gains = fig_gains(y, scores, args.outdir)
    econ, best = fig_economics(gains, len(panel), y.mean(), args.outdir,
                               args.contact_cost, args.subscriber_value, args.save_rate)

    write_findings(decay, lt, gains, best, panel, y, window, args)
    print(f"\nwrote 4 figures to {args.outdir}/ and findings.md")


def write_findings(decay, lt, gains, best, panel, y, window, args) -> None:
    def at(q):
        return gains.iloc[max(0, int(q * len(gains)) - 1)]

    synthetic = Path("data/.synthetic").exists() and args.data in Path("data/.synthetic").read_text()

    lines = ["# Findings", ""]
    if synthetic:
        lines += [
            "> # ⚠️ SYNTHETIC DATA — NOT A FINDING",
            ">",
            "> These numbers come from `make_fixture.py`, which invents its own "
            "patterns. They demonstrate that the pipeline runs and nothing more. "
            "**Do not present, cite or quote anything on this page.** "
            "Rerun against the real event logs to replace it.",
            "",
        ]
    lines += [
        f"_Generated by `run_analysis.py` on {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
        "Every number below is computed, not typed._",
        "",
        "## Setup",
        "",
        f"- Anchor date: **{window.anchor:%Y-%m-%d}**",
        f"- Features from the {window.lookback_days} days before the anchor; "
        f"label is cancellation in the {window.horizon_days} days after it.",
        f"- Users at risk at the anchor: **{len(panel):,}**",
        f"- Cancelled within the horizon: **{int(y.sum()):,}** ({y.mean():.1%})",
        "",
        "## 1. How much warning do we get?",
        "",
    ]

    if {"Cancelled", "Retained"}.issubset(decay.columns):
        c, r = decay["Cancelled"], decay["Retained"]
        baseline = c.loc[-28:-21].mean() if -28 in c.index else c.iloc[:7].mean()
        below = [d for d in c.index if d < 0 and c[d] < 0.7 * baseline]
        first = min(below) if below else None
        lines += [
            f"- Median daily activity among users who cancel falls from "
            f"**{baseline:.0f}** events/day (4 weeks out) to **{c.loc[-1]:.0f}** on the final day.",
            f"- Retained users sit flat at **{r.loc[-1]:.0f}** events/day over the same period.",
        ]
        if first is not None:
            lines.append(
                f"- Activity first drops below 70% of its own baseline at "
                f"**day {first}** — that is the size of the intervention window."
            )
    lines += ["", "![decay](figures/01_decay_curve.png)", "", "## 2. What leads cancellation?", ""]

    if not lt.empty:
        lines += ["| Signal | Direction | Cancellation rate | Lift |", "|---|---|---|---|"]
        for _, row in lt.head(8).iterrows():
            lines.append(
                f"| {SIGNAL_LABELS.get(row['signal'], row['signal'])} | {row['direction']} "
                f"| {row['churn_rate_quintile']:.1%} | **{row['lift']:.1f}x** |"
            )
        top = lt.iloc[0]
        lines += [
            "",
            f"Strongest single signal: **{SIGNAL_LABELS.get(top['signal'], top['signal'])}** "
            f"({top['direction']}) — those users cancel at {top['churn_rate_quintile']:.1%} "
            f"against a base rate of {top['base_rate']:.1%}.",
        ]
    lines += ["", "![signals](figures/02_leading_signals.png)", "",
              "## 3. Who is worth contacting?", ""]

    for q in (0.05, 0.10, 0.20):
        row = at(q)
        lines.append(
            f"- Contacting the top **{q:.0%}** by risk reaches "
            f"**{row['share_of_churners_caught']:.0%}** of everyone who will cancel "
            f"(precision {row['precision']:.0%})."
        )
    lines += [
        "",
        "![gains](figures/03_gains_curve.png)",
        "",
        "## 4. What is that worth?",
        "",
        f"At €{args.contact_cost:.2f} per contact, €{args.subscriber_value:.0f} subscriber value "
        f"and a {args.save_rate:.0%} save rate, net value peaks when contacting the top "
        f"**{best['share_contacted']:.0%}** of users.",
        "",
        "![economics](figures/04_intervention_economics.png)",
        "",
        "> The save rate is an assumption, not a measurement — it is the one input here "
        "that only an experiment can supply. See the limitations section of the README.",
        "",
    ]
    Path("findings.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
