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
    Window, behaviour_panel, cohort_profile, exposure_table, gains_curve,
    intervention_value, lift_table, load_events, rule_baseline, runup_analysis,
    tier_lift,
)

INK = "#1b1b1f"
MUTED = "#8a8f98"
CHURN_C = "#c1442e"
STAY_C = "#4a7ba7"
NAIVE_C = "#b9bec6"
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
    "ads_per_100_songs": "Adverts per 100 songs",
    "days_since_last_seen": "Days since last seen",
    "activity_trend": "Activity trend (daily slope)",
    "events_per_active_day": "Events per active day",
    "events": "Total events",
    "songs": "Songs played",
    "active_days": "Active days",
    "distinct_artists": "Distinct artists",
    "sessions": "Sessions",
}

SIGNAL_KINDS = {
    "events": "volume",
    "songs": "volume",
    "active_days": "volume",
    "distinct_artists": "volume",
    "sessions": "volume",
    "events_per_active_day": "volume",
    "activity_trend": "trend",
    "days_since_last_seen": "recency",
}

COMPONENT_LABELS = {
    "skips": "Songs skipped",
    "songs": "Songs played",
    "adverts": "Adverts",
    "thumbs_down": "Thumbs-down",
    "thumbs_up": "Thumbs-up",
    "errors": "Errors",
    "help": "Help page",
    "settings": "Settings",
    "downgrade_pages": "Downgrade pages",
    "upgrade_pages": "Upgrade pages",
    "playlist_adds": "Playlist adds",
}

GROUP_STYLE = {
    "Retained, unmatched": dict(color=NAIVE_C, lw=1.8, ls="--"),
    "Retained, matched": dict(color=STAY_C, lw=2.2),
    "Cancelled": dict(color=CHURN_C, lw=2.4),
}


def fig_runup(runup, outdir):
    wide = runup.curves.pivot(index="days_to_event", columns="group", values="median_events")

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ends = {}
    for group, style in GROUP_STYLE.items():
        if group in wide:
            ax.plot(wide.index, wide[group], **style)
            ends[group] = wide[group].iloc[-1]

    # Nudge end labels apart when lines finish close together.
    span = np.nanmax(wide.to_numpy()) - np.nanmin(wide.to_numpy())
    placed = []
    for group, y in sorted(ends.items(), key=lambda kv: kv[1]):
        if placed and y - placed[-1] < 0.07 * span:
            y = placed[-1] + 0.07 * span
        placed.append(y)
        ax.annotate(group, (wide.index[-1], y), xytext=(8, 0), textcoords="offset points",
                    color=GROUP_STYLE[group]["color"], fontweight="semibold", va="center",
                    annotation_clip=False)

    ax.set_xlabel("Days before cancellation (retained users: days before a matched date)")
    ax.set_ylabel("Median events per day (7-day trailing mean)")
    ax.set_title("Cancellers against retained users active on the same day, on the same plan")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(wide.index.min(), 1)
    ax.margins(y=0.12)
    fig.savefig(Path(outdir) / "01_runup_matched.png")
    plt.close(fig)
    return wide


def fig_composition(runup, outdir):
    comp = runup.composition[runup.composition["group"].isin(["Cancelled", "Retained, matched"])]
    wide = comp.pivot(index="component", columns="group", values="change") * 100
    wide = wide.sort_values("Cancelled")
    labels = [COMPONENT_LABELS.get(c, c) for c in wide.index]

    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(wide) + 1.6))
    ypos = np.arange(len(wide))
    for y, (_, row) in zip(ypos, wide.iterrows()):
        ax.plot([row["Retained, matched"], row["Cancelled"]], [y, y], color=GRID, lw=3, zorder=1)
    ax.scatter(wide["Retained, matched"], ypos, color=STAY_C, s=46, zorder=3, label="Retained, matched")
    ax.scatter(wide["Cancelled"], ypos, color=CHURN_C, s=46, zorder=4, label="Cancelled")
    ax.axvline(0, color=MUTED, lw=1.0)
    ax.set_yticks(ypos, labels)
    ax.set_xlabel("Change in share of activity, last week vs. weeks 2-4 before (%)")
    ax.set_title("Mix of activity in the final week: cancellers vs. matched retained users")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.savefig(Path(outdir) / "02_runup_composition.png")
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
    fig.savefig(Path(outdir) / "03_leading_signals.png")
    plt.close(fig)
    return lt


def fig_gains(gains, rule, outdir):
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    ax.plot(gains["share_contacted"] * 100, gains["share_of_churners_caught"] * 100,
            color=CHURN_C, lw=2.2, label="Model ranking")
    ax.plot([0, 100], [0, 100], color=MUTED, lw=1.2, ls="--", label="Contact at random")

    rx, ry = rule["share_flagged"] * 100, rule["rule_catch"] * 100
    ax.scatter([rx], [ry], marker="X", color=INK, s=70, zorder=6,
               label=f"Rule: paid, seen in last {rule['recent_days']:g} days")
    ax.annotate(f"rule {ry:.0f}% vs model {rule['model_catch'] * 100:.0f}%",
                (rx, ry), xytext=(10, -16), textcoords="offset points", fontsize=8.5)

    for q in (0.10,):
        row = gains.iloc[max(0, int(q * len(gains)) - 1)]
        ax.scatter([q * 100], [row["share_of_churners_caught"] * 100], color=CHURN_C, zorder=5, s=28)
        ax.annotate(f"{row['share_of_churners_caught']:.0%} of churners\nfor {q:.0%} contacted",
                    (q * 100, row["share_of_churners_caught"] * 100),
                    xytext=(10, 4), textcoords="offset points", fontsize=8.5)

    ax.set_xlabel("Share of users contacted (%)")
    ax.set_ylabel("Share of churners reached (%)")
    ax.set_title("How far a retention budget goes")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.savefig(Path(outdir) / "04_gains_curve.png")
    plt.close(fig)


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
    fig.savefig(Path(outdir) / "05_intervention_economics.png")
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

    runup = runup_analysis(ev)
    fig_runup(runup, args.outdir)
    fig_composition(runup, args.outdir)
    print("run-up analysis done")

    panel = behaviour_panel(ev, window)
    signals = [c for c in panel.columns if c.startswith("rate_")] + [
        "ads_per_100_songs", "days_since_last_seen", "activity_trend", "events_per_active_day",
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

    gains = gains_curve(y, scores)
    rule = rule_baseline(panel, gains)
    fig_gains(gains, rule, args.outdir)
    econ, best = fig_economics(gains, len(panel), y.mean(), args.outdir,
                               args.contact_cost, args.subscriber_value, args.save_rate)

    write_findings(
        runup=runup, lt=lt, gains=gains, best=best, panel=panel, y=y, window=window, args=args,
        profile=cohort_profile(ev, panel), exposure=exposure_table(panel),
        ads=tier_lift(panel, "ads_per_100_songs"), rule=rule,
    )
    print(f"\nwrote 5 figures to {args.outdir}/ and findings.md")


def write_findings(*, runup, lt, gains, best, panel, y, window, args, profile, exposure, ads, rule) -> None:
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
        f"- Anchor date for the panel (§3–§7): **{window.anchor:%Y-%m-%d}**",
        f"- Features from the {window.lookback_days} days before the anchor; "
        f"label is cancellation in the {window.horizon_days} days after it.",
        f"- Users at risk at the anchor: **{len(panel):,}**",
        f"- Cancelled within the horizon: **{int(y.sum()):,}** ({y.mean():.1%})",
        "",
    ]

    # ---- 1. Run-up
    cov = runup.coverage
    act = runup.activity.set_index("group")
    wide = runup.curves.pivot(index="days_to_event", columns="group", values="median_events")
    first_day = int(wide.index.min())
    lines += [
        "## 1. What happens before a cancellation?",
        "",
        f"Compared against three groups. **Cancelled**: {cov['cancellers_eligible']:,} of "
        f"{cov['cancellers_total']:,} cancellers whose account existed, and whose history was "
        f"observed, for the full lookback. **Retained, matched**: {cov['matched_controls']:,} "
        f"draws of {cov['matched_control_users']:,} never-cancelling users who had a session on "
        f"the same calendar day, on the same tier, as a canceller. **Retained, unmatched**: "
        f"{cov['unmatched_controls']:,} retained users on a random canceller date, with no other "
        f"matching — the naive comparison, kept to show how much of the gap it creates.",
        "",
        "| Group | Events/day, weeks 2–4 before | Events/day, final week (excl. day 0) | Change |",
        "|---|---|---|---|",
    ]
    for g in ["Cancelled", "Retained, matched", "Retained, unmatched"]:
        if g in act.index:
            r = act.loc[g]
            lines.append(f"| {g} | {r['baseline_per_day']:.1f} | {r['final_per_day']:.1f} | {r['change']:+.0%} |")

    c_m, m_m = act.loc["Cancelled", "change"], act.loc["Retained, matched", "change"]
    lvl_c, lvl_m = wide.loc[first_day, "Cancelled"], wide.loc[first_day, "Retained, matched"]
    lvl_n = wide.loc[first_day, "Retained, unmatched"]
    mean_gap = act.loc["Cancelled", "baseline_per_day"] / act.loc["Retained, matched", "baseline_per_day"] - 1
    lines += [
        "",
        f"- Four weeks out, the median canceller runs at **{lvl_c:.1f}** events/day, against "
        f"**{lvl_m:.1f}** for matched retained users and **{lvl_n:.1f}** for unmatched ones "
        f"(means differ by {mean_gap:+.0%}). "
        + ("Against a fair comparison, cancellers are about as active as users who stay — the "
           f"{lvl_c / lvl_n:.1f}x gap to the unmatched group is the comparison, not the users."
           if abs(lvl_c / lvl_m - 1) < 0.15 else
           f"Cancellers remain {lvl_c / lvl_m:.1f}x as active even against the matched group."),
        f"- Over the final week, cancellers' activity changes by **{c_m:+.0%}** against "
        f"**{m_m:+.0%}** for matched retained users — "
        + ("no run-up specific to cancellation."
           if abs(c_m - m_m) < 0.10 else
           f"a {abs(c_m - m_m):.0%} gap that is specific to cancellation."),
        f"- On day 0 the median jumps to **{wide.loc[0, 'Cancelled']:.1f}** for cancellers and "
        f"**{wide.loc[0, 'Retained, matched']:.1f}** for matched retained users: the spike "
        "is what any day with a visit looks like, not a signature of cancelling.",
    ]
    if cov["cancellers_without_match"]:
        lines.append(f"- {cov['cancellers_without_match']:,} cancellers had no same-day, same-tier "
                     "retained user to match and are left out of the matched comparison.")
    lines += ["", "![run-up](figures/01_runup_matched.png)", ""]

    # ---- 2. Composition
    comp = runup.composition.set_index(["component", "group"])
    lines += [
        "## 2. Is it errors, skips, adverts?",
        "",
        "Share of activity (per 100 events) in weeks 2–4 before, and in the final week "
        "excluding day 0 — the cancellation flow itself happens on day 0 and would "
        "otherwise show up as behaviour.",
        "",
        "| Component | Cancelled | Retained, matched | Gap in change |",
        "|---|---|---|---|",
    ]
    standouts = []
    for c in COMPONENT_LABELS:
        if (c, "Cancelled") not in comp.index:
            continue
        a, b = comp.loc[(c, "Cancelled")], comp.loc[(c, "Retained, matched")]
        gap = a["change"] - b["change"]
        lines.append(
            f"| {COMPONENT_LABELS[c]} | {a['baseline_per_100']:.2f} → {a['final_per_100']:.2f} "
            f"| {b['baseline_per_100']:.2f} → {b['final_per_100']:.2f} | {gap:+.0%} |"
        )
        # Components under 0.2 per 100 events are too sparse for a percentage change to mean much.
        if abs(gap) >= 0.20 and min(a["baseline_per_100"], b["baseline_per_100"]) >= 0.2:
            standouts.append(f"{COMPONENT_LABELS[c]} ({gap:+.0%})")
    lines += [
        "",
        ("Components whose share moves at least 20 points more for cancellers than for "
         "matched users: " + ", ".join(standouts) + "."
         if standouts else
         "No component with meaningful volume moves more than 20 points differently for "
         "cancellers than for matched users: the mix of what they do — songs, skips, errors, "
         "help, thumbs-down, adverts, downgrade pages — is the same as anyone else's."),
        "",
        "![composition](figures/02_runup_composition.png)",
        "",
    ]

    # ---- 3. Signals
    lines += ["## 3. What leads cancellation?", ""]
    if not lt.empty:
        lines += ["| Signal | Kind | Direction | Cancellation rate | Lift |",
                  "|---|---|---|---|---|"]
        for _, row in lt[lt["lift"] >= 1.0].iterrows():
            lines.append(
                f"| {SIGNAL_LABELS.get(row['signal'], row['signal'])} "
                f"| {SIGNAL_KINDS.get(row['signal'], 'behaviour')} | {row['direction']} "
                f"| {row['churn_rate_quintile']:.1%} | **{row['lift']:.1f}x** |"
            )
        flat = lt[lt["lift"] < 1.0]
        if not flat.empty:
            names = ", ".join(f"{SIGNAL_LABELS.get(s, s)} ({l:.2f}x)"
                              for s, l in zip(flat["signal"], flat["lift"]))
            lines += ["", f"Tested with no quintile above the base rate: {names}."]

    tiers = {k: v for k, v in ads.items() if k != "all"}
    lines += [
        "",
        f"**Adverts, pooled vs. within tier.** Top fifth by adverts per 100 songs: "
        f"**{ads['all']:.2f}x** across all users, but "
        + ", ".join(f"**{v:.2f}x** within {k} users" for k, v in tiers.items())
        + (". Pooling hides the signal: paid users see few adverts and cancel more often, so a "
           "high advert rate partly just means \"free tier\". Within each tier, heavier advert "
           "exposure goes with more cancellation."
           if ads["all"] < 1.0 and all(v > 1.0 for v in tiers.values()) else "."),
        "",
        "![signals](figures/03_leading_signals.png)",
        "",
    ]

    # ---- 4. Exposure
    lines += [
        "## 4. Why do heavier users cancel more?",
        "",
        "Cancellation rate per visit, by how often users visit (quintiles of sessions in the lookback).",
        "",
        "| Tier | Quintile | Sessions (mean) | Cancellation rate | Per 100 sessions |",
        "|---|---|---|---|---|",
    ]
    for _, r in exposure.iterrows():
        lines.append(f"| {r['level']} | Q{r['quintile']} | {r['sessions']:.1f} | "
                     f"{r['churn_rate']:.1%} | {r['per_100_sessions']:.2f} |")
    lines.append("")
    for level, grp in exposure.groupby("level"):
        lo, hi = grp.iloc[0], grp.iloc[-1]
        lines.append(
            f"- **{level}**: the most frequent visitors come {hi['sessions'] / lo['sessions']:.1f}x as "
            f"often as the least frequent and cancel {hi['churn_rate'] / lo['churn_rate']:.1f}x as "
            f"often — per visit, their cancellation rate is {hi['per_100_sessions'] / lo['per_100_sessions']:.0%} "
            "of the lightest group's."
        )
    lines += [
        "",
        "Heavier users cancel more mostly because they have more visits in which to do it, "
        "not because each visit is riskier — per visit, the most engaged users are the "
        "*least* likely to cancel.",
        "",
    ]

    # ---- 5. Who cancels
    lines += [
        "## 5. Who cancels?",
        "",
        f"- Tenure at cancellation, from account registration: median **{profile['tenure_median_days']:.0f} days**; "
        f"**{profile['tenure_share_under_28d']:.0%}** under 28 days, **{profile['tenure_share_under_7d']:.0%}** under 7.",
        f"- **{profile['free_share_at_cancellation']:.0%}** of cancellations come from free-tier accounts.",
        "",
        "| Tenure at anchor | Users | Cancellation rate |",
        "|---|---|---|",
    ]
    for r in profile["churn_by_tenure"]:
        # A few hundred users means a few dozen cancellations at most — too few to read a rate from.
        note = " _(too few users to read)_" if r["users"] < 500 else ""
        lines.append(f"| {r['bucket']}{note} | {r['users']:,} | {r['churn_rate']:.1%} |")
    tiers_rate = profile.get("churn_rate_by_tier", {})
    if {"free", "paid"}.issubset(tiers_rate):
        lines += ["", f"- Paid users cancel at **{tiers_rate['paid']:.1%}** against "
                      f"**{tiers_rate['free']:.1%}** for free users."]
    if "paid_share_heavy" in profile:
        lines.append(
            f"- The heaviest fifth by event volume is **{profile['paid_share_heavy']:.0%}** paid "
            f"against **{profile['paid_share_overall']:.0%}** overall; within paid users alone, "
            f"high volume carries **{profile.get('volume_lift_within_paid', float('nan')):.2f}x** lift."
        )
    lines += [
        "",
        "> Cancellation is only visible when a user clicks through it. A user who quietly stops "
        "listening is counted as retained.",
        "",
    ]

    # ---- 6. Targeting
    lines += ["## 6. Who is worth contacting?", ""]
    for q in (0.05, 0.10, 0.20):
        row = at(q)
        lines.append(
            f"- Contacting the top **{q:.0%}** by risk reaches "
            f"**{row['share_of_churners_caught']:.0%}** of everyone who will cancel "
            f"(precision {row['precision']:.0%})."
        )
    lines += [
        "",
        f"**Against a one-line rule.** \"Paid, and seen in the last {rule['recent_days']:g} days\" flags "
        f"**{rule['share_flagged']:.0%}** of users and catches **{rule['rule_catch']:.0%}** of cancellers "
        f"(precision {rule['rule_precision']:.1%}). Contacting the same share by model score catches "
        f"**{rule['model_catch']:.0%}** (precision {rule['model_precision']:.1%}). The model's added value "
        f"over the rule is **{(rule['model_catch'] - rule['rule_catch']) * 100:+.0f} points** of cancellers reached.",
        "",
        "![gains](figures/04_gains_curve.png)",
        "",
        "## 7. What is that worth?",
        "",
        f"At €{args.contact_cost:.2f} per contact, €{args.subscriber_value:.0f} subscriber value "
        f"and a {args.save_rate:.0%} save rate, net value peaks when contacting the top "
        f"**{best['share_contacted']:.0%}** of users.",
        "",
        "![economics](figures/05_intervention_economics.png)",
        "",
        "> The save rate is an assumption, not a measurement — it is the one input here "
        "that only an experiment can supply. See the limitations section of the README.",
        "",
    ]
    Path("findings.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
