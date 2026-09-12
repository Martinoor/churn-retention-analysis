"""Behavioural analysis of subscriber churn.

The question this answers is not "how accurately can we predict churn" but
"what happens before a cancellation, what explains it, and who is worth
contacting". Modelling is used only to rank users so an intervention budget can
be sized.

Every panel feature is computed over a lookback window ending at an *anchor*
date, and the label is whether the user cancels in the horizon *after* that
anchor. Nothing after the anchor enters a feature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

CHURN_PAGE = "Cancellation Confirmation"
# Pages that reveal the outcome. Never allowed into a feature.
LEAKING_PAGES = {CHURN_PAGE, "Cancel"}
# A song counts as skipped when the next event arrives before this share of it has played.
SKIP_FRACTION = 0.5

COLUMNS = ["userId", "time", "page", "level", "sessionId", "length", "artist", "registration"]

SIGNAL_PAGES = [
    "Roll Advert",
    "Thumbs Down",
    "Thumbs Up",
    "Help",
    "Error",
    "Downgrade",
    "Settings",
    "Add to Playlist",
]

# Page groups used to decompose what users do in the run-up to cancellation.
COMPONENTS = {
    "songs": {"NextSong"},
    "adverts": {"Roll Advert"},
    "thumbs_down": {"Thumbs Down"},
    "thumbs_up": {"Thumbs Up"},
    "errors": {"Error"},
    "help": {"Help"},
    "settings": {"Settings", "Save Settings"},
    "downgrade_pages": {"Downgrade", "Submit Downgrade"},
    "upgrade_pages": {"Upgrade", "Submit Upgrade"},
    "playlist_adds": {"Add to Playlist"},
}

TENURE_BINS = [0, 28, 56, 91, 182, np.inf]
TENURE_LABELS = ["under 4 weeks", "4-8 weeks", "8-13 weeks", "3-6 months", "6 months+"]


@dataclass(frozen=True)
class Window:
    """Anchor date with a lookback for features and a horizon for the label."""

    anchor: pd.Timestamp
    lookback_days: int = 28
    horizon_days: int = 10

    @property
    def lookback_start(self) -> pd.Timestamp:
        return self.anchor - pd.Timedelta(days=self.lookback_days)

    @property
    def horizon_end(self) -> pd.Timestamp:
        return self.anchor + pd.Timedelta(days=self.horizon_days)


def load_events(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=COLUMNS)
    df["time"] = pd.to_datetime(df["time"])
    df["registration"] = pd.to_datetime(df["registration"])
    df["userId"] = df["userId"].astype(str)
    return df.sort_values(["userId", "time"], kind="mergesort").reset_index(drop=True)


def churn_dates(events: pd.DataFrame) -> pd.Series:
    """First cancellation timestamp per user. Users absent from the index never churned."""
    cancels = events.loc[events["page"] == CHURN_PAGE]
    return cancels.groupby("userId")["time"].min()


def registration_dates(events: pd.DataFrame) -> pd.Series:
    """Account creation per user.

    Tenure must come from here, not from a user's first event in the logs: the
    logs start on a fixed date, so "time since first seen" is capped by how far
    into the window the user cancelled and badly understates tenure.
    """
    return events.groupby("userId")["registration"].first()


def skip_flags(events: pd.DataFrame) -> np.ndarray:
    """True for songs cut short by the next event in the same session.

    Assumes events are sorted by user and time, as `load_events` returns them.
    """
    user = pd.factorize(events["userId"])[0]
    session = events["sessionId"].to_numpy()
    t = events["time"].to_numpy()

    gap = np.full(len(events), np.nan)
    same = (user[1:] == user[:-1]) & (session[1:] == session[:-1])
    gap[:-1] = np.where(same, (t[1:] - t[:-1]) / np.timedelta64(1, "s"), np.nan)

    is_song = (events["page"] == "NextSong").to_numpy()
    return is_song & (gap < SKIP_FRACTION * events["length"].to_numpy())


def session_table(events: pd.DataFrame) -> pd.DataFrame:
    """One row per session: start, calendar day, tier at start, and whether it contains a cancellation."""
    s = (
        events.assign(_cancel=events["page"].eq(CHURN_PAGE))
        .groupby(["userId", "sessionId"], sort=False)
        .agg(start=("time", "min"), level=("level", "first"), cancel=("_cancel", "max"))
        .reset_index()
    )
    s["day"] = s["start"].dt.normalize()
    return s


# --------------------------------------------------------------------------
# A. What happens before a cancellation, and what explains it?
# --------------------------------------------------------------------------

@dataclass
class RunUp:
    curves: pd.DataFrame       # group, days_to_event, median_events
    composition: pd.DataFrame  # component, group, baseline_per_100, final_per_100, change
    activity: pd.DataFrame     # group, baseline_per_day, final_per_day, change
    coverage: dict


def _daily_components(events: pd.DataFrame) -> pd.DataFrame:
    keep = ~events["page"].isin(LEAKING_PAGES).to_numpy()
    codes, uniques = pd.factorize(events["userId"])
    pages = events["page"][keep]

    data = {
        "user_code": codes[keep],
        "day": events["time"].dt.normalize().to_numpy()[keep],
        "events": np.ones(keep.sum(), dtype=np.uint8),
        "skips": skip_flags(events)[keep].astype(np.uint8),
    }
    for name, members in COMPONENTS.items():
        data[name] = pages.isin(members).to_numpy().astype(np.uint8)

    daily = pd.DataFrame(data).groupby(["user_code", "day"], sort=False).sum().reset_index()
    daily.insert(0, "userId", uniques[daily.pop("user_code").to_numpy()])
    return daily


def runup_analysis(
    events: pd.DataFrame,
    days_before: int = 28,
    smooth_days: int = 7,
    controls_per_canceller: int = 5,
    baseline: tuple[int, int] = (-28, -8),
    final: tuple[int, int] = (-7, -1),
    seed: int = 0,
) -> RunUp:
    """Activity before cancellation, against retained users matched on the same day.

    Comparing cancellers with retained users on an arbitrary date confounds
    three things with cancellation itself:

    * calendar — activity drifts over the observation window for everyone;
    * tier — cancellers skew paid, and paid users listen more;
    * presence — cancelling requires a visit, so a canceller's day 0 is always
      a day they were active, while a random day for a retained user is not.

    Each matched control is a retained user who had a session on the same
    calendar day and tier as the canceller. Day 0 is the day the cancellation
    session *started*, so sessions crossing midnight align the same way in both
    groups. The unmatched comparison is returned alongside, because the gap
    between the two controls is the measure of how much of the apparent
    difference was the comparison rather than the users.

    The final window stops at day -1: day 0 contains the cancellation flow
    itself, which would otherwise show up as "behaviour".
    """
    rng = np.random.default_rng(seed)
    lead_in = days_before + smooth_days - 1
    lead = pd.Timedelta(days=lead_in)
    window_start = events["time"].min().normalize()
    reg = registration_dates(events)

    def eligible(users: pd.Series, days: pd.Series) -> np.ndarray:
        # The whole lookback must be observed and fall after registration, or
        # days before an account existed get counted as days of inactivity.
        return ((days >= window_start + lead) & (reg.reindex(users).to_numpy() <= days - lead)).to_numpy()

    sessions = session_table(events)
    cancellers = (
        sessions[sessions["cancel"]].sort_values("start").groupby("userId")[["day", "level"]].first()
    )
    n_cancellers = len(cancellers)
    cancellers = cancellers[eligible(cancellers.index.to_series(), cancellers["day"])]

    pool = sessions.loc[~sessions["userId"].isin(churn_dates(events).index),
                        ["userId", "day", "level"]].drop_duplicates()
    pool = pool[eligible(pool["userId"], pool["day"])]
    by_day_tier = {k: g for k, g in pool.groupby(["day", "level"])}

    picks, unmatched_cancellers = [], 0
    for (day, level), n in cancellers.groupby(["day", "level"]).size().items():
        cand = by_day_tier.get((day, level))
        if cand is None:
            unmatched_cancellers += n
            continue
        picks.append(cand.sample(n=min(len(cand), n * controls_per_canceller),
                                 random_state=int(rng.integers(2**31 - 1))))
    matched = pd.concat(picks, ignore_index=True)

    retained_users = pool["userId"].unique()
    naive = pd.DataFrame({
        "userId": retained_users,
        "day": rng.choice(cancellers["day"].to_numpy(), size=len(retained_users)),
    })
    naive = naive[eligible(naive["userId"], naive["day"])]

    anchors = pd.concat([
        pd.DataFrame({"key": cancellers.index, "userId": cancellers.index,
                      "anchor": cancellers["day"].to_numpy(), "group": "Cancelled"}),
        pd.DataFrame({"key": matched["userId"] + "|m" + matched.index.astype(str),
                      "userId": matched["userId"], "anchor": matched["day"],
                      "group": "Retained, matched"}),
        pd.DataFrame({"key": naive["userId"] + "|n", "userId": naive["userId"],
                      "anchor": naive["day"], "group": "Retained, unmatched"}),
    ], ignore_index=True)

    daily = _daily_components(events)
    w = anchors.merge(daily, on="userId")
    w["d"] = (w["day"] - w["anchor"]).dt.days
    w = w[w["d"].between(-lead_in, 0)]

    curves = []
    for group, keys in anchors.groupby("group")["key"]:
        mat = (
            w[w["group"] == group]
            .pivot_table(index="key", columns="d", values="events", aggfunc="sum", fill_value=0)
            .reindex(index=keys.to_numpy(), columns=range(-lead_in, 1), fill_value=0)
        )
        # Usage is bursty, so smooth each user's own history before taking the
        # median across users; otherwise the median of a single day is mostly zero.
        smooth = mat.T.rolling(smooth_days, min_periods=smooth_days).mean().T
        med = smooth.loc[:, -days_before:0].median()
        curves.append(pd.DataFrame({"group": group, "days_to_event": med.index, "median_events": med.to_numpy()}))
    curves = pd.concat(curves, ignore_index=True)

    counts = ["events", "skips", *COMPONENTS]
    n_keys = anchors.groupby("group").size()
    span = lambda lo_hi: lo_hi[1] - lo_hi[0] + 1  # noqa: E731
    base = w[w["d"].between(*baseline)].groupby("group")[counts].sum().reindex(n_keys.index, fill_value=0)
    fin = w[w["d"].between(*final)].groupby("group")[counts].sum().reindex(n_keys.index, fill_value=0)

    activity = pd.DataFrame({
        "baseline_per_day": base["events"] / (n_keys * span(baseline)),
        "final_per_day": fin["events"] / (n_keys * span(final)),
    })
    activity["change"] = activity["final_per_day"] / activity["baseline_per_day"] - 1
    activity = activity.rename_axis("group").reset_index()

    comp = []
    for c in ["skips", *COMPONENTS]:
        for group in n_keys.index:
            b = 100 * base.loc[group, c] / max(base.loc[group, "events"], 1)
            f = 100 * fin.loc[group, c] / max(fin.loc[group, "events"], 1)
            comp.append({"component": c, "group": group, "baseline_per_100": b,
                         "final_per_100": f, "change": f / b - 1 if b else np.nan})

    return RunUp(
        curves=curves,
        composition=pd.DataFrame(comp),
        activity=activity,
        coverage={
            "cancellers_total": n_cancellers,
            "cancellers_eligible": len(cancellers),
            "cancellers_without_match": unmatched_cancellers,
            "matched_controls": len(matched),
            "matched_control_users": int(matched["userId"].nunique()),
            "unmatched_controls": len(naive),
        },
    )


# --------------------------------------------------------------------------
# B. Which signals lead cancellation?
# --------------------------------------------------------------------------

def behaviour_panel(events: pd.DataFrame, window: Window) -> pd.DataFrame:
    """One row per user: behaviour during the lookback, label from the horizon.

    Users who already cancelled before the anchor are dropped — they are not
    at risk and including them would inflate every signal.
    """
    cancel_at = churn_dates(events)

    already_gone = cancel_at[cancel_at <= window.anchor].index
    at_risk = np.setdiff1d(events["userId"].unique(), already_gone)

    hist = events[
        (events["time"] > window.lookback_start)
        & (events["time"] <= window.anchor)
        & (events["userId"].isin(at_risk))
        & (~events["page"].isin(LEAKING_PAGES))
    ].copy()

    if hist.empty:
        raise ValueError(f"No events in lookback window ending {window.anchor:%Y-%m-%d}")

    g = hist.groupby("userId", observed=True)
    panel = pd.DataFrame(index=pd.Index(sorted(hist["userId"].unique()), name="userId"))

    panel["events"] = g.size()
    panel["active_days"] = g["time"].apply(lambda s: s.dt.normalize().nunique())
    panel["sessions"] = g["sessionId"].nunique()
    panel["songs"] = hist[hist["page"] == "NextSong"].groupby("userId").size()
    panel["distinct_artists"] = (
        hist[hist["page"] == "NextSong"].groupby("userId")["artist"].nunique()
    )
    panel["days_since_last_seen"] = (
        window.anchor - g["time"].max()
    ).dt.total_seconds() / 86400.0

    page_counts = (
        hist.groupby(["userId", "page"], observed=True).size().unstack(fill_value=0)
    )
    for page in SIGNAL_PAGES:
        col = page.lower().replace(" ", "_")
        panel[f"n_{col}"] = page_counts.get(page, 0)

    panel = panel.fillna(0)

    # Rates per 100 events make the signals comparable across activity levels.
    for page in SIGNAL_PAGES:
        col = page.lower().replace(" ", "_")
        panel[f"rate_{col}"] = 100 * panel[f"n_{col}"] / panel["events"].clip(lower=1)

    # Adverts are served between songs, so songs are the natural denominator.
    panel["ads_per_100_songs"] = 100 * panel["n_roll_advert"] / panel["songs"].clip(lower=1)
    panel["events_per_active_day"] = panel["events"] / panel["active_days"].clip(lower=1)

    # Trend: slope of daily activity across the lookback, as a fraction of the
    # user's own mean, so it measures direction of travel rather than volume.
    daily_counts = (
        hist.assign(day=hist["time"].dt.normalize())
        .groupby(["userId", "day"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=pd.date_range(
                window.lookback_start.normalize() + pd.Timedelta(days=1),
                window.anchor.normalize(),
                freq="D",
            ),
            fill_value=0,
        )
        .reindex(panel.index, fill_value=0)
    )
    x = np.arange(daily_counts.shape[1], dtype=float)
    xc = x - x.mean()
    slope = (daily_counts.to_numpy() * xc).sum(axis=1) / (xc**2).sum()
    mean_daily = daily_counts.to_numpy().mean(axis=1)
    panel["activity_trend"] = slope / np.clip(mean_daily, 1e-9, None)

    panel["level"] = (
        hist.sort_values("time").groupby("userId")["level"].last().reindex(panel.index)
    )
    panel["tenure_days"] = (
        window.anchor - registration_dates(events).reindex(panel.index)
    ).dt.total_seconds() / 86400.0

    churns_in_horizon = cancel_at[
        (cancel_at > window.anchor) & (cancel_at <= window.horizon_end)
    ].index
    panel["churned"] = panel.index.isin(churns_in_horizon).astype(int)
    return panel


def lift_table(panel: pd.DataFrame, signals: Iterable[str], n_bins: int = 5) -> pd.DataFrame:
    """Churn rate in the most extreme quintile of each signal, against the base rate.

    Both tails are tested, because some signals predict churn when high and
    others when low. The reported direction says which tail it was, so the table
    cannot be read backwards.
    """
    base = panel["churned"].mean()
    rows = []
    for col in signals:
        if col not in panel or panel[col].nunique() < n_bins:
            continue
        try:
            hi = panel[col] >= panel[col].quantile(1 - 1 / n_bins)
            lo = panel[col] <= panel[col].quantile(1 / n_bins)
        except (TypeError, ValueError):
            continue

        best = None
        for direction, mask in (("high", hi), ("low", lo)):
            if mask.sum() == 0:
                continue
            rate = panel.loc[mask, "churned"].mean()
            lift = rate / base if base else np.nan
            if best is None or lift > best["lift"]:
                best = {
                    "signal": col,
                    "direction": direction,
                    "users_in_quintile": int(mask.sum()),
                    "churn_rate_quintile": rate,
                    "base_rate": base,
                    "lift": lift,
                }
        if best is not None:
            rows.append(best)

    if not rows:
        return pd.DataFrame(
            columns=["signal", "direction", "users_in_quintile",
                     "churn_rate_quintile", "base_rate", "lift"]
        )
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


def tier_lift(panel: pd.DataFrame, signal: str, n_bins: int = 5) -> dict:
    """Top-quintile lift of one signal, pooled and within each tier.

    Pooling tiers can hide or reverse a signal: free and paid users differ both
    in how often they cancel and in how much of some behaviours they can
    exhibit at all, so the pooled quintile partly sorts users by tier.
    """
    def top_lift(grp: pd.DataFrame) -> float:
        rate = grp["churned"].mean()
        top = grp[signal] >= grp[signal].quantile(1 - 1 / n_bins)
        return float(grp.loc[top, "churned"].mean() / rate) if rate else np.nan

    out = {"all": top_lift(panel)}
    for level, grp in panel.groupby("level"):
        out[str(level)] = top_lift(grp)
    return out


def exposure_table(panel: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Cancellations per 100 sessions, by visit frequency, within each tier.

    If this rate is roughly flat across quintiles, heavier users cancel more
    mainly because they visit more often — each visit carries a similar chance
    of ending in cancellation — not because they are less satisfied.
    """
    out = []
    for level, grp in panel.groupby("level"):
        quintile = pd.qcut(grp["sessions"].rank(method="first"), n_bins, labels=range(1, n_bins + 1))
        t = grp.groupby(quintile, observed=True).agg(
            users=("churned", "size"), churn_rate=("churned", "mean"), sessions=("sessions", "mean"),
        )
        t["per_100_sessions"] = 100 * t["churn_rate"] / t["sessions"]
        out.append(t.rename_axis("quintile").reset_index().assign(level=str(level)))
    return pd.concat(out, ignore_index=True)


def cohort_profile(events: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Who cancels: tenure from registration, tier at cancellation, and the tier/volume overlap."""
    cancels = events.loc[events["page"] == CHURN_PAGE].groupby("userId").agg(
        time=("time", "min"), level=("level", "last"),
    )
    tenure = (
        cancels["time"] - registration_dates(events).reindex(cancels.index)
    ).dt.total_seconds() / 86400.0

    out = {
        "churners": int(len(cancels)),
        "tenure_median_days": float(tenure.median()),
        "tenure_share_under_7d": float((tenure < 7).mean()),
        "tenure_share_under_28d": float((tenure < 28).mean()),
        "free_share_at_cancellation": float((cancels["level"] == "free").mean()),
    }

    buckets = pd.cut(panel["tenure_days"], TENURE_BINS, labels=TENURE_LABELS, right=False)
    by_tenure = panel.groupby(buckets, observed=True)["churned"].agg(["size", "mean"])
    out["churn_by_tenure"] = [
        {"bucket": str(b), "users": int(r["size"]), "churn_rate": float(r["mean"])}
        for b, r in by_tenure.iterrows()
    ]

    by_tier = panel.groupby("level")["churned"].mean()
    out["churn_rate_by_tier"] = {str(k): float(v) for k, v in by_tier.items()}

    heavy = panel["events"] >= panel["events"].quantile(0.8)
    out["paid_share_overall"] = float((panel["level"] == "paid").mean())
    out["paid_share_heavy"] = float((panel.loc[heavy, "level"] == "paid").mean())

    paid = panel[panel["level"] == "paid"]
    if len(paid) and paid["churned"].mean():
        paid_heavy = paid["events"] >= paid["events"].quantile(0.8)
        out["volume_lift_within_paid"] = float(
            paid.loc[paid_heavy, "churned"].mean() / paid["churned"].mean()
        )
    return out


# --------------------------------------------------------------------------
# C. Who is worth contacting?
# --------------------------------------------------------------------------

def gains_curve(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Share of churners captured as a function of share of users contacted."""
    order = np.argsort(-scores)
    y = np.asarray(y_true)[order]
    n = len(y)
    total = y.sum()
    return pd.DataFrame(
        {
            "share_contacted": np.arange(1, n + 1) / n,
            "share_of_churners_caught": np.cumsum(y) / total if total else np.zeros(n),
            "precision": np.cumsum(y) / np.arange(1, n + 1),
        }
    )


def rule_baseline(panel: pd.DataFrame, gains: pd.DataFrame, recent_days: float = 3) -> dict:
    """Benchmark the model against a one-line rule: paid, and seen in the last few days.

    The model is compared at the same share of users contacted, so the
    difference is what the model adds over the rule rather than a different budget.
    """
    y = panel["churned"].to_numpy()
    rule = ((panel["level"] == "paid") & (panel["days_since_last_seen"] < recent_days)).to_numpy()
    share = float(rule.mean())
    model = gains.iloc[max(0, int(round(share * len(gains))) - 1)]
    return {
        "recent_days": recent_days,
        "share_flagged": share,
        "rule_catch": float(y[rule].sum() / y.sum()),
        "rule_precision": float(y[rule].mean()),
        "model_catch": float(model["share_of_churners_caught"]),
        "model_precision": float(model["precision"]),
    }


def intervention_value(
    gains: pd.DataFrame,
    n_users: int,
    base_rate: float,
    contact_cost: float,
    subscriber_value: float,
    save_rate: float,
) -> pd.DataFrame:
    """Net value of contacting the top-scoring share of users.

    Assumes a contacted churner is saved with probability `save_rate` and that
    contacting a non-churner costs money but does no harm. Both assumptions are
    optimistic and are flagged as such in the write-up.
    """
    df = gains.copy()
    contacted = df["share_contacted"] * n_users
    churners_reached = df["share_of_churners_caught"] * base_rate * n_users
    df["cost"] = contacted * contact_cost
    df["value_saved"] = churners_reached * save_rate * subscriber_value
    df["net_value"] = df["value_saved"] - df["cost"]
    return df
