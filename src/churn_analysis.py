"""Behavioural analysis of subscriber churn.

The question this answers is not "how accurately can we predict churn" but
"what does the run-up to cancellation look like, how much warning do we get,
and who is worth contacting". Modelling is used only to rank users so the
intervention budget can be sized.

Design choice worth stating explicitly: every feature is computed over a
lookback window ending at an *anchor* date, and the label is whether the user
cancels in the horizon *after* that anchor. Nothing after the anchor enters a
feature. This is the forward-looking definition the business actually cares
about, and it is what makes the gains curve in `gains_curve` honest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

CHURN_PAGE = "Cancellation Confirmation"
# Pages that reveal the outcome. Never allowed into a feature.
LEAKING_PAGES = {CHURN_PAGE, "Cancel"}

# Behavioural signals tracked in the run-up to cancellation.
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
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df["userId"] = df["userId"].astype(str)
    return df.sort_values(["userId", "time"], kind="mergesort").reset_index(drop=True)


def churn_dates(events: pd.DataFrame) -> pd.Series:
    """First cancellation timestamp per user. Users absent from the index never churned."""
    cancels = events.loc[events["page"] == CHURN_PAGE]
    return cancels.groupby("userId")["time"].min()


# --------------------------------------------------------------------------
# A. How much warning do we get?
# --------------------------------------------------------------------------

def decay_curve(
    events: pd.DataFrame,
    days_before: int = 28,
    seed: int = 0,
    smooth_days: int = 7,
) -> pd.DataFrame:
    """Daily activity aligned on the day of cancellation.

    Churners are aligned at day 0 = cancellation. Non-churners are given a
    placebo anchor drawn from the churners' anchor distribution, so both groups
    are measured over comparable calendar time and the comparison is not an
    artefact of the observation window ending.

    Each user's daily counts are smoothed over a trailing `smooth_days` window
    before being summarised across users, so the curve tracks recent activity
    level rather than whether a given calendar day happened to contain a session.

    Only users whose history covers the whole window are included. A day before
    a user's first-ever event is not a day they were idle — they were not yet a
    subscriber — and counting those as zeros measures account age rather than
    disengagement. Most cancellers have short histories, so this filter changes
    the result substantially; `coverage` on the returned frame reports how many
    users survived it.
    """
    cancel_at = churn_dates(events)
    rng = np.random.default_rng(seed)

    all_users = events["userId"].unique()
    churner_anchors = cancel_at.dt.normalize()
    non_churners = np.setdiff1d(all_users, cancel_at.index.to_numpy())

    placebo = pd.Series(
        rng.choice(churner_anchors.to_numpy(), size=len(non_churners), replace=True),
        index=non_churners,
    )
    anchors = pd.concat([churner_anchors, placebo])
    anchors.index.name = "userId"

    activity = events.loc[~events["page"].isin(LEAKING_PAGES)].copy()
    activity["day"] = activity["time"].dt.normalize()
    daily = (
        activity.groupby(["userId", "day"], observed=True)
        .size()
        .rename("events")
        .reset_index()
    )

    # The trailing mean at the earliest plotted day needs smooth_days of real
    # history behind it, or that end of the curve is left unsmoothed and the
    # comparison across the window is not like-for-like.
    lead_in = days_before + smooth_days - 1
    first_seen = activity.groupby("userId")["day"].min()
    observed_from = (anchors - first_seen.reindex(anchors.index)).dt.days
    eligible = observed_from[observed_from >= lead_in].index

    daily = daily[daily["userId"].isin(eligible)]
    daily["anchor"] = daily["userId"].map(anchors)
    daily = daily.dropna(subset=["anchor"])
    daily["days_to_event"] = (daily["day"] - daily["anchor"]).dt.days
    daily = daily[daily["days_to_event"].between(-lead_in, 0)]

    daily["group"] = np.where(
        daily["userId"].isin(cancel_at.index), "Cancelled", "Retained"
    )

    # Within a user's own history, a day with no events is a real zero.
    full = (
        daily.set_index(["userId", "days_to_event"])["events"]
        .unstack(fill_value=0)
        .reindex(columns=range(-lead_in, 1), fill_value=0)
    )
    # Real usage is bursty: most users are idle on any given day, so a per-day
    # median across users collapses to zero and hides the decline. Smoothing each
    # user's own history first measures recent intensity rather than "logged in today".
    full = full.T.rolling(smooth_days, min_periods=smooth_days).mean().T
    full = full.loc[:, range(-days_before, 1)]

    group = daily.groupby("userId")["group"].first()

    out = (
        full.stack()
        .rename("events")
        .reset_index()
        .rename(columns={"level_1": "days_to_event"})
    )
    out["group"] = out["userId"].map(group)
    curve = (
        out.groupby(["group", "days_to_event"], observed=True)["events"]
        .median()
        .reset_index()
    )
    curve.attrs["coverage"] = {
        "users_plotted": int(group.size),
        "churners_plotted": int((group == "Cancelled").sum()),
        "churners_total": int(len(cancel_at)),
    }
    return curve


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

    panel["events_per_active_day"] = panel["events"] / panel["active_days"].clip(lower=1)

    # Trend: slope of daily activity across the lookback, as a fraction of the
    # user's own mean. A two-bucket half-vs-half ratio dilutes a late decline
    # into a 14-day average; a slope uses every day and does not depend on where
    # a midpoint happens to fall. Normalising by the mean keeps the signal about
    # direction of travel rather than volume, so it stays comparable across
    # light and heavy users.
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

    churns_in_horizon = cancel_at[
        (cancel_at > window.anchor) & (cancel_at <= window.horizon_end)
    ].index
    panel["churned"] = panel.index.isin(churns_in_horizon).astype(int)
    return panel


def lift_table(panel: pd.DataFrame, signals: Iterable[str], n_bins: int = 5) -> pd.DataFrame:
    """Churn rate in the most extreme quintile of each signal, against the base rate.

    Both tails are tested, because some signals predict churn when high (adverts
    seen, thumbs down) and others when low (songs played, activity trend). The
    reported direction says which tail it was, so the table cannot be read
    backwards.

    Lift is a ratio, so it reads the same way to a non-technical audience
    regardless of the underlying scale: 2.0 means twice as likely to cancel.
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


def cohort_profile(events: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Who cancels, independent of the run-up.

    Tenure is measured against a user's own first event, which is the only
    registration proxy the logs carry. The tier split is here because heavy
    usage and paid tier are entangled — reporting volume lift without it invites
    the reader to conclude that engagement causes cancellation.
    """
    cancel_at = churn_dates(events)
    first_seen = events.groupby("userId")["time"].min()
    tenure = (
        cancel_at - first_seen.reindex(cancel_at.index)
    ).dt.total_seconds() / 86400.0

    out = {
        "churners": int(len(cancel_at)),
        "tenure_median_days": float(tenure.median()),
        "tenure_share_under_7d": float((tenure < 7).mean()),
        "tenure_share_under_28d": float((tenure < 28).mean()),
    }

    if "level" in panel and panel["level"].notna().any():
        by_tier = panel.groupby("level")["churned"].mean()
        out["churn_rate_by_tier"] = {k: float(v) for k, v in by_tier.items()}

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
