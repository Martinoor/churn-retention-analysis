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
) -> pd.DataFrame:
    """Daily activity aligned on the day of cancellation.

    Churners are aligned at day 0 = cancellation. Non-churners are given a
    placebo anchor drawn from the churners' anchor distribution, so both groups
    are measured over comparable calendar time and the comparison is not an
    artefact of the observation window ending.
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

    daily["anchor"] = daily["userId"].map(anchors)
    daily = daily.dropna(subset=["anchor"])
    daily["days_to_event"] = (daily["day"] - daily["anchor"]).dt.days
    daily = daily[daily["days_to_event"].between(-days_before, 0)]

    daily["group"] = np.where(
        daily["userId"].isin(cancel_at.index), "Cancelled", "Retained"
    )

    # Days with no events are real zeros, not missing data — reindex to restore them.
    full = (
        daily.set_index(["userId", "days_to_event"])["events"]
        .unstack(fill_value=0)
        .reindex(columns=range(-days_before, 1), fill_value=0)
    )
    group = daily.groupby("userId")["group"].first()

    out = (
        full.stack()
        .rename("events")
        .reset_index()
        .rename(columns={"level_1": "days_to_event"})
    )
    out["group"] = out["userId"].map(group)
    return (
        out.groupby(["group", "days_to_event"], observed=True)["events"]
        .median()
        .reset_index()
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

    panel["events_per_active_day"] = panel["events"] / panel["active_days"].clip(lower=1)

    # Trend: second half of the lookback against the first.
    midpoint = window.lookback_start + pd.Timedelta(days=window.lookback_days / 2)
    first = hist[hist["time"] <= midpoint].groupby("userId").size()
    second = hist[hist["time"] > midpoint].groupby("userId").size()
    panel["activity_trend"] = (
        second.reindex(panel.index).fillna(0) + 1
    ) / (first.reindex(panel.index).fillna(0) + 1)

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
