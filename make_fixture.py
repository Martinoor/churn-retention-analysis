"""Generate a synthetic event log with the same schema as the competition data.

Smoke-testing only. Nothing in the analysis write-up is derived from this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_START = pd.Timestamp("2018-10-01")
WINDOW_END = pd.Timestamp("2018-11-20")

PAGES = {
    "NextSong": 0.78,
    "Thumbs Up": 0.05,
    "Roll Advert": 0.04,
    "Add to Playlist": 0.03,
    "Thumbs Down": 0.02,
    "Home": 0.03,
    "Help": 0.01,
    "Settings": 0.01,
    "Downgrade": 0.01,
    "Error": 0.01,
    "Logout": 0.01,
}


def _events_for_day(rng, day_index, n_days, churns):
    """Daily event count. Churners decay over their final fortnight."""
    base = rng.lognormal(mean=3.0, sigma=0.6)
    if churns:
        days_left = n_days - day_index
        if days_left < 14:
            base *= max(0.08, days_left / 14.0)
    return max(0, int(base))


def build(n_users: int = 600, churn_rate: float = 0.223, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pages = list(PAGES)
    probs = np.array(list(PAGES.values()))
    probs = probs / probs.sum()

    rows = []
    n_days = (WINDOW_END - WINDOW_START).days + 1

    for uid in range(n_users):
        churns = rng.random() < churn_rate
        level = "paid" if rng.random() < 0.75 else "free"
        agent = rng.choice(["Windows NT 6.1", "Macintosh; Intel Mac OS X", "iPhone; CPU iPhone OS"])
        loc = rng.choice(["Boston-Cambridge, MA", "Phoenix-Mesa, AZ", "Dallas-Fort Worth, TX"])
        session = 0

        for d in range(n_days):
            n_ev = _events_for_day(rng, d, n_days, churns)
            if n_ev == 0:
                continue
            session += 1
            day = WINDOW_START + pd.Timedelta(days=d)
            # Churners see more adverts and register more dissatisfaction near the end.
            p = probs.copy()
            if churns and (n_days - d) < 14:
                for pg, mult in (("Roll Advert", 2.4), ("Thumbs Down", 2.2), ("Help", 1.8)):
                    p[pages.index(pg)] *= mult
                p = p / p.sum()

            chosen = rng.choice(pages, size=n_ev, p=p)
            offsets = np.sort(rng.integers(0, 86400, size=n_ev))
            for i, (pg, off) in enumerate(zip(chosen, offsets)):
                rows.append(
                    {
                        "userId": str(100000 + uid),
                        "time": day + pd.Timedelta(seconds=int(off)),
                        "page": pg,
                        "level": level,
                        "sessionId": session,
                        "itemInSession": i,
                        "length": float(rng.normal(240, 60)) if pg == "NextSong" else np.nan,
                        "song": f"song_{rng.integers(0, 400)}" if pg == "NextSong" else None,
                        "artist": f"artist_{rng.integers(0, 90)}" if pg == "NextSong" else None,
                        "userAgent": agent,
                        "location": loc,
                        "auth": "Logged In",
                        "status": 200,
                    }
                )

        if churns:
            last = max((r["time"] for r in rows if r["userId"] == str(100000 + uid)), default=WINDOW_END)
            for pg in ("Cancel", "Cancellation Confirmation"):
                rows.append(
                    {
                        "userId": str(100000 + uid), "time": last + pd.Timedelta(minutes=1),
                        "page": pg, "level": level, "sessionId": session, "itemInSession": 999,
                        "length": np.nan, "song": None, "artist": None, "userAgent": agent,
                        "location": loc, "auth": "Cancelled", "status": 200,
                    }
                )

    return pd.DataFrame(rows).sort_values(["userId", "time"]).reset_index(drop=True)


if __name__ == "__main__":
    import pathlib

    df = build()
    df.to_parquet("data/train.parquet", index=False)
    # Sentinel read by run_analysis.py so synthetic output is always labelled.
    pathlib.Path("data/.synthetic").write_text("data/train.parquet\n")
    n_churn = df.loc[df.page == "Cancellation Confirmation", "userId"].nunique()
    print(f"wrote data/train.parquet — {len(df):,} events, "
          f"{df.userId.nunique():,} users, {n_churn:,} churned")
