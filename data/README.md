# Data

The event logs are not committed — competition data is not redistributable.

To reproduce the analysis, place the two files here:

```
data/train.parquet
data/test.parquet
```

Each row is one user interaction event. Columns used by this analysis:

| Column | Meaning |
|---|---|
| `userId` | User identifier |
| `time` | Event timestamp |
| `page` | Page or action (`NextSong`, `Thumbs Down`, `Roll Advert`, `Cancellation Confirmation`, …) |
| `level` | Subscription tier at the time of the event (`free` / `paid`) |
| `sessionId` | Session identifier |
| `length` | Track length in seconds, where applicable |
| `song`, `artist` | Track metadata |
| `userAgent`, `location` | Device and geography |

Observation window: 2018-10-01 to 2018-11-20. A user is treated as churned if
they reach the `Cancellation Confirmation` page — see the note on label
definition in the top-level README.

## Synthetic fixture

`make_fixture.py` in the repo root generates a small synthetic dataset with the
same schema, so the analysis code can be exercised without the real data. It is
for smoke-testing only — **no finding in the README is derived from it.**

```bash
python make_fixture.py          # writes data/train.parquet (synthetic)
```
