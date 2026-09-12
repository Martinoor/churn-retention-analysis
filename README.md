# Who is about to cancel, and what should we do about it?

A behavioural analysis of subscriber churn on a music-streaming service, using
51 days of raw event logs from ~19k users.

**The short version:** the intuitive story — users quietly disengage for weeks,
and cancelling is the end of a long fade — is not in this data. Neither is the
story a first look suggested instead, that cancellers are unusually heavy users
who ramp up before they leave. Compared fairly, against users on the same plan
who visited on the same day, cancellers behave like everyone else right up to
the moment they cancel: the same amount of listening, the same skips, the same
errors, the same thumbs-down. Cancellation looks like a **decision taken during
an ordinary visit**, and heavy users cancel more mainly because they visit more
often. That moves the lever away from predicting who is drifting and towards the
moment of decision itself: the cancellation flow.

> Findings with live numbers are in **[findings.md](findings.md)**, regenerated
> from the data by `run_analysis.py`. Nothing in it is typed by hand.

---

## Attribution

The underlying modelling work was done with a teammate for the 2025/26 Kaggle
churn competition, **which we won**. The competition repository is separate.

This repository is my own: the behavioural analysis, the framing, the
intervention economics and all findings here are mine.

---

## 1. The problem

A subscription business has exactly two levers on revenue — acquire more
subscribers, or keep the ones it has. The second is usually cheaper, but only if
you can answer three questions in order:

1. **Is churn predictable at all**, or does it look like a coin flip until the
   moment someone clicks cancel?
2. **If it is predictable, from what?** A signal that only fires the day before
   cancellation is useless; nobody can act on it.
3. **Where should the retention effort go?** Contacting everybody costs more
   than it saves.

The original competition asked only a narrower version of question 1 — maximise
balanced accuracy on a held-out set. That is a leaderboard question, not a
business one. This analysis answers all three.

## 2. The data, and two constraints

51 days of event logs (2018-10-01 to 2018-11-20). One row per user interaction:
page viewed, timestamp, subscription tier, session, track metadata, device,
location, and account registration date. A user is treated as churned when they
reach the `Cancellation Confirmation` page.

**Constraint one: the prediction window is not in the data.** The competition
brief asked for churn "in the 10 days following 2018-11-20" — but the released
event logs *end* on 2018-11-20. Two resolutions exist:

| Option | What it means | Cost |
|---|---|---|
| Label = "ever cancelled in the observed window" | Usable immediately | Train and test define the target differently |
| Label = "cancels in the N days after an anchor date inside the window" | Matches the business question | Shrinks usable data |

The competition submission used the first, which is what the released data
supports and what the leaderboard rewarded. **This analysis uses the second**:
every feature is computed over a lookback ending at an anchor date, and the
label is cancellation strictly *after* it.

**Constraint two: nothing before 1 October is observed.** A user's first event
in the logs is not when they joined — it is when the logs start. Tenure is
therefore measured from the registration date, and any "what happened in the
month before" comparison only uses users whose whole lookback falls inside the
window and after they registered. For the run-up analysis that keeps 1,108 of
4,271 cancellers: not the longest-tenured ones, but the ones who cancelled late
enough in the window for five weeks of history to be visible.

## 3. Approach

**a. A fair comparison for the run-up.** Setting cancellers against retained
users on an arbitrary date mixes cancellation up with three other things:

- *calendar* — activity drifts across the window for everyone;
- *tier* — cancellers skew paid, and paid users listen more;
- *presence* — you can only cancel on a day you visit, while a random day for a
  retained user is often a day they did not.

Each canceller is therefore compared with retained users **who had a session on
the same calendar day, on the same plan**. Day 0 is the day the cancellation
session started, so a session that crosses midnight is aligned the same way for
both groups. The unmatched comparison is kept alongside, because the gap between
the two is the measure of how much the comparison itself was producing.

**b. Decomposing the activity.** For each kind of activity — songs, skips,
adverts, thumbs up and down, errors, help, settings, downgrade and upgrade pages,
playlist adds — its share of all events is compared between weeks 2–4 before
and the final week. The final week stops at day -1: day 0 contains the
cancellation flow itself, which would otherwise show up as "behaviour".

**c. Normalising by exposure.** Cancellation rate per 100 sessions, by visit
frequency, within each tier — to separate "this user is riskier" from "this user
simply has more visits".

**d. Lift over both tails, and within tier.** For each signal, the cancellation
rate in its most extreme fifth against the base rate. Signals that differ by
tier are also checked within each tier, because pooling can hide or reverse them.

**e. From ranking to budget, against a baseline.** A cross-validated
gradient-boosted model ranks users by risk; the ranking is turned into a gains
curve and a net-value curve — and compared with a one-line rule, so the model is
credited only with what it adds.

## 4. What the data says

### I expected a fade. The first cut showed the opposite.

Aligned on the day of cancellation and set against retained users on random
dates, cancellers looked about three times as active, their activity climbed
through the final month, and it spiked in the last days. That is the reverse of
the expected story — and, it turns out, mostly a product of the comparison.

### Three explanations, tested

| Explanation | Test | Result |
|---|---|---|
| **Calendar** — everyone listens more as the weeks go on | Unmatched retained users over the same dates | Their activity rises too (+9% into the final week) |
| **Tier** — cancellers are more often paid, and paid users listen more | Match on tier | Accounts for part of the level gap |
| **Presence** — cancelling requires a visit | Match on having visited that same day | Closes the rest |

Matched on day, plan and a visit that day, the difference disappears:

| | Events/day, weeks 2–4 before | Final week | Change |
|---|---|---|---|
| Cancelled | 36.6 | 39.2 | +7% |
| Retained, matched | 35.8 | 38.3 | +7% |
| Retained, unmatched | 20.8 | 22.7 | +9% |

On day 0 the median jumps to 33.6 events for cancellers and 32.1 for matched
retained users. The spike is what any day with a visit looks like.

![run-up](figures/01_runup_matched.png)

### Is the activity made of errors, skips, adverts?

No. The mix of what cancellers do in their final week is the same as for
matched users: skips 12.5 → 12.2 per 100 events (matched: 12.6 → 12.3), errors
flat at 0.10, thumbs-down 1.02 → 0.97, downgrade pages 0.83 → 0.89 (matched:
0.74 → 0.79). Advert share falls about 30% for both groups alike. Nothing in the
experience gets worse before a cancellation.

![composition](figures/02_runup_composition.png)

### Then why do heavier users cancel more?

Because they visit more. Among paid users, the most frequent visitors come 12x
as often as the least frequent and cancel 5.5x as often — so **per visit**, their
cancellation rate is less than half the lightest group's. Every visit carries a
chance of ending in cancellation; engaged users have more visits, even though
each one is safer.

### What does carry signal

- **Downgrade-page visits** (1.6x lift) and **thumbs-down** (1.3x) — intent and
  dissatisfaction, though modest once exposure is accounted for.
- **Advert exposure, within tier** — 1.27x among free users and 1.35x among paid,
  but 0.81x when pooled. Paid users see few adverts and cancel more often, so
  pooling turns a real signal into an apparently protective one.
- **Not tenure.** Measured from registration, the median canceller has been a
  member for 51 days, and the cancellation rate is flat at 4.3–4.4% for the 97%
  of users between four weeks and six months old.
- A third of cancellations come from **free-tier** accounts, so this is not only
  a paid-subscription story.

### Ranking still sizes a budget — but mostly finds who will visit

Contacting the top 10% by risk reaches 26% of future cancellers. But a one-line
rule — *paid, and seen in the last 3 days* — flags 33% of users and already
catches 55%; the model, at the same share, catches 66%. The model is worth
11 points over the rule, not the whole curve.

![gains](figures/04_gains_curve.png)

## 5. What this means for the business

**Cancellation here is a decision point, not a symptom.** Users do not signal it
through a worse experience; up to the moment they cancel, they are getting the
same product in the same way as those who stay. Whatever tips the decision sits
outside these logs — price against perceived value, a billing date, an
alternative, a change in circumstances. This is the profile of *value* churn,
not *product-failure* churn.

Three consequences follow.

1. **An engagement-drop alert would not fire.** There is no decline to detect.
2. **Outbound campaigns driven by activity mostly reach frequent visitors** —
   who are, visit for visit, the least likely to cancel. An activity-based risk
   score is largely a "will be in the app soon" score.
3. **The one place the user reliably shows intent is the cancellation path.**
   42% of cancellations are reached from the downgrade page. That is where a save
   offer — a pause, a cheaper plan, a discount, a reminder of what they would
   lose — meets a user who is actually deciding, and it is cheap to A/B test.

## 6. Limitations

- **The label captures only explicit cancellation.** A user who quietly stops
  listening is recorded as retained. Silent abandonment is invisible here and,
  for a real business, likely the larger problem.
- **The logs are very likely simulated.** The schema matches Udacity's public
  "Sparkify" dataset, which was generated by an event simulator, and the data
  bears that out: 30% of cancellations are reached directly from an advert and
  11% from a thumbs-down — not navigation paths a real app offers. A chance of
  cancelling attached to each visit, with no change in behaviour beforehand, is
  exactly what a page-transition simulator produces. If so, the specific result
  is partly by construction. **The method is what transfers**: matched controls,
  decomposition, exposure normalisation, and benchmarking against a simple rule
  would each catch the same traps on real data.
- **Observational, not causal.** Every recommendation below is a hypothesis to
  test.
- **Matching is on day, plan and presence — not on everything.** Controls are
  drawn with replacement (5,540 draws from 4,195 users), and a small residual gap
  remains in median activity (18.7 vs 17.0 events/day four weeks out).
- **The run-up analysis covers a quarter of cancellers**, those whose full
  lookback is observed.
- **The save rate is assumed, not measured.** It drives the net-value curve and
  only an experiment can supply it.
- **51 days is one season, and the panel uses one anchor date.**
- **Contacting users is modelled as harmless beyond its cost.** A badly targeted
  retention message can remind a wavering user to cancel.

## 7. Recommended next steps

1. **Put the intervention where the decision happens.** Test a save flow on the
   downgrade and cancellation path — pause, cheaper plan, discount — against a
   randomised holdout. This one experiment also measures the save rate that
   every budget calculation here depends on.
2. **Do not build an engagement-decline alert.** There is nothing for it to catch.
3. **Hold any outbound model to the one-line rule.** Pay for the 11 points it
   adds, not for the part a rule already delivers.
4. **Test advert frequency for free users** — the one experience lever that
   carries signal within tier.
5. **Collect what the logs cannot show**: billing dates, plan price, and exit
   reasons. The drivers of a decision taken during an ordinary visit are not in
   behavioural data.
6. **Add a behavioural churn label** (no activity for N days) next to explicit
   cancellation, and re-run.
7. **Track cancellations per 1,000 sessions**, not only raw churn, so a rise in
   visits is not mistaken for a rise in risk.

---

## Reproducing

```bash
pip install -r requirements.txt
# place train.parquet in data/ — see data/README.md
python run_analysis.py --data data/train.parquet
```

Writes five charts to `figures/` and regenerates `findings.md` (about 30 seconds).

To exercise the code without the real data, in a copy of the repo with no
`data/train.parquet`:

```bash
python make_fixture.py && python run_analysis.py --horizon 14
```

The fixture is synthetic and exists only to prove the pipeline runs. No finding
is derived from it, `findings.md` stamps a warning on its face whenever it was
generated from fixture data, and `make_fixture.py` refuses to overwrite a real
`data/train.parquet`.

```
src/churn_analysis.py   Matched run-up, decomposition, panel, lift, exposure, gains, economics
run_analysis.py         Driver: charts + findings.md
make_fixture.py         Synthetic data, smoke-test only
findings.md             Generated — the numbers
```
