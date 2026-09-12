# Who is about to cancel, and what should we do about it?

A behavioural analysis of subscriber churn on a music-streaming service, using
51 days of raw event logs from ~19k users.

**The short version:** the intuitive story about churn — that people quietly
disengage for weeks, and cancelling is the end of a long fade — is not what this
data shows. Cancellers are the *more* active users, roughly twice as active as
those who stay, right up to the day they leave. What actually marks them out is
being **new**: the median canceller has 18 days of history. That changes the
recommendation from "detect the fade" to "fix the first three weeks."

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
3. **Who is worth contacting?** Contacting everybody costs more than it saves.

The original competition asked only a narrower version of question 1 — maximise
balanced accuracy on a held-out set. That is a leaderboard question, not a
business one. This analysis answers all three, and the answer to the second one
is not the one I expected.

## 2. The data, and two real constraints

51 days of event logs (2018-10-01 to 2018-11-20). One row per user interaction:
page viewed, timestamp, subscription tier, session, track metadata, device,
location. A user is treated as churned when they reach the
`Cancellation Confirmation` page.

**Constraint one: the prediction window is not in the data.** The competition
brief asked for churn "in the 10 days following 2018-11-20" — but the released
event logs *end* on 2018-11-20. There is no data in the window we were asked to
predict. Two resolutions exist:

| Option | What it means | Cost |
|---|---|---|
| Label = "ever cancelled in the observed window" | Usable immediately | Train and test define the target differently |
| Label = "cancels in the N days after an anchor date inside the window" | Matches the business question | Shrinks usable data |

The competition submission used the first, which is what the released data
supports and what the leaderboard rewarded. **This analysis uses the second**,
because the business question is inherently forward-looking: every feature is
computed over a lookback window ending at an anchor date, and the label is
cancellation strictly *after* that anchor. Nothing after the anchor can enter a
feature.

**Constraint two: most cancellers are too new to have a run-up.** The median
canceller has only 18.4 days of history before cancelling; 68% have fewer than
28 days, and 25% fewer than 7. This matters more than it sounds. Any "what did
the month before cancellation look like" question can only be asked of the
minority with a month of history behind them — 923 of 4,271 cancellers. Treating
the days before a user's first event as days they were "inactive" does not
measure disengagement, it measures account age, and an earlier version of this
analysis made exactly that mistake.

## 3. Approach

Three steps, each answering one of the questions above.

**a. Event-time alignment.** Each cancelling user is aligned on their own
cancellation day rather than on the calendar, so individual trajectories do not
average away. Two controls matter: retained users are given a *placebo anchor*
drawn from the cancellers' anchor distribution, without which the comparison
would simply recover the fact that the observation window ends; and only users
whose history spans the full window are included, without which the curve
measures tenure instead of behaviour.

**b. Lift over both tails.** For each behavioural signal, the cancellation rate
in its most extreme fifth is compared with the base rate. Both tails are tested,
because a signal may predict churn when high (downgrade-page visits) or when low
(days since last seen). Lift is reported as a ratio, which means the same thing
to a non-technical reader regardless of the underlying units.

Signals are also split by **kind**, because two different claims get confused
otherwise. *Volume* signals say how much a user did; *decline* signals say which
way they were heading. A volume signal with high lift tells you who cancels, not
that anything deteriorated first.

**c. From ranking to budget.** A gradient-boosted model over the panel ranks
users by risk (cross-validated, so no user is scored by a model that saw them).
The ranking is then converted into a gains curve — what share of churners you
reach for a given share of users contacted — and finally into a net-value curve
given a contact cost, a subscriber value and an assumed save rate.

Accuracy is deliberately *not* the headline. The model is a ranking device for
sizing a budget, not the finding.

## 4. What the data actually says

The live numbers are in [findings.md](findings.md); this is what they mean.

**There is no pre-cancellation decline.** Among cancellers with a full month of
history, median activity *rises* over the final four weeks (23 → 34 events/day)
while retained users sit flat around 11. Cancellers are the heavier group at
every point in the window. The sharp lift in the last 48 hours is an artefact of
the alignment — cancelling requires an active session, a placebo anchor day does
not — and should not be read as behaviour. A purpose-built decline metric (the
slope of daily activity across the lookback) shows no signal in either tail.

**So the "intervention window" framing does not survive.** There is no fade to
catch early, which removes the trigger the earlier version of this write-up
recommended building.

**What does predict cancellation is volume and recency, pointing the same way.**
Users in the top fifth by active days, distinct artists, songs, events or
sessions cancel at roughly 2.1–2.2x the base rate, and users seen *most*
recently cancel more, not less. Part of this is tier: paid users cancel at 5.8%
against 2.8% for free users, and the heavy-usage quintile is 85% paid against
56% overall. But it is not only tier — within paid users alone, high volume
still carries 1.83x lift.

**The genuine behavioural signals are intent, not fatigue.** Downgrade-page
visits (1.6x) and thumbs-down (1.3x) carry real lift. Notably, **advert exposure
does not** — at 0.77x it sits *below* the base rate. An earlier version of this
write-up recommended testing the advert lever, and the synthetic fixture was
built with advert exposure rising before cancellation; both encoded the same
prior expectation, written before the analysis had been run against the real
logs. The data does not support it, and the recommendation is withdrawn.

**Ranking still works well enough to size a budget.** Contacting the top 5% by
risk reaches 18% of everyone who will cancel, at 16% precision against a 4.5%
base rate. That does not depend on the decline story and is unaffected by its
collapse.

## 5. Limitations

Stated plainly, because several of these bound what the analysis can claim.

- **The label captures only explicit cancellation.** A user who quietly stops
  listening and never returns is recorded as retained. This is the single most
  important caveat here: it biases the entire analysis toward engaged, paying
  users — the ones who bother to formally cancel — and it is a plausible part of
  why heavy usage predicts cancellation. Silent abandonment is invisible in this
  data, and for a real business it is likely the larger problem.
- **Observational, not causal.** Every recommendation below is a *hypothesis to
  test*, not a proven lever.
- **The save rate is assumed, not measured.** It is the most influential input
  to the net-value curve and the only one no amount of historical data can
  supply. It is presented as a sensitivity, not a point estimate.
- **The run-up analysis covers a minority of cancellers.** Requiring a full
  month of history keeps 923 of 4,271. The 78% excluded are the short-tenure
  cancellers — arguably the more important population, and one that 51 days of
  data cannot characterise this way.
- **51 days is one season.** No annual cycle, no post-holiday effect, no
  price-change response is observable here.
- **One anchor date.** Findings are computed at a single anchor; a production
  system would need them stable across many, which `Window` supports but this
  write-up does not exercise.
- **Contacting users is modelled as costless beyond the contact price.** In
  reality a badly-targeted retention email can *remind* a wavering user to
  cancel. That risk is real and unquantified here.

## 6. Recommended next steps

1. **Treat this as an onboarding problem, not a disengagement problem.** The
   median canceller is under three weeks old. Effort spent detecting fade in
   tenured users is aimed at the wrong population; the first few weeks are where
   the cancellations are.
2. **Run the experiment the analysis cannot substitute for.** Randomise the
   top-risk decile into treatment and holdout, and measure the actual save rate.
   Until that number exists, the net-value curve is an argument, not a forecast.
3. **Trigger on intent, not on decline.** Downgrade-page visits are the
   strongest actionable behavioural signal and have a clear action attached.
   There is no decline signal worth triggering on.
4. **Fix the label before trusting any of this in production.** Define a
   behavioural churn label (no activity for N days) alongside explicit
   cancellation, and re-run. If silent abandonment behaves differently — and the
   volume result suggests it will — the conclusions here describe only the
   minority who cancel formally.
5. **Re-anchor weekly.** Rerun across rolling anchors to confirm the signals are
   stable rather than an artefact of one date.

---

## Reproducing

```bash
pip install pandas numpy pyarrow scikit-learn matplotlib
# place train.parquet in data/ — see data/README.md
python run_analysis.py --data data/train.parquet
```

Writes four charts to `figures/` and regenerates `findings.md`.

To exercise the code without the real data:

```bash
python make_fixture.py && python run_analysis.py --horizon 14
```

The fixture is synthetic and exists only to prove the pipeline runs. **No
finding is derived from it**, and `findings.md` stamps a warning on its face
whenever it was generated from fixture data. Worth knowing why that guard
matters: the fixture's invented patterns were built to match what this write-up
originally expected to find, so a fixture run looks like a confirmation of the
hypothesis rather than a test of it. See §4 for where the real data disagreed.

```
src/churn_analysis.py   Windowing, panel construction, lift, gains, economics
run_analysis.py         Driver: charts + findings.md
make_fixture.py         Synthetic data, smoke-test only
findings.md             Generated — the numbers
```
