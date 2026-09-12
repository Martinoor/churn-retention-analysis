# Who is about to cancel, and what should we do about it?

A behavioural analysis of subscriber churn on a music-streaming service, using
51 days of raw event logs from ~19k users.

**The short version:** cancellation is not a sudden decision. It is the end of a
measurable decline that begins weeks earlier, which means there is a window in
which it can be interrupted — and the analysis sizes both the window and the
budget worth spending inside it.

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
2. **If it is predictable, how early?** A model that fires the day before
   cancellation is useless; nobody can act on it.
3. **Who is worth contacting?** Contacting everybody costs more than it saves.

The original competition asked only a narrower version of question 1 — maximise
balanced accuracy on a held-out set. That is a leaderboard question, not a
business one. This analysis answers all three.

## 2. The data, and one real constraint

51 days of event logs (2018-10-01 to 2018-11-20). One row per user interaction:
page viewed, timestamp, subscription tier, session, track metadata, device,
location. A user is treated as churned when they reach the
`Cancellation Confirmation` page.

**The constraint worth being upfront about.** The competition brief asked for
churn "in the 10 days following 2018-11-20" — but the released event logs *end*
on 2018-11-20. There is no data in the window we were asked to predict. Two
resolutions exist:

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

That choice costs accuracy and buys interpretability. It is the reason the
"days of warning" result in §1 of the findings means anything at all.

## 3. Approach

Three steps, each answering one of the questions above.

**a. Event-time alignment.** Each cancelling user is aligned on their own
cancellation day rather than on the calendar, so individual declines do not
average away. Retained users are given a *placebo anchor* drawn from the
cancellers' anchor distribution — without this, the comparison would simply
recover the fact that the observation window ends.

**b. Lift over both tails.** For each behavioural signal, the cancellation rate
in its most extreme fifth is compared with the base rate. Both tails are tested,
because some signals predict churn when *high* (adverts served, thumbs-down) and
others when *low* (songs played, activity trend). Lift is reported as a ratio,
which means the same thing to a non-technical reader regardless of the
underlying units.

**c. From ranking to budget.** A gradient-boosted model over the panel ranks
users by risk (cross-validated, so no user is scored by a model that saw them).
The ranking is then converted into a gains curve — what share of churners you
reach for a given share of users contacted — and finally into a net-value curve
given a contact cost, a subscriber value and an assumed save rate.

Accuracy is deliberately *not* the headline. The model is a ranking device for
sizing a budget, not the finding.

## 4. Limitations

Stated plainly, because several of these bound what the analysis can claim.

- **Observational, not causal.** The analysis can show that heavy advert
  exposure precedes cancellation. It cannot show that reducing adverts prevents
  it — both could follow from a user drifting toward the free tier. Every
  recommendation below is therefore a *hypothesis to test*, not a proven lever.
- **The save rate is assumed, not measured.** It is the single most influential
  input to the net-value curve and the only one no amount of historical data can
  supply. The figure is presented as a sensitivity, not a point estimate.
- **51 days is one season.** No annual cycle, no post-holiday effect, no
  price-change response is observable here.
- **One anchor date.** Findings are computed at a single anchor; a production
  system would need them to be stable across many, which `Window` supports but
  this write-up does not exercise.
- **Contacting users is modelled as costless beyond the contact price.** In
  reality a badly-targeted retention email can *remind* a wavering user to
  cancel. That risk is real and unquantified here.

## 5. Recommended next steps

1. **Run the experiment the analysis cannot substitute for.** Randomise the
   top-risk decile into treatment and holdout, and measure the actual save rate.
   Until that number exists, the net-value curve is an argument, not a forecast.
2. **Intervene on the decline, not on the cancel click.** The decay curve gives
   the window; trigger on sustained drops in activity within it.
3. **Test the advert lever specifically.** Of the leading signals, advert
   exposure is the one the business directly controls. It is the cheapest
   causal test available and the one with a clear action attached.
4. **Re-anchor weekly.** Rerun across rolling anchors to confirm the signals are
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
finding is derived from it.**

```
src/churn_analysis.py   Windowing, panel construction, lift, gains, economics
run_analysis.py         Driver: charts + findings.md
make_fixture.py         Synthetic data, smoke-test only
findings.md             Generated — the numbers
```
