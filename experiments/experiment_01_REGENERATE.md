# Experiment 01: regenerate after the model correction

## Status

Do not publish or compare the old numerical results. They were generated under different information timing, horizon, BM settlement and Dynamic Containment assumptions.

## Frozen configuration to report

Record the exact values below before the run:

```text
Git commit:
Python version:
Dependency lock hash:
Input parquet SHA-256 hashes:
DA model: native hourly or original 30-minute model
Backtest period:
Number of scenarios:
Seeds:
Mean weight:
CVaR alpha:
Terminal SoC mode:
DC partial-edge-block rule:
```

Before training or backtesting, audit BM records on the composite publication
key `(startTime, createdDateTime)`. Exact copies may collapse only when all
fields consumed by that pipeline agree. A conflicting value at one composite
key is an input error; do not select one by row order before choosing the latest
vintage for a settlement period.

Weather predictors must match the exact last complete hourly timestamp before
the common D-1 cutoff. Do not forward-fill across a missing weather hour;
retain the missing value so incomplete feature rows are excluded visibly.

## Required strategies

Use at least the following controlled cases:

```text
median
previous complete comparable day
previous week
gaussian copula risk-neutral
gaussian copula with CVaR
historical joint OOF error paths risk-neutral
historical joint OOF error paths with CVaR
perfect foresight for a clearly labelled upper-bound diagnostic only
```

The primary historical-error-path results must use uncensored residual paths. If
`--error-cap-pct` is used, label the result as a winsorisation sensitivity and
report the percentile, market-specific cap values, and number of altered cells.
Admit a residual day to the pool only when DA, BM, DCL and DCH each contain the
exact 48 UTC half-hours from 00:00 through 23:30. Row count alone is not a
completeness test; duplicate or off-grid residual timestamps are invalid.

## Required forecast metrics

For DA and the BM proxy report MAE, q10 pinball loss, q90 pinball loss, 80 percent interval coverage and Winkler score at their evaluated timestamp resolution.

For DC Low and DC High report the same metrics on complete EFA blocks. Do not count eight duplicated half-hour observations as eight independent auction outcomes.
The DC LightGBM fits and early-stopping evaluations must use inverse EFA-block-size
row weights, normalized to mean one, so each auction has equal total influence despite
DST or missing-row differences.

For the direct spread model also report sign accuracy and magnitude-weighted sign accuracy at period, EFA-block and daily levels.

## Required economic metrics

Report results only on the intersection of clean dates solved by every compared strategy. Include:

```text
number of eligible calendar days
number of evaluated clean days
number and reason for excluded days
number of failed dates for every seed, separated by `failure_stage`
number of successful seeds for every date in each averaged artifact
total net revenue
annualised sample mean revenue
average daily net revenue
standard deviation across seeds
worst daily revenue
5th percentile daily revenue
average revenue in the worst 10 percent of days
DA revenue
BM SSP proxy revenue
DC Low revenue
DC High revenue
degradation cost
throughput and equivalent full cycles
```

A clean date must have `solve_failed=false`, `da_missing=false`,
`da_imputed=false`, and `state_reset=false` in every compared strategy. Apply
this definition to fixed-range diagnostic comparisons as well as the main
baseline experiment; a successfully solved date is not automatically clean.

An averaged date is eligible only when every requested seed solved it. The
averaged artifact must retain the union of attempted dates, with partial or
missing seed results marked as `failure_stage="seed_aggregation"`. Do not
average the successful subset for a date, because doing so conditions reported
revenue on numerical success. Retain the per-seed artifacts for the exact
underlying failure stages and reasons.

## Required sensitivity checks

Run at least these sensitivities:

```text
scenario counts: 20, 50, 100 for risk-neutral runs; CVaR runs only where
                 (1 - alpha) × scenario count >= 5
at least five distinct non-negative random seeds
risk-neutral versus CVaR
native-hourly versus original 30-minute DA forecaster
1 MW / 1 MWh versus at least one longer-duration battery
with and without direct spread forecasts
uncalibrated versus leakage-safe calibrated spread intervals
```

Do not count repeated seed IDs as independent replications. If any requested
seed returns no dated result rows, fail the experiment instead of silently
continuing with a smaller replication set.

## Interpretation rules

Do not call the BM proxy real BM revenue. Do not infer that DA arbitrage is structurally unprofitable from the DA accounting component alone because DA charging may support another modelled revenue stream. Do not claim copula dependence improves a risk-neutral fully first-stage plan without demonstrating the mechanism. Do not scale a 1 MW result linearly to a large battery without qualifying price impact, acceptance and network access.

## Result tables

Populate these tables only from regenerated files.

### Forecast performance

| Market | Resolution | Observations | MAE | Pinball q10 | Pinball q90 | Coverage | Winkler |
|---|---:|---:|---:|---:|---:|---:|---:|
| DA | | | | | | | |
| BM SSP proxy price | | | | | | | |
| DC Low | EFA block | | | | | | |
| DC High | EFA block | | | | | | |

### Strategy performance

| Strategy | Clean days | Seeds | Net revenue | Annualised sample mean | Daily mean | Across-seed uncertainty | Worst day | P05 | Worst 10 percent mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | | | | | | | |

### Revenue attribution

| Strategy | DA | BM SSP proxy | DC Low | DC High | Degradation | Net |
|---|---:|---:|---:|---:|---:|---:|
| | | | | | | |

## Publication decision

A paper draft should begin only after the full run can be reproduced from a clean clone. The draft must carry the BM proxy limitation into the title or abstract when that stream is central to the contribution.
