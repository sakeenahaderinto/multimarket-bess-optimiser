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

## Required forecast metrics

For DA and the BM proxy report MAE, q10 pinball loss, q90 pinball loss, 80 percent interval coverage and Winkler score at their evaluated timestamp resolution.

For DC Low and DC High report the same metrics on complete EFA blocks. Do not count eight duplicated half-hour observations as eight independent auction outcomes.

For the direct spread model also report sign accuracy and magnitude-weighted sign accuracy at period, EFA-block and daily levels.

## Required economic metrics

Report results only on the intersection of clean dates solved by every compared strategy. Include:

```text
number of eligible calendar days
number of evaluated clean days
number and reason for excluded days
number of solver failures for every seed
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

## Required sensitivity checks

Run at least these sensitivities:

```text
scenario counts: 20, 50, 100
at least five random seeds
risk-neutral versus CVaR
native-hourly versus original 30-minute DA forecaster
1 MW / 1 MWh versus at least one longer-duration battery
with and without direct spread forecasts
uncalibrated versus leakage-safe calibrated spread intervals
```

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
