# Multimarket BESS Optimiser

## Overview

The Multimarket BESS Optimiser is a pre-market backtesting tool for optimising Battery Energy Storage Systems (BESS) across multiple energy markets in Great Britain. This repository demonstrates an end-to-end pipeline, moving from raw data ingestion and feature engineering to probabilistic price forecasting and scenario-based Mixed-Integer Programming (MIP).

The challenge in battery optimisation is that multiple markets compete for the same physical asset. This model simultaneously optimises dispatch across three distinct revenue streams:

- **Day-Ahead (DA) Market:** Wholesale energy arbitrage.
- **Dynamic Containment (DC):** Frequency response ancillary services (DC Low and DC High).
- **Balancing Mechanism (BM):** Real-time grid balancing, modelled as a System Sell Price proxy (see Known Limitations).

By leveraging quantile regression and stochastic optimisation, the model finds the dispatch allocation that maximises expected net revenue, accounting for degradation costs across a distribution of possible price scenarios.

---
## Architecture

```mermaid
flowchart LR
A[Data Ingestion] --> B[Feature Engineering]
B --> C[Probabilistic Forecasting]
C --> D[Scenario Generation]
D --> E[Pyomo MIP Optimiser]
E --> F[Walk-Forward Backtest]
```

1. **Ingestion**: Retrieves raw historical market data from Elexon BMRS (system prices), Electricity Maps (N2EX day-ahead prices), and NESO (DC auction results).
2. **Feature Pipeline**: Merges and aligns the data sources at a 30-minute resolution. Real-time features are restricted to a 09:00 UTC snapshot on D-1, and lagged so no decision uses information unavailable at day-ahead gate closure.
3. **Forecasting**: Trains LightGBM quantile regression models ($q_{10}$, $q_{50}$, $q_{90}$) using walk-forward cross-validation to predict DA, BM, and DC clearing prices, along with a BM-DA spread model and a DA model with an hourly resolution.
4. **Scenario Generation**: A Gaussian copula with Cholesky decomposition samples correlated price scenarios across all four markets, preserving cross-market dependence structure estimated from historical data. A historical error-path sampler is also available as an alternative.
5. **Optimisation**: A Pyomo-based MIP that maximises expected net revenue across generated price scenarios over a 24-hour delivery day. Solved using the HiGHS solver.
6. **Backtesting**: Walks forward day-by-day, solving the MIP using out-of-sample forecast scenarios and settling revenue against actual, realised market prices to accurately evaluate strategy performance. Any window with missing settlement periods is dropped from the backtest.

Copula dependence falls back to the configured default matrix and zero temporal
persistence only when OOF residual inputs are unavailable or fewer than 500
publication-eligible observations exist. Malformed, non-finite, or degenerate residual
data raises an error rather than silently changing the stochastic model.

---

## Markets Modelled

| Market                       | Revenue Mechanism                                           | Commitment Timing                                                                  |
| ---------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Day-Ahead (DA)**           | Buy/sell against N2EX prices                                | Fixed schedule before gate closure                                                 |
| **Dynamic Containment (DC)** | Availability payment per MW reserved for frequency response | Committed per 4-hour EFA block in daily auction                                    |
| **Balancing Mechanism (BM)** | Realised System Sell Price on submitted volume              | Opportunistic; assumes 100% acceptance as a simplification                         |

---

## Key Features

- **Leakage-controlled validation.** One shared market calendar, restricted real-time features taken from a 09:00 UTC D-1 snapshot, exact-timestamp settlement grids, and a hard failure on missing settlement periods.
- **Realistic operational constraints.** One-way efficiency applied on each side of the round trip, directional power limits, mutually exclusive charge and discharge modes via binary variables, and a cyclic terminal state of charge so the battery does not artificially deplete at the end of each solve.
- **Dynamic Containment modelling.** DC commitments held constant across each 4-hour EFA block, a 15-minute energy headroom requirement, and separate discharge-direction and charge-direction power limits for DC Low and DC High. Incomplete EFA blocks at the edges of a midnight-based horizon are disabled by default. This is conservative and avoids committing on a partial block. DC scenario prices use the predecessor auction's zero floor before 1 November 2023 and permit negative prices after the EAC transition.
- **Probabilistic forecasting.** Quantile LightGBM models under walk-forward cross-validation, with strict lag controls. DC clearing outcomes are grouped by EFA block so one repeated auction result cannot straddle model-fitting and early-stopping validation subsets, and inverse block-size weights give every auction equal total training influence.
- **Baseline benchmarking.** The full probabilistic model is compared against deterministic baselines (median forecast, previous-day naive, previous-week naive) on a shared clean-date evaluation set.
- **Auditable backtest failures.** Non-strict runs retain data, scenario, optimisation, and settlement failures as dated result rows with a failure stage and reason; failed dates cannot disappear from aggregate failure counts.

---

## Optimisation Model

**Decision variables**

- `da_charge`, `da_discharge`: day-ahead schedule (MW), first-stage.
- `dc_low`, `dc_high`: DC capacity per EFA block (MW), first-stage.
- `bm_offer`: BM proxy volume submitted (MW), first-stage. Under full acceptance, `bm_dispatch = bm_offer`.
- `soc`: state of charge trajectory.
- `charge_mode`: binary variable preventing simultaneous charging and discharging.

**Objective.** Maximise expected net revenue across scenarios minus expected degradation cost.

**Key constraints**

- Energy balance with asymmetric round-trip efficiency.
- DC commitments constant within each EFA block.
- DC energy headroom: enough state of charge to physically deliver the committed response.
- Cyclic terminal state of charge (return to the initial value at the end of each 24-hour solve).

The full mathematical formulation is in [`experiments/optimisation_model_formulation.md`](experiments/optimisation_model_formulation.md).

### Interpreting the stochastic model

Under the full-acceptance BM proxy, every physical decision is scenario-independent. With `mean_weight=1.0`, the risk-neutral optimisation is therefore driven by scenario-average prices, and the Gaussian copula does not by itself create operational robustness. Cross-market dependence becomes decision-relevant when a CVaR term is active, or under a properly identified recourse model with real BM acceptance uncertainty. Empirical claims about sampled means and risk-aware objectives should rest on multi-seed results, not a single run.

When CVaR is active, the optimiser requires at least five effective tail scenarios: `(1 - cvar_alpha) × n_scenarios >= 5`. At the default `cvar_alpha=0.90`, use at least 50 scenarios. This is a minimum adequacy guard, not a substitute for scenario-count and multi-seed sensitivity analysis.

---
## Quickstart

**1. Install**

Python 3.11 or later is recommended.

```bash
git clone https://github.com/sakeenahaderinto/multimarket-bess-optimiser
cd multimarket-bess-optimiser
uv sync
```

Create a `.env` from `.env.example` only when running ingestion.

**2. Input data**

Ingestion writes raw data via `backfill.py`. The processed files the pipeline consumes are at:

```text
data/processed/bmrs_processed.parquet
data/processed/day_ahead_processed.parquet
data/processed/dc_auction_processed.parquet
data/raw/weather/weather_backfill.parquet
```

Expected columns:

| File | Columns |
|---|---|
| `bmrs_processed.parquet` | `startTime`, `createdDateTime`, `systemSellPrice`, `totalAcceptedOfferVolume`, `netImbalanceVolume` |
| `day_ahead_processed.parquet` | `datetime`, `value` |
| `dc_auction_processed.parquet` | `delivery_start`, `service`, `clearing_price` |
| `weather_backfill.parquet` | `date`, `temperature_2m`, `shortwave_radiation`, `wind_speed_10m`, `cloud_cover` |

`verify_inputs.py` checks these files before a run.

Timestamp-indexed DA, forecast, and weather inputs may contain repeated identical
observations, which are collapsed. Conflicting values at the same timestamp are rejected
instead of being selected by file or row order. BM publication revisions and DC services
use their explicit publication-time and service keys rather than this generic rule.
Quantile forecast artifacts must also contain finite numeric values satisfying
`q10 <= q50 <= q90` at every timestamp; invalid distributions are rejected before
calibration, upsampling, scenario generation, or backtesting.

**3. Run the pipeline, in order**

```bash
uv run data_quality.py          # DA imputation audit and quality flags
uv run train_forecasters.py     # feature table + DA, BM, DC Low, DC High quantile models
uv run train_da_hourly.py       # native-hourly DA alternative
uv run train_spread.py          # direct BM-minus-DA spread model
uv run calibrate_spread.py      # optional calibration of the spread model
```

**4. Baseline comparison (multi-seed)**

```bash
uv run run_baselines.py --period cv  --use-hourly-da --n-scenarios 50 --seeds 0 1 2 3 4
uv run run_baselines.py --period oos --use-hourly-da --n-scenarios 50 --seeds 0 1 2 3 4
```

Multi-seed artifacts retain every date attempted by any seed. Revenue is
averaged only when every seed solved the date; otherwise the aggregate row is
marked as a `seed_aggregation` failure and reports the number of successful
seed runs. Seed IDs must be distinct non-negative integers, and every requested
run must produce a dated artifact. Exact underlying reasons remain in the
per-seed artifacts.

**5. Single-strategy backtest**

```bash
uv run run_backtest.py --period oos --use-hourly-da --n-scenarios 50 --strict
```

**6. Tests**

```bash
uv run pytest -q
```

> Forecast files written by `train_forecasters.py` are historical out-of-fold and out-of-sample predictions for backtesting only. They are not live next-day forecasts.

---
## Project Structure

```text
multimarket-bess-optimiser/
├── ingestion/            # Fetchers: BMRS, Electricity Maps, NESO/ESO, solar, weather
├── features/             # Feature engineering and lag alignment
│   ├── pipeline.py               # shared 30-min feature table
│   ├── pipeline_da_hourly.py     # native-hourly DA feature table
│   ├── cyclical_time.py, lag_features.py, market_features.py,
│   └── rolling_stats.py, weather_features.py
├── forecasting/          # LightGBM quantile forecasters
│   ├── base.py                   # shared training + walk-forward CV
│   ├── da_prices.py, da_prices_hourly.py
│   ├── bm_price.py, bm_da_spread.py
│   └── dc_tender.py
├── optimiser/            # Pyomo MIP model and scenario generation
├── baselines/            # Deterministic baseline scenario builders
├── backtest/             # Walk-forward engine and revenue settlement
├── tests/                # Unit tests
├── experiments/          # Model formulation and experiment templates
├── data/                 # Processed inputs, features, forecasts, results
├── config.py             # Central configuration
├── market_calendar.py    # Shared market calendar (UTC/local time, EFA blocks)
├── excluded_dates.py     # Dates excluded from error-path sampling
└── *.py                  # Pipeline, backtest and diagnostic scripts (see below)
```

---

## Scripts Reference

**Pipeline scripts**

| Script | Description |
|---|---|
| `backfill.py` | Pulls historical raw data from Elexon BMRS, Electricity Maps, NESO and Open-Meteo. Run once to populate the raw data directory. |
| `verify_inputs.py` | Validates that the required input files exist and match the expected schema. |
| `data_quality.py` | DA price imputation audit. Flags every half-hourly slot where the DA price was forward-filled by more than one step and classifies it by how it is used (settlement window, training target or lag feature). |
| `train_forecasters.py` | Builds the feature table and trains the DA, BM, DC Low and DC High quantile models with walk-forward CV. Saves out-of-fold predictions (2022–2024) and out-of-sample predictions (2025). |
| `train_da_hourly.py` | Trains an alternative DA forecaster at native hourly resolution, upsampled to 30 minutes. Built to test whether spurious sub-hourly forecast variance in the 30-minute DA model affects arbitrage revenue. |
| `train_spread.py` | Trains a model directly on the BM-minus-DA spread, rather than deriving the spread from two marginal forecasts. |
| `calibrate_spread.py` | Calibrates the direct spread model's predictions. |
| `run_baselines.py` | Runs the full copula strategy against a deterministic median baseline and two naive lag baselines (previous-day, previous-week) on the same valid dates, across multiple seeds. |
| `run_backtest.py` | Runs the stochastic optimiser in walk-forward mode for a single period (CV or OOS), settling each day's decisions against realised prices. |
| `run_fixed_month_comparison.py` | Compares strategies within a fixed calendar month to isolate regime-specific behaviour, such as high-volatility winter periods. It retains both per-seed and failure-aware averaged artifacts and applies the same shared clean-date rule as the main baseline comparison. |

**Diagnostics and experiment scripts**

| Script | Description |
|---|---|
| `historical_error_scenarios.py` | Historical error-path scenario generator. Samples complete joint forecast-error days (DA, BM, DC Low, DC High) from a pool, preserving temporal and cross-market dependence by construction rather than through an estimated correlation matrix. Paths are uncensored by default; residual winsorisation is available only as an explicit sensitivity option. |
| `check_q50_spread_sign.py` | Tests whether BM-DA q50 spread sign mismatches are a systematic, volatility-conditional failure or an isolated event, by splitting daily sign accuracy by volatility quartile. |
| `check_pool_spike_magnitude.py` | Compares the error pool's typical magnitude in the 16:00–19:00 spike window against the realised spread on a target day, to test whether passivity in the copula strategy is explained by the pool lacking large enough errors. |
| `sanity_check_error_pool.py` | Pre-flight check of the error-path pool: size, magnitude plausibility, shape of sampled paths, and one full scenario construction. |
| `inspect_day_schedule.py` | Prints the per-period dispatch schedule (DA, BM, DC, SoC) for a single backtest day by reading the solved Pyomo model's decision variables. |
| `check_hourly_lag_alignment.py` | Manual sanity check of lag construction in the hourly DA feature pipeline before a full training run. |
| `excluded_dates.py` | Registry of dates excluded from the error-path sampling pool: UK clock-change dates and genuine BM data-gap days. |

---
## Known Limitations

**BM is an SSP proxy.** There is no BOA acceptance model, offer-price curve, transmission constraint, loss model, availability failure, transaction fee, imbalance exposure or market impact. Full acceptance overstates what a real unit would earn from the BM.

**Single price-taking asset.** The model represents one 1 MW battery that does not move prices. Revenue does not scale linearly to a large asset, because acceptance, network access and price impact change with size.

**Training data coverage.** Models are trained on 2022–2024. DC auction data before 2022 has incomplete market coverage as the service was still rolling out, so 2022 is the study start. The CV backtest (2022–2024) uses out-of-fold predictions to avoid look-ahead bias. The OOS backtest (2025) evaluates unseen data, but regime changes in 2025 may degrade forecast accuracy.

**DC price resolution.** DC prices clear per 4-hour EFA block, not per 30-minute period. Block prices are forward-filled to 30 minutes for consistency with the optimiser, so DC forecast accuracy should be read at block level. Day-ahead prices are expanded from hourly to 30 minutes with a single fill, and longer source gaps remain missing.

**No same-origin multi-day lookahead.** Earlier versions used a 48-hour lookahead. It was removed because the feature files did not contain genuine same-origin 48-hour forecasts. Each solve covers one 24-hour UTC delivery day.

**Weather inputs.** Weather features are lagged realised observations, not archived numerical weather forecasts. A stronger forecasting study would use forecast vintages actually available at each decision time.

**Degradation convention.** Degradation cost uses `replacement_cost / (2 × cycle_life)` applied to both charge and discharge throughput, treating one full cycle as one charge plus one discharge. Replacement cost must be finite and non-negative, and cycle life must be finite and positive.

---

