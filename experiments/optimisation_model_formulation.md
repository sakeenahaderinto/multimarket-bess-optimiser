# Revised mathematical formulation

## 1. Scope

The model represents one or more price-taking batteries over one 24-hour UTC delivery day. The horizon contains 48 half-hour periods. Day-ahead energy, Dynamic Containment and a stylised BM full-acceptance SSP proxy compete for the same battery.

The BM term is not actual Bid-Offer Acceptance settlement. It assumes the proxy offer volume is accepted in full and valued at the System Sell Price. This assumption is used only to make the physical energy path and the reported proxy cashflow internally consistent.

## 2. Sets and indices

Let:

- $t \in \mathcal{T}=\{0,\ldots,47\}$ denote half-hour periods.
- $\tau \in \mathcal{T}^{SOC}=\{0,\ldots,48\}$ denote state boundaries.
- $b \in \mathcal{B}$ denote batteries.
- $s \in \mathcal{S}=\{0,\ldots,N-1\}$ denote sampled price scenarios.

The period duration is $\Delta=0.5$ hours.

## 3. Battery parameters

For battery $b$:

- $P_b$ is rated power in MW.
- $C_b$ is usable energy capacity in MWh.
- $e_b^0$ is initial state of charge as a fraction.
- $\eta_b^{RTE}$ is round-trip efficiency.
- $\eta_b=\sqrt{\eta_b^{RTE}}$ is one-way efficiency.
- $N_b^{cyc}$ is design cycle life.
- $C^{rep}$ is replacement cost in pounds per MWh of capacity.

The linear degradation coefficient is:

$$
\kappa_b=\frac{C^{rep}}{2N_b^{cyc}}.
$$

## 4. Scenario prices

For every $s,t$ the optimiser receives:

$$
p_{s,t}^{DA},\quad p_{s,t}^{BM},\quad p_{s,t}^{DCL},\quad p_{s,t}^{DCH}.
$$

The BM price is an SSP proxy input. DC prices use pounds per MW per hour. DA and BM inputs use pounds per MWh.

## 5. Decision variables

All physical decisions are made before the sampled price scenario resolves:

- $x_{t,b}^{c}\ge0$: DA charging power.
- $x_{t,b}^{d}\ge0$: DA discharging power.
- $v_{t,b}^{off}\ge0$: BM proxy offer power.
- $y_{t,b}^{L}\ge0$: DC Low reservation.
- $y_{t,b}^{H}\ge0$: DC High reservation.
- $z_{t,b}\in\{0,1\}$: DA charge-mode indicator.

For implementation compatibility the model keeps:

- $v_{t,b,s}^{disp}\ge0$: BM proxy dispatch.
- $e_{\tau,b,s}$: state of charge.

Full proxy acceptance imposes:

$$
v_{t,b,s}^{disp}=v_{t,b}^{off}\qquad \forall t,b,s.
$$

Therefore the physical dispatch and SoC are equal across scenarios when the model is solved exactly.

## 6. Scenario net revenue

For scenario $s$:

$$
\begin{aligned}
R_s = \sum_{t,b}\Delta\big[&
 p_{s,t}^{DA}(x_{t,b}^{d}-x_{t,b}^{c})
 +p_{s,t}^{BM}v_{t,b,s}^{disp}\\
&+p_{s,t}^{DCL}y_{t,b}^{L}
 +p_{s,t}^{DCH}y_{t,b}^{H}\\
&-\kappa_b(x_{t,b}^{c}+x_{t,b}^{d}+v_{t,b,s}^{disp})
\big].
\end{aligned}
$$

DC activation energy and degradation are not modelled because the available input represents availability prices rather than activation records. This omission must be disclosed.

## 7. Objective

The risk-neutral objective is:

$$
\max \frac{1}{N}\sum_s R_s.
$$

With downside CVaR active the model introduces $\zeta\in\mathbb{R}$ and $u_s\ge0$:

$$
u_s\ge \zeta-R_s.
$$

The lower-tail revenue CVaR is:

$$
\operatorname{CVaR}_{\alpha}(R)
=\zeta-\frac{1}{(1-\alpha)N}\sum_s u_s.
$$

The combined objective is:

$$
\max \lambda\frac{1}{N}\sum_s R_s
 +(1-\lambda)\operatorname{CVaR}_{\alpha}(R).
$$

When $\lambda=1$ and all physical decisions are first-stage the objective depends on scenario-average prices. Copula dependence does not create a distinct recourse policy in that case. It can affect a CVaR objective and finite-sample estimates.

For empirical CVaR, the implementation requires at least five effective tail
scenarios:

$$
(1-\alpha)N\ge5.
$$

At $\alpha=0.90$, this requires at least 50 scenarios. This is a minimum
adequacy guard; reported risk results still require scenario-count and
multi-seed sensitivity analysis.

## 8. State of charge

Initial condition:

$$
e_{0,b,s}=e_b^0.
$$

Dynamics:

$$
e_{t+1,b,s}=e_{t,b,s}
 +\frac{\eta_b x_{t,b}^{c}\Delta}{C_b}
 -\frac{(x_{t,b}^{d}+v_{t,b,s}^{disp})\Delta}{\eta_b C_b}.
$$

Bounds:

$$
e^{min}\le e_{\tau,b,s}\le e^{max}.
$$

The publication configuration uses a cyclic terminal condition:

$$
e_{48,b,s}=e_b^0.
$$

This prevents artificial end-of-day depletion. A floor or tolerance band can be selected for sensitivity analysis.

## 9. Directional power constraints

DC Low is treated as a discharge-direction service. DC High is treated as a charge-direction service.

Discharge direction:

$$
x_{t,b}^{d}+v_{t,b,s}^{disp}+y_{t,b}^{L}\le P_b.
$$

Charge direction:

$$
x_{t,b}^{c}+y_{t,b}^{H}\le P_b.
$$

The model does not sum opposite directions into one scalar constraint because opposite response directions are not delivered simultaneously in one activation direction. This remains a simplified availability model.

DA mutual exclusivity and the BM discharge mode are:

$$
x_{t,b}^{c}\le P_bz_{t,b},
$$

$$
x_{t,b}^{d}\le P_b(1-z_{t,b}),
$$

$$
v_{t,b}^{off}\le P_b(1-z_{t,b}).
$$

## 10. Dynamic Containment constraints

The maximum reservation per direction is:

$$
y_{t,b}^{L}\le f^{DC}P_b,
\qquad
y_{t,b}^{H}\le f^{DC}P_b.
$$

The sustained response duration is $\tau^{DC}=0.25$ hours. Energy headroom is:

$$
e_{t,b,s}\ge e^{min}
 +\frac{y_{t,b}^{L}\tau^{DC}}{\eta_b C_b},
$$

$$
e_{t,b,s}\le e^{max}
 -\frac{y_{t,b}^{H}\tau^{DC}\eta_b}{C_b}.
$$

DC reservations are constant within every local-time EFA block. EFA block starts are 23:00, 03:00, 07:00, 11:00, 15:00 and 19:00 in Europe/London time.

A midnight UTC 24-hour horizon usually contains partial EFA blocks at its edges. The publication configuration fixes DC Low and DC High to zero in those incomplete blocks. This is conservative. A future extension should use forecasts generated at one common information origin and extend the optimisation far enough to cover complete final blocks.

## 11. Scenario generation

The Gaussian copula method estimates rank dependence from OOF forecast residuals available before the target cutoff. This is preferable to fitting dependence to raw price levels because scenarios are deviations around conditional forecasts.

Spearman correlation $\rho^S$ is converted to a Gaussian latent correlation:

$$
\rho^G=2\sin\left(\frac{\pi\rho^S}{6}\right).
$$

A nearest positive-definite correlation projection is applied before Cholesky factorisation.

For every forecast row the latent normal variable is transformed by a two-piece normal map. The lower and upper scales are chosen so that latent normal percentiles 0.10, 0.50 and 0.90 map exactly to forecast q10, q50 and q90. The far tails are clipped only at a configurable probability of 0.005 rather than truncating every scenario to the central 80 percent interval.

DC latent draws and DC quantiles are held constant within EFA blocks.

## 12. Settlement

Only the 48 modelled periods are settled. DA revenue is:

$$
\Pi^{DA}=\sum_t\tilde p_t^{DA}(x_t^d-x_t^c)\Delta.
$$

The BM proxy is:

$$
\Pi^{BM,proxy}=\sum_t\tilde p_t^{SSP}v_t^{off}\Delta.
$$

DC availability revenue is:

$$
\Pi^{DCL}=\sum_t\tilde p_t^{DCL}y_t^L\Delta,
\qquad
\Pi^{DCH}=\sum_t\tilde p_t^{DCH}y_t^H\Delta.
$$

Reported degradation is:

$$
D=\sum_t\kappa(x_t^c+x_t^d+v_t^{off})\Delta.
$$

Net proxy revenue is:

$$
\Pi=\Pi^{DA}+\Pi^{BM,proxy}+\Pi^{DCL}+\Pi^{DCH}-D.
$$

## 13. Backtest rules

Each eligible UTC date is evaluated on an exact 48-period grid. Missing actuals or forecasts cause that date to be rejected. Publication runs should use strict mode or report every skipped and failed date.

Random seeds are derived from the base seed and calendar date. This means a date receives the same scenario sample even when the requested backtest range changes.

All strategies are compared on the intersection of clean solved dates. Stochastic strategies are repeated over several base seeds. Both seed-level files and averaged files are retained.

## 14. Limitations that must remain in a paper

The BM stream is an SSP proxy rather than BOA settlement. DC activation is not simulated. Weather uses lagged realised data rather than archived forecast vintages. Network constraints, losses, availability failures, transaction costs, imbalance positions and market impact are omitted. The battery is a price taker. Partial edge EFA blocks are conservatively disabled.
