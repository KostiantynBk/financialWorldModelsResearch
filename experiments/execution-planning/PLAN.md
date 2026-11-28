# Research plan: execution planning with a market model

**Last updated:** 28 November 2026. **Status:** Stages 1, 2 and 3 and the real-data calibration complete. Gate C decided: Stage 4 tests volatility forecasting on real data. It is prespecified below and has not been run. This plan is updated as results arrive. Results are in the [experiment README](README.md) and [results/](results/).

## Question

The repository asks when a world model improves a financial decision compared with a simpler baseline. This experiment applies that question to one well-defined decision: **buying a fixed quantity before a deadline at the lowest cost.**

Hudson River Trading's write-up on [signal-driven optimal execution](https://www.hudsonrivertrading.com/hrtbeat/intern-spotlight-hrt-ai-labs-project-2026/) (Erginbas, 2026) supplies the test problem and one baseline, its hand-crafted urgency rule. This is not a replication of HRT's study, which compared model-free reinforcement-learning policies.

## Terminology

- **World model:** in this repository, a predictive model *learned from data*. It builds its own internal state from observations and forecasts how the environment evolves (in the sense of [Ha & Schmidhuber, 2018](https://arxiv.org/abs/1803.10122) and [Hafner et al., 2023](https://arxiv.org/abs/2301.04104)).
- **Hand-specified model:** equations we write ourselves, with known or fitted parameters. Stage 1 uses one as a reference ceiling, and later stages use one as a classical baseline. It is not a world model in the sense above.
- **Simulator:** the rules that generate synthetic market data and charge costs. It is separate from any model the agent uses, even when, as in Stage 1, the two coincide.

## Setup

| | In this experiment |
|---|---|
| **Observed** at each step | Mid-price and a noisy short-term signal (for example, order-flow imbalance), up to the current step |
| **Predicted** | The expected price path from now to the deadline |
| **Action** | How much to buy this step: between zero and the remaining quantity. The last step completes the order. |
| **Decision improved** | The execution schedule. Cost is implementation shortfall against the arrival mid-price. |
| **Agent state vs market state** | Remaining quantity and time are known exactly through bookkeeping. The market state must be estimated. |
| **Market impact** | Not modeled in Stages 1–4. Our trades pay the spread and a temporary impact cost but do not move future prices. |

Because the market is exogenous in Stages 1–4, the only part that can be learned is the **market forecast**. Our actions affect only our own bookkeeping and costs. Learning how actions change the market requires modeled impact (Stage 5).

**Planning loop:** build the state → the Predictor forecasts the price path → the Planner computes the cheapest schedule that completes the order without selling → execute the first step → repeat. With risk-neutral costs and linear impact, the cheapest schedule is a closed-form projection, so no search is needed. Risk limits, nonlinear costs, or impact remove that structure; the Planner then searches over candidate plans, for example with the cross-entropy method.

## Stages

| Stage | Agent's model | Market | Question | Status |
|---|---|---|---|---|
| **1** | Hand-specified, with the simulator's true parameters (ceiling) | Linear-Gaussian simulator | Are the simulator and planner correct? Is there headroom over simple rules? | **Done** (25 Nov 2026) |
| **2** | **Learned world model:** a small recurrent neural predictor; plus a fitted hand-specified model as the classical baseline | Same simulator | Can a learned model match the classical model when the classical assumptions hold? How much data does it need? | **Done** (27 Nov 2026) |
| **3** | Same models, plus a fitted regime-aware classical model and a larger learned variant | Misspecified simulator: regime switches, fat-tailed shocks, drifting signal strength | Does learning beat hand-specification when its assumptions are wrong? | **Done** (28 Nov 2026) |
| **4** | Learned volatility model vs classical volatility models, with a risk-aware planner | Real crypto market data (Binance BTCUSDT, ETHUSDT), with chronological splits | Do volatility forecasts, and learned ones in particular, make risk-aware execution cheaper on real prices? | Prespecified (28 Nov 2026) |
| **5** | Action-conditioned learned model; the planner searches candidate plans | Simulator with market impact | Can a model learn how our trades move the market, and does planning with it help? | Planned |
| Secondary | Model-free PPO (HRT-style) | As in Stages 2–3 | Is it better to plan with a model or to learn a policy directly? | Planned |

**Calibration with real data** (done, 28 November 2026; see the [experiment README](README.md#calibration-on-real-data-how-predictive-are-public-signals)). On public crypto data, measure the volatility and spread, and how strongly and for how long candidate signals predict the next minutes' price moves. Candidates include order-flow imbalance and recent returns. These numbers place reality within the Stage 1 headroom table. Order-flow imbalance is known to track same-interval price changes ([Cont, Kukanov & Stoikov, 2014](https://arxiv.org/abs/1011.6402)); its forward-looking power must be measured.

## Hypotheses

- **H1 (primary):** A planner using a learned world model beats the best simple rule on mean paired implementation shortfall on held-out test episodes. The best simple rule is TWAP or the tuned urgency rule, selected on training data. Prespecified: base-case parameters, training sets of 50, 200 and 1,000 episodes, test seed 303. *Falsified if* the 95% interval includes zero or favors the rule. *Supported in Stage 2 at all three training sizes (synthetic base case).*
- **H2 (headroom):** The gain grows with signal strength and persistence, and is zero without a signal. *Supported in Stage 1 with the true model.*
- **H3 (model error → decision loss):** The gap between true-model and learned-model planning shrinks as forecast error falls. *Supported in Stage 2, with a plateau from 200 training episodes.*
- **H4 (learning vs hand-specification):** The learned model matches the fitted hand-specified model when the simulator matches that model's assumptions (Stage 2), and beats it when it does not (Stage 3). *Stage 2 half not supported in the strict sense: the fitted model was cheaper at every training size, by 0.06 bps from 200 episodes. Stage 3 half supported in two of three scenarios (H4b below), by small margins.*
- **H5 (memory):** Using the signal's history beats using only the latest reading. *Supported in Stage 1 with the true model.*
- **H6 (secondary):** Planning with a learned model matches or beats model-free PPO with memory, at equal training data and completion.
- **Planned ablation:** the planner versus the same forecast fed to a tuned simple rule. This separates the value of planning from the value of forecasting.

## Protocol

- **Splits:** training seed 101 (Stage 2 uses seeds 101–105, one per run), validation seed 202, test seed 303. The test seed was first used for H1 in Stage 2.
- **Comparability:** every policy runs on the same paths, with the same completion rule and the same cost model.
- **Metrics, reported separately:**
  - forecast quality: negative log-likelihood or CRPS at several horizons, plus calibration of predictive intervals
  - decision quality: mean implementation shortfall in bps of arrival notional, CVaR95, completion diagnostics
  - Stage 2 deviation: the learned model gives point forecasts, so forecast quality is reported as R² at 1, 5 and 15 minutes and to the deadline instead of likelihood scores. Every policy completes the order by construction.
- **Statistics:** paired 95% intervals. They use a normal approximation for independent simulated episodes and a block bootstrap by day for real data. Each learned model is trained with at least five seeds.
- **Probing learned models:** any readout from a learned model's internal state is compared with the same readout from untrained networks of the same architecture. Decision value is judged by the forecasts the planner uses, not by what the internal state contains. See the [Stage 2 supplement](results/stage2_probe.md).
- **Reporting:** training budgets (data size, steps, wall-clock time, parameter count) are reported. Negative results are reported.

## Stage 3 prespecification

Registered on 28 November 2026, before any Stage 3 code was run. Changes after this point will be listed as deviations.

**Question (H4, second half):** when the simulated market breaks the classical model's assumptions, does the learned model + planner beat the fitted classical model + planner?

**Design principle.** A violated assumption matters only if it changes the forecast the planner uses. So each scenario must first show misspecification headroom (Gate C1) before any learned model is trained. Every scenario keeps the average signal strength of the base case (the trend explains about 1% of one-minute price variance) and its cost model. Regimes are drawn once per hour and stay fixed within it, so the best possible forecaster can be computed. The base case's price noise is σ = 5 bps per minute; *a* denotes the base trend's standard deviation, which gives 1%.

| | Scenario | Specification | Severe setting (used only if Gate C1 fails) |
|---|---|---|---|
| **A** | Signal on/off | Each hour is active with probability ½. Active: trend explaining 2%, half-life 10 minutes, signal noise equal to the active trend's standard deviation. Quiet: no trend, and the signal is pure noise with the same total variance as an active signal. | Active with probability 0.3, trend explaining 3.3% |
| **B** | Fast or slow trends | Each hour has a half-life of 3 or 30 minutes, with probability ½ each. Trend size *a* and signal noise *a* in both. | Half-lives of 2 and 60 minutes |
| **C** | Signal outliers | Base trend. Each signal reading's noise is normal with standard deviation *s* with probability 0.9, or 3*s* with probability 0.1, where *s* = *a*/√1.8, so the noise variance is unchanged. | 5% of readings at 5*s*, with *s* = *a*/√2.2 |
| D | Price outliers (negative control) | Base trend and signal. Price noise as in C, scaled to σ. Expected to fail Gate C1. | None |

**Strategies** (same planner and cost model as Stages 1–2):
- TWAP, and the urgency rule with β tuned on training hours; the best simple rule is selected on training hours.
- **True model + planner** (ceiling): the posterior-mean forecast under the scenario's true structure and parameters. In A and B it is exact: two Kalman filters weighted by the posterior regime probability. In C and D it is a Gaussian-sum filter with moment matching, checked against a particle filter on validation hours.
- **Fitted classical model + planner:** Stage 2's linear-Gaussian model fitted by maximum likelihood. It is misspecified here.
- **Fitted regime-aware model + planner** (secondary; A and B only): the correct two-regime structure with all parameters fitted by maximum likelihood.
- **Learned model + planner** (primary): Stage 2's architecture and training recipe, unchanged (1,666 parameters).
- **Larger learned model + planner** (secondary), selected on validation hours before the test run.
  - Candidates: GRU with 32 units, latent 16, width 64 (6,146 parameters); or GRU with 64 units, latent 16, width 64 (16,258 parameters). Training recipe unchanged.
  - Selection rule: train each candidate on run 0's 1,000 training hours in every main scenario that passes Gate C1. Choose the lower mean validation cost averaged over those scenarios; on a tie, the smaller candidate.

**Hypotheses:**
- **H4b (primary).** In each main scenario (A, B, C) that passes Gate C1, the learned model + planner has a lower mean test cost than the fitted classical model + planner, at 1,000 training hours.
  - *Supported* if the paired 95% interval of (learned − fitted) lies entirely below zero.
  - We also report whether it survives a Bonferroni correction over the three scenarios (98.3% intervals).
  - *Result: supported in A (−0.07 bps [−0.10, −0.05]) and C (−0.04 [−0.07, −0.02]), both surviving the correction; not supported in B (+0.02 [−0.01, +0.05]).*
- **Secondary:**
  - the learned model + planner beats the best simple rule at 1,000 hours;
  - its cost above the ceiling shrinks with training data and as its forecast approaches the ceiling's forecast;
  - comparisons with the fitted regime-aware model (A and B) and with the larger learned variant.
- **Negative control:** in D, we expect Gate C1 to fail and no learned advantage.

**Gate C1 (misspecification headroom).** Run on validation seed 212 before any learned model is trained:
1. Fit the classical model on run 0's 1,000 training hours.
2. Measure gap = fitted classical + planner cost − true model + planner cost, on 20,000 validation hours.
3. **Pass** if the gap is at least 0.15 bps and its 95% interval excludes zero. (Even in Stage 2's ideal setting, the learned model stayed 0.06 bps above the ceiling.)
4. If a main scenario fails, it is rerun once at its severe setting. If it still fails, it is reported as a case where the violation does not matter for decisions, and no learned models are trained for it.
5. D is reported regardless, and the primary learned model is trained for it at 1,000 hours only, to check for spurious wins.

**Data and seeds:**
- Training sets of 200, 1,000 and 5,000 hours, nested within a 5,000-hour pool per run; 5 runs.
- Run *r* uses training seed 111 + *r*, and *r* also seeds the network.
- Validation seed 212 is used for Gate C1, model selection and all code checks. Test seed 313 (20,000 hours) is used only by the final run.
- Base shocks are shared across scenarios (common random numbers); regime and outlier draws use a separate random stream.

**Deviations and implementation notes (added after the run):** B and C ran at their severe settings, as the Gate C1 rule required. Forecast metrics use the first 5,000 of the 20,000 test hours. The particle-filter check used 1,000 validation hours and 2,000 particles. An aborted launch ran for a few seconds before the reported run and produced no results. No hypothesis, threshold, seed or model setting was changed.

**Metrics:**
- mean implementation shortfall with paired 95% intervals, and run-to-run ranges;
- CVaR95;
- forecast R² at 1, 5 and 15 minutes ahead and to the deadline;
- the 30-minute forecast's correlation, size and relative error against the true model's forecast;
- costs by regime (A: active vs quiet hours; B: fast vs slow hours);
- training budgets.

## Stage 4 prespecification

Registered on 28 November 2026, before any Stage 4 code was run. Changes after this point will be listed as deviations.

**Why this stage.** The calibration found no out-of-sample predictability of price direction in the public one-minute signals tested, so a planner has no drift to exploit. Volatility is different. In an exploratory check on November–December 2025, inside the validation window below, the past predicted next-hour realized variance with in-sample R² of 0.61–0.70, and per-minute volatility differed about 4× between calm and busy hours. A risk-averse execution schedule depends on volatility forecasts rather than drift forecasts ([Almgren & Chriss, 2001](https://doi.org/10.21314/JOR.2001.041)).

**Question:** on real prices, does a planner that uses a learned volatility model achieve a lower risk-adjusted execution cost than one that uses the best classical volatility model, or no volatility forecast?

| | In Stage 4 |
|---|---|
| **Observed** at each minute | Past one-minute returns, volume, trade imbalance and time of day |
| **Predicted** | The variance of each remaining minute's price change until the deadline |
| **Action** | How much to buy this minute: between zero and the remaining quantity |
| **Decision improved** | The schedule of a one-hour buy order, judged by expected cost plus a risk penalty |
| **Market impact** | Not modeled in the price path: real prices are replayed unchanged, and our trades pay an assumed temporary impact κv², with κ = 30 bps |

**Decision problem:**
- *Episodes:* every UTC clock hour with complete data in the hour and the 24 hours before it.
- *Prices and trades:* decision times are t = 0, …, 59. Sₜ is the log close of the minute before decision t, in bps relative to S₀ (the last close before the hour). One unit must be bought by t = 59; buying vₜ costs vₜ(Sₜ + κvₜ).
- *Exposure:* let qⱼ be the quantity still to buy after the trades at times 0, …, j−1. Implementation shortfall is IS = Σₜ κvₜ² + Σⱼ qⱼΔSⱼ for j = 1, …, 59, where ΔSⱼ = Sⱼ − Sⱼ₋₁.
- *Primary metric:* the hour's risk-adjusted cost C = Σₜ κvₜ² + λ Σⱼ qⱼ²ΔSⱼ². With no predictable drift, its expectation is the mean-variance objective. Using realized squared moves avoids the large noise of the unpredictable price-move term. Mean IS, and mean IS + λ × variance of IS across hours, are also reported.
- *Planner:* at each decision time, with R still to buy and forecast variances σ̂ⱼ² for the remaining minutes, choose the remaining quantities to minimize κΣv² + λΣσ̂ⱼ²qⱼ², with qₜ = R and q₆₀ = 0. The minimizer solves a tridiagonal linear system and never sells. The first trade is executed and the plan is recomputed the next minute.
- *Risk aversion:* λ is set per market from the training period. At the training median of hourly per-minute volatility, the constant-volatility planner completes half the order in 15 minutes (primary), or in 5 or 30 minutes (secondary).

**Strategies** (same planner and cost model):
- TWAP.
- **Constant-volatility model + planner:** the training median volatility, as in classic Almgren–Chriss.
- **Seasonal EWMA model + planner (classical):**
  - a time-of-day variance profile: the training mean of squared returns per minute of the day, smoothed by a centered 31-minute circular moving average;
  - multiplied by an exponentially weighted average of deseasonalized squared returns, with its half-life chosen on training data from {15, 30, 60, 120, 240} minutes by QLIKE;
  - the forecast level is flat over the horizon.
- **Seasonal HAR model + planner (classical**, [Corsi, 2009](https://doi.org/10.1093/jjfinec/nbp001)**):**
  - a least-squares regression of the log of the next 60 minutes' mean deseasonalized squared return on the logs of its means over the last 60, 360 and 1,440 minutes;
  - fitted on training decision times every 15 minutes, with a log-normal bias correction;
  - forecast = profile × level, flat over the horizon.
- **Learned model + planner (primary learned model):**
  - a GRU encoder (32 units) reads the last 240 minutes and the current hour so far;
  - inputs: scaled return and log squared return, standardized log volume, trade imbalance, and time of day (sine and cosine at one and two cycles per day);
  - a 16-dimensional latent state is rolled forward by a learned residual transition that also receives each future minute's time of day, and decoded into the log variance of each future minute;
  - trained by the Gaussian negative log-likelihood of the returns, a proper score for variance forecasts, with Adam, on January 2023 to December 2024, with early stopping on January to June 2025;
  - 5 seeds per market; no hyperparameter search.
- **Hour-volatility oracle + planner (not achievable):** the hour's own realized variance per minute, constant within the hour. A reference for perfect hour-level volatility knowledge.
- **Best classical model:** whichever of the constant, EWMA and HAR models has the lowest validation risk-adjusted cost at the primary urgency, chosen per market.

**Hypotheses:**
- **V2 (primary):** the learned model + planner has a lower mean risk-adjusted cost than the best classical model + planner on test hours, at the primary urgency. Each market is tested separately. *Supported* if the 95% interval of the paired difference lies below zero; whether it survives a Bonferroni correction over the two markets (97.5% intervals) is also reported.
- **V1 (forecast quality):** the learned model's test QLIKE is lower than the best classical model's. QLIKE ([Patton, 2011](https://doi.org/10.1016/j.jeconom.2010.03.034)) is computed for the forecast total variance of the remaining exposure minutes, at decision times 0, 15, 30 and 45, and averaged.
- **V3:** the best classical model + planner has a lower mean risk-adjusted cost than the constant-volatility model + planner, i.e. volatility forecasts have decision value at all.
- **V4 (secondary; follows the candidate test in our Lambda research note):** does the learned model's *surprise* add information beyond classical features? Three variants of the seasonal HAR are compared by test QLIKE for the next hour's variance at decision time 0:
  - (a) the seasonal HAR alone;
  - (b) (a) plus classical extras: the HAR's own last-hour surprise (log of realized over forecast deseasonalized variance) and the mean absolute trade imbalance over the last 15 minutes;
  - (c) (b) plus the learned surprise: the log of the last 15 minutes' squared returns over the seed-averaged learned model's one-step forecasts.

  *Supported* if (c) has lower QLIKE than (b), with the 95% interval below zero.
- **Exploratory:** secondary urgencies; costs by time of day and in calm vs busy hours; mean IS and the across-hour mean-variance objective.

**Gate V (decision headroom).** Run on validation hours before any learned model is trained. For each market, the hour-volatility oracle + planner must have a mean risk-adjusted cost at least 3% below the constant-volatility planner's, with the 95% interval excluding zero. Markets that fail are reported as such. If neither passes, no learned models are trained.

**Data and protocol:**
- Binance spot BTCUSDT and ETHUSDT one-minute bars from the public archive, with checksums verified. The raw data is git-ignored.
- Training January 2023 to June 2025; validation July to December 2025 (Gate V, model choice, code checks); test January to October 2026, used once by the final run.
- Statistics: paired comparisons on the same hours; 95% intervals from a block bootstrap over days (2,000 resamples).
- *Disclosure:* the calibration used November 2025 to October 2026 for price-direction tests and reported volatility for those periods. The feasibility check used November–December 2025. The test period has not been used for any volatility forecast or schedule comparison.

## Decision gates

- **Gate A — simulator sanity:** passed (Stage 1).
- **Gate B — headroom:** passed in all 21 settings tested (Stage 1).
- **Gate C1 — misspecification headroom:** prespecified for Stage 3 (above).
- **Gate C — learned vs classical vs simple rules:** after Stages 2–3, decide whether real data and the PPO comparison are justified. Stages 2 and 3 and the calibration are done. **Decision (28 November 2026):** switch the real-data stage from price-direction forecasts, which the calibration found unpredictable, to volatility forecasts (Stage 4). The PPO comparison is deferred. Evidence: learning beats a misspecified model by small margins (Stage 3), and the public one-minute signals tested show essentially no out-of-sample predictability (next-minute R² at most about 0.01%, against 0.11% in the weakest Stage 1 setting).

## Stage 1 summary

With the simulator's true rules, the planner cost −1.41 bps in the base case, against +1.46 bps for TWAP, the best simple rule. That is a headroom of +2.87 bps, 95% CI [+2.69, +3.05]. Headroom ranged from +0.28 to +10.2 bps depending on signal strength and persistence. Using signal history added +0.62 bps. A planner that could see the true trend would gain a further 0.64 bps.

The no-sell constraint binds often, so the closed-form optimum does not apply directly; the true-model planner is the reference, not a proven optimum. Full details are in the [experiment README](README.md).

## Stage 2 summary

On 20,000 test hours, a learned model with 1,666 parameters plus the planner beat TWAP, the best simple rule, at all three training sizes. It captured 80% of the true model's 2.91 bps headroom with 50 training episodes, and 98% with 200 or 1,000. The fitted classical model, which has exactly the simulator's form, reached the ceiling from 50 episodes. It was cheaper than the learned model at every size, by 0.06 bps from 200 episodes. Decision loss fell as the learned forecast improved, but stayed flat between 200 and 1,000 episodes. Full details are in the [experiment README](README.md).

## Stage 3 summary

Gate C1 on validation hours:
- A passed at its prespecified setting (gap +0.24 bps).
- B and C passed only at their severe settings (+0.15 and +0.16).
- The negative control D failed (+0.02), as predicted.

On 20,000 test hours with 1,000 training hours, the learned model + planner beat the misspecified fitted classical model in A (−0.07 bps [−0.10, −0.05]) and C (−0.04 [−0.07, −0.02]), both surviving a Bonferroni correction. In B it did not (+0.02 [−0.01, +0.05]). That recovers 27% and 38% of the classical model's shortfall from the ceiling in A and C. With 5,000 hours the learned model was ahead in all three scenarios; with 200 hours it was behind in all three.

A fitted classical model with the correct regime structure beat the learned model in A and B at every training size. The larger learned variant did not improve on the primary one. In D the learned model did not beat the classical model (+0.05 [+0.04, +0.06]).

Conclusion: learning can beat a hand-specified model that is wrong in decision-relevant ways, but by small margins (about 0.04–0.07 bps here, against 2–3 bps of planning value over simple rules), and less than a correctly specified model would. Full details are in the [experiment README](README.md).

## Scope and assumptions

- **Data and execution:** offline simulation and historical data only; no broker connection or live trading.
- **Problem:** single asset, buying only, fixed target and deadline.
- **Costs:** stylized. Impact parameters are assumed and varied, because public data cannot reveal the impact of our own trades.
- **Interpretation:** results are not evidence of live trading performance.

## References

- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5–39. [doi:10.21314/JOR.2001.041](https://doi.org/10.21314/JOR.2001.041)
- Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47–88. [arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196. [doi:10.1093/jjfinec/nbp001](https://doi.org/10.1093/jjfinec/nbp001)
- Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008). Efficient projections onto the ℓ1-ball for learning in high dimensions. *ICML*. [PDF](https://icml.cc/imls/conferences/2008/papers/361.pdf)
- Erginbas, Y. E. (2026). [Intern Spotlight: HRT AI Labs Project](https://www.hudsonrivertrading.com/hrtbeat/intern-spotlight-hrt-ai-labs-project-2026/). Hudson River Trading.
- Ha, D., & Schmidhuber, J. (2018). World models. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
- Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering diverse domains through world models (DreamerV3). [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45. [doi:10.1115/1.3662552](https://doi.org/10.1115/1.3662552)
- Lehalle, C.-A., & Neuman, E. (2019). Incorporating signals into optimal trading. *Finance and Stochastics*, 23, 275–311. [doi:10.1007/s00780-019-00382-7](https://doi.org/10.1007/s00780-019-00382-7)
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256. [doi:10.1016/j.jeconom.2010.03.034](https://doi.org/10.1016/j.jeconom.2010.03.034)
