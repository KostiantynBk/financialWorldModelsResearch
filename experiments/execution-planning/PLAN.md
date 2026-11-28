# Research plan: execution planning with a market model

**Last updated:** 27 November 2026. **Status:** Stages 1 and 2 complete; Stage 3 is next. This plan is updated as results arrive. Results are in the [experiment README](README.md) and [results/](results/).

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
| **3** | Same models | Misspecified simulator: regime switches, fat-tailed shocks, drifting signal strength | Does learning beat hand-specification when its assumptions are wrong? | Next |
| **4** | Same, trained on historical data | Real crypto market data, with chronological splits | Do the gains survive on a real market? | Planned |
| **5** | Action-conditioned learned model; the planner searches candidate plans | Simulator with market impact | Can a model learn how our trades move the market, and does planning with it help? | Planned |
| Secondary | Model-free PPO (HRT-style) | As in Stages 2–3 | Is it better to plan with a model or to learn a policy directly? | Planned |

**Calibration with real data**, which can run alongside Stage 3. On public crypto data, measure the volatility and spread, and how strongly and for how long candidate signals predict the next minutes' price moves. Candidates include order-flow imbalance and recent returns. These numbers place reality within the Stage 1 headroom table. Order-flow imbalance is known to track same-interval price changes ([Cont, Kukanov & Stoikov, 2014](https://arxiv.org/abs/1011.6402)); its forward-looking power must be measured.

## Hypotheses

- **H1 (primary):** A planner using a learned world model beats the best simple rule on mean paired implementation shortfall on held-out test episodes. The best simple rule is TWAP or the tuned urgency rule, selected on training data. Prespecified: base-case parameters, training sets of 50, 200 and 1,000 episodes, test seed 303. *Falsified if* the 95% interval includes zero or favors the rule. *Supported in Stage 2 at all three training sizes (synthetic base case).*
- **H2 (headroom):** The gain grows with signal strength and persistence, and is zero without a signal. *Supported in Stage 1 with the true model.*
- **H3 (model error → decision loss):** The gap between true-model and learned-model planning shrinks as forecast error falls. *Supported in Stage 2, with a plateau from 200 training episodes.*
- **H4 (learning vs hand-specification):** The learned model matches the fitted hand-specified model when the simulator matches that model's assumptions (Stage 2), and beats it when it does not (Stage 3). *Stage 2 half not supported in the strict sense: the fitted model was cheaper at every training size, by 0.06 bps from 200 episodes.*
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

## Decision gates

- **Gate A — simulator sanity:** passed (Stage 1).
- **Gate B — headroom:** passed in all 21 settings tested (Stage 1).
- **Gate C — learned vs classical vs simple rules:** after Stages 2–3, decide whether real data and the PPO comparison are justified. Stage 2 is done; the decision follows Stage 3.

## Stage 1 summary

With the simulator's true rules, the planner cost −1.41 bps in the base case, against +1.46 bps for TWAP, the best simple rule. That is a headroom of +2.87 bps, 95% CI [+2.69, +3.05]. Headroom ranged from +0.28 to +10.2 bps depending on signal strength and persistence. Using signal history added +0.62 bps. A planner that could see the true trend would gain a further 0.64 bps.

The no-sell constraint binds often, so the closed-form optimum does not apply directly; the true-model planner is the reference, not a proven optimum. Full details are in the [experiment README](README.md).

## Stage 2 summary

On 20,000 test hours, a learned model with 1,666 parameters plus the planner beat TWAP, the best simple rule, at all three training sizes. It captured 80% of the true model's 2.91 bps headroom with 50 training episodes, and 98% with 200 or 1,000. The fitted classical model, which has exactly the simulator's form, reached the ceiling from 50 episodes. It was cheaper than the learned model at every size, by 0.06 bps from 200 episodes. Decision loss fell as the learned forecast improved, but stayed flat between 200 and 1,000 episodes. Full details are in the [experiment README](README.md).

## Scope and assumptions

- **Data and execution:** offline simulation and historical data only; no broker connection or live trading.
- **Problem:** single asset, buying only, fixed target and deadline.
- **Costs:** stylized. Impact parameters are assumed and varied, because public data cannot reveal the impact of our own trades.
- **Interpretation:** results are not evidence of live trading performance.

## References

- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5–39. [doi:10.21314/JOR.2001.041](https://doi.org/10.21314/JOR.2001.041)
- Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47–88. [arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
- Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008). Efficient projections onto the ℓ1-ball for learning in high dimensions. *ICML*. [PDF](https://icml.cc/imls/conferences/2008/papers/361.pdf)
- Erginbas, Y. E. (2026). [Intern Spotlight: HRT AI Labs Project](https://www.hudsonrivertrading.com/hrtbeat/intern-spotlight-hrt-ai-labs-project-2026/). Hudson River Trading.
- Ha, D., & Schmidhuber, J. (2018). World models. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
- Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering diverse domains through world models (DreamerV3). [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45. [doi:10.1115/1.3662552](https://doi.org/10.1115/1.3662552)
- Lehalle, C.-A., & Neuman, E. (2019). Incorporating signals into optimal trading. *Finance and Stochastics*, 23, 275–311. [doi:10.1007/s00780-019-00382-7](https://doi.org/10.1007/s00780-019-00382-7)
