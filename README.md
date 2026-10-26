# Optimal Consumption & Investment (Finite Horizon)

A Python implementation of the Merton Consumption-Investment problem over a finite horizon. This project solves for the optimal dynamic strategy using **Dynamic Programming (Bellman Equation)**, compares it against a static Merton baseline, and checks it against the problem's **closed-form solution**.

## Project Overview
We model an agent who must choose:
1.  **Consumption Rate ($c_t$):** How much wealth to spend per instant.
2.  **Investment Allocation ($\pi_t$):** What fraction of wealth to put into a risky asset (Geometric Brownian Motion).

**The Goal:** Maximize expected Discounted Utility (CRRA) over a 30-year horizon with **no bequest motive** (terminal utility = 0 by default; set `--kappa` to add one).

### Key Result
The **Dynamic Programming (DP)** agent learns the "Die with Zero" strategy: it consumes faster as the horizon approaches and runs wealth down to zero by $T$. The static baseline applies the infinite-horizon Merton rule at every date, so it keeps accumulating wealth that is never consumed.

At the default parameters (5,000 paths, same random shocks for every policy):

| Policy | Expected utility | Certainty-equivalent consumption | vs baseline |
| :--- | ---: | ---: | ---: |
| Static Merton ($\pi^*=0.5$, $\alpha^*=3\%$) | −660.6 ± 2.7 | 0.0300 | — |
| DP Agent | −391.9 ± 1.6 | 0.0505 | **+68.6%** |
| Closed-form Optimum | −391.9 ± 1.6 | 0.0505 | +68.6% |

The DP agent matches the closed-form optimum to within 0.01%.

**Why certainty-equivalent consumption?** Raw CRRA utility has no natural scale (with $\gamma=2$, $u(c)=-1/c$), so "% more utility" is not meaningful. The certainty equivalent is the constant consumption rate that would give the agent the same expected utility, so percentage differences are in consumption units.

## Project Structure
```text
TOY-CONSUMPTION-INVESTMENT/
├── data/                   # Generated plots and CSVs (created on first run)
├── scripts/
│   ├── run_grid_search.py  # Stage 1: Searches constant (pi, alpha) policies; compares with Merton values
│   ├── run_dp_solver.py    # Stage 2: Trains the DP Solver, plots heatmaps and DP vs closed form
│   └── run_comparison.py   # Stage 3: Head-to-head simulation (Static vs DP vs Optimum)
├── src/
│   ├── models/             # Wealth dynamics, CRRA utility, policies, metrics, parameters
│   ├── solvers/            # Finite Horizon DP Solver (Backward Induction)
│   └── experiment.py       # Shared train -> simulate -> score pipeline
├── tests/                  # pytest suite (checks the DP against the closed form)
├── main.py                 # CLI Entry Point (Run Experiments Here)
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Dependencies
├── requirements-dev.txt    # Dependencies + pytest
└── README.md               # Documentation
```

## How to Run

### 1. Setup Environment
Requires Python 3.10+.
```bash
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate        # Windows
source .venv/bin/activate       # macOS / Linux

# Install requirements
pip install -r requirements.txt
```

### 2. Run and Experiment
Run the simulation with default settings ($\gamma=2, \sigma=0.2$):
```bash
python main.py
```

### 3. Custom Experiments
Test different scenarios by passing arguments:
High Risk Tolerance ($\gamma=1.0$) in a Volatile Market ($\sigma=0.3$):
```bash
python main.py --gamma 1.0 --sigma 0.3
```
Conservative Investor ($\gamma=5.0$) over 50 Years:

```bash
python main.py --gamma 5.0 --T 50.0
```
### Available Arguments
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--gamma` | `2.0` | Risk Aversion (Higher = More conservative) |
| `--sigma` | `0.20` | Volatility (Standard Deviation) |
| `--mu` | `0.06` | Expected Stock Return (Drift) |
| `--r` | `0.02` | Risk-Free Rate |
| `--rho` | `0.03` | Discount Rate (Time Preference) |
| `--T` | `30.0` | Investment Horizon (Years) |
| `--kappa` | `0.0` | Bequest weight on terminal wealth (0 = none) |
| `--paths` | `5000` | Number of simulation paths |
| `--seed` | `42` | Random seed (all policies see the same shocks) |
| `--output` | `"data"` | Directory to save generated plots |

All policies share the same constraint: risky weight in $[0, 1.5]$ (`ModelParams.pi_max`).

### 4. Run the Tests
```bash
pip install -r requirements-dev.txt
pytest
```

## Web Dashboard (Interactive UI)

For a more visual experience, the project includes a web-based dashboard built with **Streamlit**. Adjust the parameters (returns, volatility, risk aversion, discount rate, horizon, bequest weight) in the sidebar and click **Run Simulation**.

### How to Launch
```bash
streamlit run app.py
```

This will automatically open the dashboard in your default web browser at http://localhost:8501.

#### Features
Cached Solver: The DP agent is trained once per parameter set (a few seconds); re-running with a different number of paths reuses it.

Three-Way Comparison: Plots average wealth and consumption for the Static Merton baseline, the DP agent, and the closed-form optimum on the same graph.

Performance Metrics: Shows each policy's certainty-equivalent consumption and its gain over the baseline.

## Theory & Results
The problem is solved using the **Hamilton-Jacobi-Bellman (HJB)** equation logic, discretized via backward induction on a monthly grid:
$$V(t, W) = \max_{\pi, c} \left[ u(c)dt + e^{-\rho dt} \mathbb{E}[V(t+dt, W_{t+1})] \right]$$

The expectation uses Gauss-Hermite quadrature. Because CRRA utility is homothetic, $V(t,W) = f(t)\,W^{1-\gamma}/(1-\gamma)$; the solver interpolates $V$ in a transformed coordinate where this is linear in $\log W$, which keeps the policy free of grid-edge artifacts.

**Closed-form check.** With CRRA utility the finite-horizon problem has an exact solution, which the tests use as an oracle:
$$\pi^* = \frac{\mu - r}{\gamma\sigma^2}, \qquad \frac{c_t}{W_t} = \frac{\alpha^*}{1 - e^{-\alpha^*(T-t)}} \quad (\kappa = 0)$$
where $\alpha^* = \frac{\rho}{\gamma} + \frac{\gamma - 1}{\gamma}\left(r + \frac{(\mu-r)^2}{2\gamma\sigma^2}\right)$ is the infinite-horizon Merton consumption rate.

**Findings** (default parameters):
* **Early Game:** The agent invests $\pi^* = 50\%$ and consumes about 5.1% of wealth per year, already above the infinite-horizon rate $\alpha^* = 3\%$, because there is no reason to keep wealth past $T$.
* **End Game:** The consumption rate grows like $1/(T-t)$: about 22% with five years left and 100% with one year left, driving wealth to zero at $T=30$. In the final months the DP invests slightly less than $\pi^*$, which is the exact optimum of the discrete-time model (consumption leaves the account during each step).
