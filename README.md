# Optimal Consumption & Investment (Finite Horizon)

A Python implementation of the Merton Consumption-Investment problem over a finite horizon. This project solves for the optimal dynamic strategy using **Dynamic Programming (Bellman Equation)** and compares it against a static Merton baseline.

## Project Overview
We model an agent who must choose:
1.  **Consumption Rate ($c_t$):** How much wealth to spend per instant.
2.  **Investment Allocation ($\pi_t$):** What fraction of wealth to put into a risky asset (Geometric Brownian Motion).

**The Goal:** Maximize expected Discounted Utility (CRRA) over a 30-year horizon with **no bequest motive** (terminal utility = 0).

### Key Result
The **Dynamic Programming (DP)** agent significantly outperforms the static baseline by learning the "Die with Zero" strategy. While the baseline preserves wealth indefinitely, the DP agent optimally consumes remaining capital as the horizon approaches, maximizing total lifetime utility.

## Project Structure
```text
TOY-CONSUMPTION-INVESTMENT/
├── data/                   # Generated plots (Heatmaps, Trajectories)
├── scripts/
│   ├── run_grid_search.py  # Stage 1: Verifies physics vs. Theoretical Merton
│   ├── run_dp_solver.py    # Stage 2: Trains the DP Solver & plots heatmaps
│   └── run_comparison.py   # Stage 3: Head-to-head simulation (Smart vs Static)
├── src/
│   ├── models/             # Wealth dynamics, CRRA utility, Parameters
│   └── solvers/            # Finite Horizon DP Solver (Backward Induction)
└── requirements.txt        # Dependencies
```

## How to Run

### 1. Setup Environment
```bash
# Create and activate virtual environment (Windows)
python -m venv .venv
.\.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Verify Baseline (Grid Search)
Checks if the simulator physics match Merton's theoretical infinite-horizon solution ($\pi^* \approx 0.5, \alpha^* \approx 0.04$).
```bash
python scripts/run_grid_search.py
```

### 3. Train the Solver
Runs backward induction to solve the Bellman equation and generates heatmaps of the optimal policy.
```bash
python scripts/run_dp_solver.py
# Output: data/dp_policy_pi.png and data/dp_policy_alpha.png
```

### 4. Run the Showdown
Simulates 10,000 lifetimes to compare the Smart Agent (DP) against the Baseline.
```bash
python scripts/run_comparison.py
# Output: Comparison plots of Wealth and Consumption trajectories.
```

### Model Parameters Table
## Model Parameters
| Parameter | Symbol | Value | Description |
| :--- | :---: | :---: | :--- |
| **Risk-Free Rate** | $r$ | 0.02 | 2% Annual risk-free return |
| **Drift** | $\mu$ | 0.06 | 6% Annual expected stock return |
| **Volatility** | $\sigma$ | 0.20 | 20% Annual standard deviation |
| **Risk Aversion** | $\gamma$ | 2.0 | CRRA curvature parameter |
| **Discount Rate** | $\rho$ | 0.03 | Time preference |
| **Horizon** | $T$ | 30.0 | Years until the game ends |
| **Terminal Weight** | $\kappa$ | 0.0 | No value assigned to leftover wealth |

### Theory & Results
## Theory & Results
The problem is solved using the **Hamilton-Jacobi-Bellman (HJB)** equation logic, discretized via backward induction:
$$V(t, W) = \max_{\pi, c} \left[ u(c)dt + e^{-\rho dt} \mathbb{E}[V(t+dt, W_{t+1})] \right]$$

**Findings:**
* **Early Game ($t < 20$):** The agent mimics the infinite horizon solution (invest 50%, consume 4%).
* **End Game ($t > 25$):** The agent dramatically ramps up consumption (to >20% rate) to drive wealth to zero by $T=30$, capturing utility that the static baseline leaves on the table.