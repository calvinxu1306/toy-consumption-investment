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
Available Arguments:| Flag | Default | Description || :--- | :--- | :--- || --gamma | 2.0 | Risk Aversion (Higher = More conservative) || --sigma | 0.20 | Volatility (Standard Deviation) || --mu | 0.06 | Expected Return (Drift) || --T | 30.0 | Investment Horizon (Years) || --paths | 5000 | Number of simulation paths |

### Theory & Results
## Theory & Results
The problem is solved using the **Hamilton-Jacobi-Bellman (HJB)** equation logic, discretized via backward induction:
$$V(t, W) = \max_{\pi, c} \left[ u(c)dt + e^{-\rho dt} \mathbb{E}[V(t+dt, W_{t+1})] \right]$$

**Findings:**
* **Early Game ($t < 20$):** The agent mimics the infinite horizon solution (invest 50%, consume 4%).
* **End Game ($t > 25$):** The agent dramatically ramps up consumption (to >20% rate) to drive wealth to zero by $T=30$, capturing utility that the static baseline leaves on the table.