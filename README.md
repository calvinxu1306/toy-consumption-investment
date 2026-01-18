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