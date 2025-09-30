# Consumption–Investment Stochastic Control 

1. Problem Overview

We study a continuous-time, single-agent consumption–investment problem. The agent allocates wealth between:
	•	a risk-free asset with constant rate r, and
	•	a risky asset following geometric Brownian motion with drift mu and volatility sigma.

At each time t, the agent chooses:
	•	consumption rate c_t >= 0 (currency per unit time), and
	•	risky weight pi_t in [0,1] (fraction of wealth invested in the risky asset; can relax later).

The goal is to maximize expected discounted utility of consumption over a planning horizon.

2. Model & Dynamics

Let W_t be wealth, B_t a standard Brownian motion. Wealth evolves as:
dW_t = [ r*W_t + pi_t*(mu - r)*W_t - c_t ] dt
       + pi_t*sigma*W_t dB_t
with initial wealth W_0 = w0 > 0.

Controls & Constraints
	•	Controls: pi_t in  Pi ⊆ R, c_t in C ⊆ R_{>=0}.
	•	Typical choices:
	•	Boxed risky weight: Pi = [0,1] (no leverage/shorting initially).
	•	Consumption cap: 0 <= c_t <= cbar * W_t (e.g., cbar in [0,1]).
	•	No-bankruptcy (optional): keep W_t >= 0. If W_t -> 0, set c_t=0, pi_t=0 (absorbing boundary).

3. Objective

Choose admissible (pi_t, c_t) to maximize:
	•	Finite horizon:
Max E[ ∫_0^T exp(-rho*t) * u(c_t) dt + exp(-rho*T) * g(W_T) ]
	•	Infinite horizon (set T = ∞, typically g ≡ 0):
Max E[ ∫_0^∞ exp(-rho*t) * u(c_t) dt ]

Utility
	•	CRRA (Constant Relative Risk Aversion):
u(c) = (c^(1-gamma)) / (1-gamma)     if gamma != 1
u(c) = log(c)                        if gamma == 1
	•	Risk aversion gamma > 0, discount rho > 0. 
	•	Terminal utility: g(W) = kappa * (W^(1-gamma)) / (1-gamma) (optional).

4. Baseline Parameters
Risk-free rate r     = 0.02         # 2% annual
Risky drift mu       = 0.06         # 6% annual
Risky vol sigma      = 0.20         # 20% annual
Discount rho         = 0.03
Risk aversion gamma  = 2.0
Initial wealth w0    = 1.0
Planning horizon T   = 30.0 years   # or np.inf for infinite horizon
Time step dt         = 1/252        # daily
Consumption cap cbar = 0.08         # 8% of wealth per year
Risky bounds         = [0, 1]
Terminal weight kappa= 0.0

5. Discretization
W_{k+1} = W_k + [ r*W_k + pi_k*(mu - r)*W_k - c_k ] * dt
          + pi_k*sigma*W_k * sqrt(dt) * epsilon_k

epsilon_k ~ N(0,1)

Policy examples (baseline heuristics):
	•	Fixed risky weight: pi_k = const (e.g., 0.5)
	•	Proportional consumption: c_k = alpha * W_k with alpha <= cbar

These give you something to simulate before optimization.

6. Solution Approaches (roadmap)
	•	0) Baselines: simulate wealth/consumption under fixed (pi, alpha); compute mean utility
	•	1) Grid Search: search over constant (pi, alpha)
	•	2) Dynamic Programming (finite horizon):
		•	Discretize wealth grid {w_i}, time grid {t_k}
		•	Backward-induction Bellman equation:
V_k(w) = max_{pi,c} { u(c)*dt + exp(-rho*dt) * E[ V_{k+1}(W_{k+1}) | W_k=w ] }
	•	3) Infinite horizon approximation:
		•	Value iteration or policy iteration with contraction mapping
		•	Parametric policy (e.g., c = alpha*W, pi = beta) + policy gradient
	•	4) Refinements: transaction costs, borrowing/shorting, habit formation, mortality, constraints

7. Outputs & Diagnostics
	•	Plots: sample paths of W_t, c_t; distribution of W_T; probability of ruin; average utility
	•	Tables: policy comparison (mean/median W_T, CVaR, Sharpe of consumption stream, expected utility)
	•	Tests:
	•	If sigma = 0 and mu = r, stochastic term vanishes (deterministic check)
	•	If c_t = 0, wealth follows Merton portfolio growth — sanity check drift/vol parts

8. Milestone Checklist
	•	Document problem
	•	Implement utility + wealth stepper
	•	Implement baseline policies (pi=const, c=alpha*W)
	•	Monte Carlo simulate and plot W_t, c_t
	•	Evaluate baseline expected discounted utility
	•	Add DP or policy search (finite horizon first)
	•	Compare policies; add tests; tidy docs

