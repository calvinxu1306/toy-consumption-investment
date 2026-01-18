import numpy as np
from src.models.params import ModelParams, SimConfig
from src.models.policies import constant_policy
from src.models.utility import crra, terminal_crra

RUIN_EPS = 1e-6

def step_wealth(
        wealth_k: np.ndarray,
        pi_k: np.ndarray,
        c_k: np.ndarray,
        params: ModelParams,
        dt: float,
        z: np.ndarray,
        absorb_at_zero: bool = True,
) -> np.ndarray:
    """
    Euler–Maruyama step for wealth dynamics.
    """
    r, mu, sigma = params.r, params.mu, params.sigma
    
    # Calculate Drift and Diffusion
    drift = (r * wealth_k + pi_k * (mu - r) * wealth_k - c_k) * dt
    diff = (pi_k * sigma * wealth_k) * np.sqrt(dt) * z
    
    wealth_next = wealth_k + drift + diff
    
    if absorb_at_zero:
        wealth_next = np.maximum(wealth_next, 0.0)
        
    return wealth_next

def simulate(
        params: ModelParams = ModelParams(),
        sim: SimConfig = SimConfig(),
        policy=None,
        w0: float = 1.0,
):
    """
    Simulates wealth trajectories over time T.
    Returns dictionary with time grid, wealth paths, controls, and utility stats.
    """
    # Default policy if None provided
    if policy is None:
        policy = lambda w, t, p: constant_policy(w, t, p, pi_const=0.5, alpha_consume=0.04)

    n_steps = int(round(sim.T / sim.dt))
    time_grid = np.linspace(0.0, sim.T, n_steps + 1)

    rng = np.random.default_rng(sim.seed)
    z = rng.standard_normal((n_steps, sim.n_paths))

    # Pre-allocate arrays
    wealth = np.zeros((n_steps + 1, sim.n_paths))
    wealth[0] = w0
    cons = np.zeros((n_steps, sim.n_paths))
    risky = np.zeros((n_steps, sim.n_paths))

    disc = np.exp(-params.rho * time_grid[:-1])
    du = np.zeros(sim.n_paths)
    alive = np.ones(sim.n_paths, dtype=bool)

    # --- Time Loop ---
    for k in range(n_steps):
        # Get controls
        pi_k, c_k = policy(wealth[k], time_grid[k], params)
        risky[k] = pi_k
        cons[k] = c_k

        # Accumulate Running Utility (only if alive)
        if np.any(alive):
            du[alive] += disc[k] * crra(c_k[alive], params.gamma) * sim.dt

        # Step Wealth
        # We perform the step for everyone, but dead paths stay at 0 due to absorption
        next_w_values = step_wealth(
            wealth[k], pi_k, c_k, params, sim.dt, z[k], 
            absorb_at_zero=sim.absorb_at_zero
        )
        
        # Explicitly enforce death state if previously dead (optional safeguard)
        mask_dead = ~alive
        next_w_values[mask_dead] = 0.0
        
        wealth[k + 1] = next_w_values

        # Update alive status
        alive = wealth[k + 1] > RUIN_EPS

    # --- Terminal Utility ---
    # Add terminal utility only for paths that survived to T
    survived_T = wealth[-1] > RUIN_EPS
    if np.any(survived_T):
        term_reward = terminal_crra(wealth[-1][survived_T], params.gamma, params.kappa)
        du[survived_T] += np.exp(-params.rho * sim.T) * term_reward

    return {
        "times": time_grid,
        "W": wealth,
        "C": cons,
        "PI": risky,
        "DU_vec": du,
        "DU": float(np.mean(du)),
    }