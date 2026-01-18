import numpy as np
from src.models.utility import crra, terminal_crra

def simulate(params, sim, policy_func):
    """
    Simulates wealth paths under a given policy.
    Uses params.T to determine simulation horizon (from CLI args).
    """
    # 1. Setup Time Grid
    # FIX: Use params.T (which comes from CLI) instead of sim.T
    n_steps = int(round(params.T / sim.dt))
    time_grid = np.linspace(0, params.T, n_steps + 1)
    dt = sim.dt
    sqrt_dt = np.sqrt(dt)
    
    # 2. Initialize Arrays
    # W: Wealth [n_paths, n_steps+1]
    # C: Consumption [n_paths, n_steps]
    # Pi: Risky Weight [n_paths, n_steps]
    W = np.zeros((sim.n_paths, n_steps + 1))
    C = np.zeros((sim.n_paths, n_steps))
    Pi = np.zeros((sim.n_paths, n_steps))
    
    W[:, 0] = 1.0  # Start with wealth = 1.0
    
    # 3. Random Shocks
    if sim.seed is not None:
        np.random.seed(sim.seed)
    
    # Generate Brownian increments (Standard Normal)
    Z = np.random.randn(sim.n_paths, n_steps)
    
    # 4. Time Loop
    for k in range(n_steps):
        t = time_grid[k]
        w_curr = W[:, k]
        
        # Get optimal controls from the policy function
        # The policy function handles vectorization (arrays of w)
        pi_k, c_k = policy_func(w_curr, t, params)
        
        # Store Controls
        Pi[:, k] = pi_k
        C[:, k] = c_k
        
        # Evolve Wealth: dW = (r + pi*(mu-r))*W*dt - C*dt + pi*sigma*W*dW
        drift = (params.r * w_curr + pi_k * (params.mu - params.r) * w_curr - c_k) * dt
        diffusion = (pi_k * params.sigma * w_curr) * (Z[:, k] * sqrt_dt)
        
        w_next = w_curr + drift + diffusion
        
        # Absorb at zero (cannot have negative wealth)
        if sim.absorb_at_zero:
            w_next = np.maximum(w_next, 1e-6)
            
        W[:, k+1] = w_next

    # 5. Compute Utility (The Score)
    # Discounted Utility from Consumption
    disc_factors = np.exp(-params.rho * time_grid[:-1])
    u_flow = crra(C, params.gamma) * dt
    # Sum over time (weighted by discount factor)
    total_u_c = np.sum(u_flow * disc_factors[None, :], axis=1)
    
    # Discounted Utility from Terminal Wealth (Bequest)
    term_u = terminal_crra(W[:, -1], params.gamma, params.kappa)
    term_disc = np.exp(-params.rho * params.T)
    total_u_term = term_u * term_disc
    
    # Total Discounted Utility
    total_utility = total_u_c + total_u_term

    return {
        "W": W,
        "C": C,
        "Pi": Pi,
        "times": time_grid,
        "DU": np.mean(total_utility)
    }