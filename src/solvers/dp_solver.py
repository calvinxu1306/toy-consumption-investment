import numpy as np
from scipy.interpolate import interp1d
from src.models.params import ModelParams
from src.models.utility import crra, terminal_crra

class DPSolverFiniteHorizon:
    def __init__(self, params: ModelParams, 
                 w_min=0.1, w_max=10.0, n_w_points=100, 
                 n_time_steps=None):
        """
        Sets up the grid for Finite Horizon Dynamic Programming.
        """
        self.params = params
        
        # 1. Wealth Grid (State Space)
        # We use a non-linear grid (squaring concentrates points near 0)
        self.w_grid = np.linspace(w_min**0.5, w_max**0.5, n_w_points)**2
        self.n_w = n_w_points
        
        # 2. Time Grid
        # If n_time_steps not provided, derive from T and dt
        T = 30.0  # Default horizon
        dt = 1/252
        if n_time_steps is None:
            self.n_t = int(round(T / dt))
        else:
            self.n_t = n_time_steps
            
        self.dt = T / self.n_t
        self.time_grid = np.linspace(0, T, self.n_t + 1)
        
        # 3. Value Function Container (Time x Wealth)
        # V[k, i] = Value at time step k and wealth grid point i
        self.V = np.zeros((self.n_t + 1, self.n_w))
        
        # 4. Optimal Controls Containers
        # opt_pi[k, i] = Optimal risky weight at time k, wealth i
        # opt_c[k, i]  = Optimal consumption RATE (alpha) at time k, wealth i
        self.opt_pi = np.zeros((self.n_t, self.n_w))
        self.opt_c = np.zeros((self.n_t, self.n_w))

    def initialize_terminal_condition(self):
        """
        Set V at time T (the last column) equal to the Bequest Utility g(W_T).
        """
        # V(T, W) = kappa * u(W)
        self.V[-1, :] = terminal_crra(
            self.w_grid, 
            self.params.gamma, 
            self.params.kappa
        )
        print("Initialized Terminal Condition (Time T).")

    def train_backward(self):
        """
        Iterate backwards from T-dt down to 0 to fill V, opt_pi, and opt_c.
        """
        print(f"Starting Backward Induction over {self.n_t} steps...")
        
        # 1. Pre-compute standard normal shocks for expectation (Quantization)
        # Using 7 simple quantile points to approximate the integral E[V]
        n_shocks = 7
        z_shocks = np.linspace(-2.5, 2.5, n_shocks) 
        # Gaussian weights
        weights = np.exp(-0.5 * z_shocks**2)
        weights /= weights.sum()

        # 2. Control Grids (Discretized Action Space)
        # We test these values at every state to find the max
        # Pi: 0% to 150% (allow some leverage)
        pi_options = np.linspace(0.0, 1.5, 31)       
        # Consumption (alpha): 0% to 40% (needs to be high near T)
        alpha_options = np.linspace(0.0, 0.40, 41)   

        # Create a meshgrid of controls for vectorized evaluation
        PI, ALPHA = np.meshgrid(pi_options, alpha_options)
        PI = PI.flatten()
        ALPHA = ALPHA.flatten()
        
        # Params extraction for speed
        r, mu, sigma = self.params.r, self.params.mu, self.params.sigma
        gamma, rho = self.params.gamma, self.params.rho
        dt = self.dt
        sqrt_dt = np.sqrt(dt)

        # Loop backwards: T-1, T-2, ..., 0
        for k in range(self.n_t - 1, -1, -1):
            
            # Create interpolator for V_{k+1} (Value at next time step)
            # We use 'fill_value="extrapolate"' to handle edge cases safely
            v_next_func = interp1d(self.w_grid, self.V[k+1], 
                                   kind='linear', fill_value="extrapolate")
            
            # Iterate over each Wealth State in our grid
            for i, w in enumerate(self.w_grid):
                
                # --- A. Calculate Next Wealth W_{t+1} for ALL controls & ALL shocks ---
                # Dimensions: [num_controls, num_shocks]
                c_rates = ALPHA * w
                
                # Drift: (r + pi*(mu-r))*w - c
                drift_part = (r * w + PI * (mu - r) * w - c_rates) * dt
                
                # Diffusion: pi * sigma * w * z * sqrt(dt)
                # We broadcast PI (controls) against z_shocks
                diff_part = (PI[:, None] * sigma * w) * (z_shocks[None, :] * sqrt_dt)
                
                w_next = w + drift_part[:, None] + diff_part
                
                # Absorb at zero
                w_next = np.maximum(w_next, 1e-6)
                
                # --- B. Evaluate Expected Value E[V_{k+1}] ---
                # Map w_next -> Utility values using interpolator
                v_next_values = v_next_func(w_next) # Shape: [num_controls, num_shocks]
                
                # Weighted average across shocks -> Expectation
                ev_next = np.dot(v_next_values, weights)
                
                # --- C. Total Bellman Objective ---
                # Obj = U(c)*dt + exp(-rho*dt) * E[V]
                current_u = crra(c_rates, gamma)
                objective = current_u * dt + np.exp(-rho * dt) * ev_next
                
                # --- D. Find Max ---
                best_idx = np.argmax(objective)
                
                self.V[k, i] = objective[best_idx]
                self.opt_pi[k, i] = PI[best_idx]
                self.opt_c[k, i] = ALPHA[best_idx] # Storing alpha, not raw c
            
            if k % 50 == 0:
                print(f"Solved time step {k}/{self.n_t}...", end="\r")

        print("\nBackward Induction Complete.")

    def get_policy_function(self):
        """
        Returns a callable function policy(w, t, params) that interpolates
        the trained tables to find optimal pi and c.
        """
        
        def trained_policy(wealth_arr, t_val, params):
            # 1. Find time index k
            # (Clip to valid range [0, n_t-1])
            k = int(round(t_val / self.dt))
            k = min(k, self.n_t - 1)
            
            # 2. Interpolate Policy Tables at time k
            # We treat wealth_arr as a vector
            # Use NumPy's interp which is fast and robust
            pi_k = np.interp(wealth_arr, self.w_grid, self.opt_pi[k])
            alpha_k = np.interp(wealth_arr, self.w_grid, self.opt_c[k])
            
            # 3. Convert alpha back to raw consumption c
            c_k = alpha_k * wealth_arr
            
            return pi_k, c_k
            
        return trained_policy