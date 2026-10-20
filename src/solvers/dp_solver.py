import logging

import numpy as np
from src.models.dynamics import WEALTH_FLOOR
from src.models.params import ModelParams
from src.models.utility import crra, terminal_crra

logger = logging.getLogger(__name__)


class DPSolverFiniteHorizon:
    def __init__(self, params: ModelParams,
                 w_min=0.01, w_max=15.0, n_w_points=40,
                 n_time_steps=None):
        """
        Sets up the grid for Finite Horizon Dynamic Programming.
        n_time_steps defaults to monthly steps (12 per year).
        """
        self.params = params

        # 1. Wealth Grid (State Space)
        # Log-spaced, because the value function is interpolated in log-wealth
        self.w_grid = np.geomspace(w_min, w_max, n_w_points)
        self.n_w = n_w_points

        # 2. Time Grid
        T = params.T
        if n_time_steps is None:
            self.n_t = max(1, int(round(T * 12)))
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

    def _encode(self, v):
        """
        Map values to a coordinate that is linear in log-wealth for CRRA utility:
            V = f(t) W^(1-gamma) / (1-gamma)  =>  log((1-gamma) V) = log f + (1-gamma) log W
            V = a(t) log W + b(t)  (log utility)
        Interpolating, and extrapolating past the grid edges, in this space is exact
        for the CRRA value function; linear interpolation of V itself is not, and its
        errors at the edges make risk look attractive (the agent levers up).
        """
        if self._log_utility:
            return v
        return np.log((1 - self.params.gamma) * v)

    def _expected_decoded(self, y, weights):
        """E[V] from encoded values y [..., num_shocks]. Overwrites y."""
        if self._log_utility:
            return y @ weights
        # V = exp(y) / (1-gamma); apply the constant after the (linear) expectation
        return (np.exp(y, out=y) @ weights) / (1 - self.params.gamma)

    def train_backward(self):
        """
        Iterate backwards from T-dt down to 0 to fill V, opt_pi, and opt_c.
        """
        self.initialize_terminal_condition()
        self._log_utility = abs(self.params.gamma - 1.0) < 1e-12
        logger.info("Starting Backward Induction over %d steps...", self.n_t)

        # 1. Gauss-Hermite quadrature for E[V] over z ~ N(0, 1)
        # (exact for polynomials in z up to degree 13)
        n_shocks = 7
        z_shocks, weights = np.polynomial.hermite_e.hermegauss(n_shocks)
        weights /= weights.sum()

        # 2. Control Grids (Discretized Action Space)
        # We test these values at every state to find the max
        # Pi: 0 to pi_max in steps of 0.05
        pi_options = np.linspace(0.0, self.params.pi_max,
                                 int(round(self.params.pi_max / 0.05)) + 1)
        # Consumption (alpha): 0, then log-spaced up to 1/dt (consume all wealth
        # within one step), which the optimal policy needs right before T
        alpha_options = np.concatenate([[0.0], np.geomspace(1e-3, 1.0 / self.dt, 80)])

        # Create a meshgrid of controls for vectorized evaluation
        PI, ALPHA = np.meshgrid(pi_options, alpha_options)
        PI = PI.flatten()
        ALPHA = ALPHA.flatten()

        # Params extraction for speed
        r, mu, sigma = self.params.r, self.params.mu, self.params.sigma
        gamma, rho = self.params.gamma, self.params.rho
        dt = self.dt
        sqrt_dt = np.sqrt(dt)

        # 3. Everything except V_{k+1} is the same at every time step, so compute
        # it once. Dimensions: [n_w, num_controls] and [n_w, num_controls, num_shocks]
        w = self.w_grid[:, None]
        c_rates = ALPHA * w
        flow_u = crra(c_rates, gamma) * dt

        # Drift: (r + pi*(mu-r))*w - c ; Diffusion: pi * sigma * w * z * sqrt(dt)
        drift_part = (r * w + PI * (mu - r) * w - c_rates) * dt
        diff_part = (PI * sigma * w)[..., None] * (z_shocks * sqrt_dt)
        w_next = np.maximum((w + drift_part)[..., None] + diff_part, WEALTH_FLOOR)

        # Linear-interpolation brackets in log-wealth. frac falls outside [0, 1]
        # when w_next is off the grid, which extrapolates along the edge segment.
        x_grid = np.log(self.w_grid)
        x_next = np.log(w_next)
        lo = np.clip(np.searchsorted(x_grid, x_next) - 1, 0, self.n_w - 2)
        frac = (x_next - x_grid[lo]) / (x_grid[lo + 1] - x_grid[lo])

        discount = np.exp(-rho * dt)
        rows = np.arange(self.n_w)
        y_next = np.empty_like(frac)

        # Loop backwards: T-1, T-2, ..., 0
        for k in range(self.n_t - 1, -1, -1):

            # --- A. Expected continuation value E[V_{k+1}(W_{t+1})] ---
            v_next = self.V[k + 1]
            if np.any(v_next):
                y = self._encode(v_next)
                # y_next = y[lo] + frac * (y[lo+1] - y[lo]), computed in place;
                # np.take is much faster than fancy indexing for this gather
                np.take(np.diff(y), lo, out=y_next)
                y_next *= frac
                y_next += np.take(y, lo)
                ev_next = self._expected_decoded(y_next, weights)
            else:
                ev_next = 0.0  # V_T = 0 when there is no bequest motive

            # --- B. Total Bellman Objective ---
            # Obj = U(c)*dt + exp(-rho*dt) * E[V]
            objective = flow_u + discount * ev_next

            # --- C. Find Max for every wealth level ---
            best_idx = np.argmax(objective, axis=1)

            self.V[k] = objective[rows, best_idx]
            self.opt_pi[k] = PI[best_idx]
            self.opt_c[k] = ALPHA[best_idx] # Storing alpha, not raw c

        logger.info("Backward Induction Complete.")

    def get_policy_function(self):
        """
        Returns a callable function policy(w, t, params) that interpolates
        the trained tables to find optimal pi and c.
        """

        def trained_policy(wealth_arr, t_val, params):
            # 1. Find the time step k whose interval [t_k, t_k + dt) contains t
            # (Clip to valid range [0, n_t-1])
            k = int(np.floor(t_val / self.dt + 1e-9))
            k = min(max(k, 0), self.n_t - 1)

            # 2. Interpolate Policy Tables at time k
            # We treat wealth_arr as a vector
            # Use NumPy's interp which is fast and robust
            pi_k = np.interp(wealth_arr, self.w_grid, self.opt_pi[k])
            alpha_k = np.interp(wealth_arr, self.w_grid, self.opt_c[k])

            # 3. Convert alpha back to raw consumption c
            c_k = alpha_k * wealth_arr

            return pi_k, c_k

        return trained_policy
