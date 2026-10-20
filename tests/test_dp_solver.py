import numpy as np
import pytest

from src.experiment import BASELINE, DP_AGENT, OPTIMUM, compare_policies, train_dp_agent
from src.models.params import ModelParams, SimConfig
from src.models.policies import finite_horizon_merton_policy, get_merton_values
from src.solvers.dp_solver import DPSolverFiniteHorizon


@pytest.fixture(scope="module", params=[2.0, 1.0, 0.5], ids=lambda g: f"gamma={g}")
def short_horizon(request):
    params = ModelParams(gamma=request.param, T=5.0)
    return params, train_dp_agent(params)


def test_policy_does_not_depend_on_wealth(short_horizon):
    # CRRA utility is homothetic: the optimal (pi, alpha) is the same at every wealth
    _, solver = short_horizon
    assert np.ptp(solver.opt_pi, axis=1).max() < 0.01
    assert (np.ptp(solver.opt_c, axis=1) / solver.opt_c.mean(axis=1)).max() < 0.01


def test_dp_agent_scores_close_to_closed_form(short_horizon):
    params, solver = short_horizon
    results = compare_policies(params, SimConfig(n_paths=2000, seed=0), solver)
    assert results[DP_AGENT]["CE"] == pytest.approx(results[OPTIMUM]["CE"], rel=0.005)
    assert results[DP_AGENT]["CE"] > results[BASELINE]["CE"]


@pytest.mark.parametrize("gamma", [2.0, 3.0])  # pi* = 0.5 is on the control grid; 1/3 is not
def test_policy_matches_closed_form(gamma):
    params = ModelParams(gamma=gamma)
    solver = train_dp_agent(params)
    pi_star, _ = get_merton_values(params)
    i = np.argmin(np.abs(solver.w_grid - 1.0))
    # Up to t = 20, alpha * dt stays small, so the discrete- and continuous-time
    # optima agree; right before T the discrete one invests less (see run_dp_solver)
    for k in range(0, 20 * 12, 6):
        _, c_exact = finite_horizon_merton_policy(np.array([1.0]), solver.time_grid[k], params)
        assert solver.opt_pi[k, i] == pytest.approx(pi_star, abs=0.01)
        assert solver.opt_c[k, i] == pytest.approx(c_exact[0], rel=0.01)


def test_policy_lookup_uses_the_step_containing_t():
    solver = DPSolverFiniteHorizon(ModelParams(T=1.0), n_time_steps=4)  # dt = 0.25
    solver.opt_pi[:] = np.arange(4)[:, None]  # step k -> pi = k
    policy = solver.get_policy_function()
    for t, k in [(0.0, 0), (0.24, 0), (0.25, 1), (0.74, 2), (0.99, 3), (1.0, 3)]:
        pi, _ = policy(np.array([1.0]), t, solver.params)
        assert pi[0] == k
