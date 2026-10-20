import numpy as np
import pytest

from src.models.params import ModelParams
from src.models.policies import finite_horizon_merton_policy, get_merton_values


def test_merton_values_at_defaults():
    pi_star, alpha_star = get_merton_values(ModelParams())
    assert pi_star == pytest.approx(0.5)
    assert alpha_star == pytest.approx(0.03)


def test_log_utility_consumes_at_discount_rate():
    _, alpha_star = get_merton_values(ModelParams(gamma=1.0))
    assert alpha_star == pytest.approx(0.03)


def test_merton_weight_is_clipped_to_leverage_limit():
    params = ModelParams(gamma=0.5)  # unconstrained pi* = 2.0
    pi_star, alpha_star = get_merton_values(params)
    assert pi_star == params.pi_max
    # alpha* is computed from the certainty-equivalent return of the clipped weight
    ce_return = params.r + 1.5 * (params.mu - params.r) - 0.5 * 0.5 * (params.sigma * 1.5) ** 2
    assert alpha_star == pytest.approx((params.rho - 0.5 * ce_return) / 0.5)


def test_finite_horizon_policy_matches_infinite_horizon_far_from_T():
    params = ModelParams(T=1000.0)
    pi_star, alpha_star = get_merton_values(params)
    pi, c = finite_horizon_merton_policy(np.array([2.0]), 0.0, params)
    assert pi[0] == pytest.approx(pi_star)
    assert c[0] == pytest.approx(2.0 * alpha_star, rel=1e-6)


def test_finite_horizon_policy_runs_wealth_down_near_T():
    params = ModelParams()
    _, c = finite_horizon_merton_policy(np.array([1.0]), params.T - 0.01, params)
    assert c[0] == pytest.approx(1 / 0.01, rel=1e-3)  # alpha ~ 1 / (T - t)


def test_finite_horizon_policy_with_bequest_stays_finite_at_T():
    params = ModelParams(kappa=1.0)
    _, c = finite_horizon_merton_policy(np.array([1.0]), params.T, params)
    assert c[0] == pytest.approx(1.0)  # alpha(T) = kappa^(-1/gamma)
