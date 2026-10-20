import numpy as np
import pytest

from src.models.dynamics import simulate
from src.models.metrics import certainty_equivalent
from src.models.params import ModelParams, SimConfig
from src.models.policies import constant_policy, zero_consumption_with_constant_pi


def test_output_shapes_are_paths_by_time():
    res = simulate(ModelParams(T=2.0), SimConfig(n_paths=50, dt=1 / 12), constant_policy)
    assert res["W"].shape == (50, 25)
    assert res["C"].shape == res["Pi"].shape == (50, 24)
    assert res["times"].shape == (25,)
    assert res["U"].shape == (50,)


def test_zero_volatility_matches_compound_growth():
    params = ModelParams(sigma=0.0, T=10.0)
    res = simulate(params, SimConfig(n_paths=3, dt=1 / 12), zero_consumption_with_constant_pi)
    growth = params.r + 0.6 * (params.mu - params.r)
    np.testing.assert_allclose(res["W"][:, -1], (1 + growth / 12) ** 120)


def test_same_seed_gives_same_paths():
    a = simulate(ModelParams(T=1.0), SimConfig(n_paths=20, seed=3), constant_policy)
    b = simulate(ModelParams(T=1.0), SimConfig(n_paths=20, seed=3), constant_policy)
    np.testing.assert_array_equal(a["W"], b["W"])


@pytest.mark.parametrize("gamma", [0.5, 1.0, 3.0])
def test_certainty_equivalent_of_constant_consumption(gamma):
    # No risk and consumption of r * W keep wealth at 1 and consumption at r
    params = ModelParams(gamma=gamma, sigma=0.0)

    def policy(w, t, p):
        return constant_policy(w, t, p, pi_const=0.0, alpha_consume=p.r)

    res = simulate(params, SimConfig(n_paths=2), policy)
    assert certainty_equivalent(res["DU"], params, res["times"]) == pytest.approx(params.r)
