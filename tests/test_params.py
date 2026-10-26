import os
import subprocess
import sys

import pytest

from src.models.params import ModelParams, SimConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("kwargs", [
    dict(gamma=0.0), dict(gamma=-1.0), dict(sigma=-0.2), dict(T=0.0), dict(T=-5.0),
    dict(kappa=-1.0), dict(pi_max=-0.5), dict(mu=float("nan")), dict(T=float("inf")),
])
def test_model_params_reject_invalid_values(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        ModelParams(**kwargs)


def test_model_params_accept_boundary_values():
    ModelParams(sigma=0.0, kappa=0.0, pi_max=0.0, r=-0.01, rho=0.0)


@pytest.mark.parametrize("kwargs", [
    dict(n_paths=0), dict(n_paths=1), dict(dt=0.0), dict(dt=float("nan")), dict(seed=-1),
])
def test_sim_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        SimConfig(**kwargs)


@pytest.mark.parametrize("args, message", [
    (["--gamma", "0"], "gamma must be > 0"),
    (["--sigma", "-0.2"], "sigma must be >= 0"),
    (["--T", "-5"], "T must be > 0"),
    (["--paths", "1"], "n_paths must be >= 2"),
])
def test_cli_reports_invalid_arguments_without_traceback(args, message, tmp_path):
    proc = subprocess.run([sys.executable, "main.py", *args, "--output", str(tmp_path)],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2
    assert message in proc.stderr
    assert "Traceback" not in proc.stderr
