import numpy as np
import matplotlib.pyplot as plt


def simulate_gbm(
        initial_price: float = 100.0,
        drift: float = 0.07,
        volatility: float = 0.20,
        years: float = 1.0,
        dt_years: float = 1 / 252,
        seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate one GBM price path via Euler scheme.
    dS = mu*S*dt + sigma*S*dB
    """
    n_steps = int(round(years / dt_years))
    time_grid = np.linspace(0.0, years, n_steps + 1)

    rng = np.random.default_rng(seed)
    random_normals = rng.standard_normal(n_steps)

    prices = np.empty(n_steps + 1, dtype=float)
    prices[0] = initial_price

    for k in range(1, n_steps + 1):
        dt = dt_years
        dlogS = (drift - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * random_normals[k - 1]
        prices[k] = prices[k - 1] * np.exp(dlogS)

    return time_grid, prices


if __name__ == "__main__":
    t, s = simulate_gbm()
    plt.plot(t, s)
    plt.title("Simulated GBM Stock Price")
    plt.xlabel("Years")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()
