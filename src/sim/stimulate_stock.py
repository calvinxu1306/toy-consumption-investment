import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm(S0=100, mu=0.07, sigma=0.2, T=1, dt=1/252, seed=42):
    np.random.seed(seed)
    N = int(T/dt)
    S = np.zeros(N+1)
    S[0] = S0
    for t in range(1, N+1):
        Z = np.random.normal()
        S[t] = S[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    return S

if __name__ == "__main__":
    S = simulate_gbm()
    plt.plot(S)
    plt.title("Simulated Stock Price Path")
    plt.xlabel("Steps")
    plt.ylabel("Price")
    plt.show()
