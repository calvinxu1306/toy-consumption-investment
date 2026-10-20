import streamlit as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import your modules
from src.experiment import BASELINE, compare_policies, dp_gap, gain_vs_baseline, plot_comparison, train_dp_agent
from src.models.params import ModelParams, SimConfig
from src.models.policies import get_merton_values

# Page Config
st.set_page_config(page_title="Wealth Optimizer", layout="wide")

st.title("Optimal Consumption & Investment Simulator")

# --- Sidebar: User Inputs ---
st.sidebar.header("Model Parameters")
mu = st.sidebar.number_input("Expected Return (Drift)", 0.0, 0.2, 0.06, 0.01)
sigma = st.sidebar.number_input("Volatility (Sigma)", 0.05, 0.5, 0.20, 0.01)
r = st.sidebar.number_input("Risk-Free Rate", 0.0, 0.1, 0.02, 0.01)
rho = st.sidebar.number_input("Discount Rate (Rho)", 0.0, 0.1, 0.03, 0.01)
gamma = st.sidebar.slider("Risk Aversion (Gamma)", 0.5, 10.0, 2.0, 0.1)
T = st.sidebar.slider("Time Horizon (Years)", 5, 60, 30, 5)
kappa = st.sidebar.slider("Bequest Weight (Kappa)", 0.0, 5.0, 0.0, 0.5,
                          help="Utility weight on wealth left at T. 0 = no bequest motive.")
n_paths = st.sidebar.slider("Simulation Paths", 500, 10000, 5000, 500)


@st.cache_resource(show_spinner=False)
def get_trained_solver(params: ModelParams):
    # Cached per parameter set: changing only the number of paths does not retrain
    return train_dp_agent(params)


if st.button("Run Simulation"):
    params = ModelParams(mu=mu, r=r, sigma=sigma, rho=rho, gamma=gamma, T=float(T), kappa=kappa)

    with st.spinner("Training DP agent (backward induction)..."):
        solver = get_trained_solver(params)
    with st.spinner("Running Monte Carlo simulations..."):
        results = compare_policies(params, SimConfig(n_paths=n_paths), solver)

    # --- Scores: certainty-equivalent consumption ---
    st.subheader("Certainty-Equivalent Consumption")
    st.caption("The sure, constant consumption rate each policy is worth to the agent "
               "(higher is better). Unlike raw utility, percentage differences are meaningful.")
    for col, (name, res) in zip(st.columns(3), results.items()):
        delta = None if name == BASELINE else f"{gain_vs_baseline(results, name):+.1%} vs baseline"
        col.metric(label=name, value=f"{res['CE']:.4f}", delta=delta)

    pi_star, alpha_star = get_merton_values(params)
    st.caption(f"Static Merton uses pi* = {pi_star:.2f} and alpha* = {max(alpha_star, 0.0):.4f} "
               f"at every date. The DP agent is within {dp_gap(results)} of the closed-form optimum.")

    st.subheader("Wealth and Consumption Trajectories")
    fig = plot_comparison(results)
    st.pyplot(fig)
    plt.close(fig)
