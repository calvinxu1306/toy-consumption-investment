import streamlit as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import traceback # To print the full error

# Import your modules
from src.models.params import ModelParams, SimConfig
from src.models.dynamics import simulate
from src.models.policies import constant_policy
from src.solvers.dp_solver import DPSolverFiniteHorizon

# Page Config
st.set_page_config(page_title="Wealth Optimizer", layout="wide")

st.title("Optimal Consumption & Investment Simulator")

# --- Sidebar: User Inputs ---
st.sidebar.header("Model Parameters")
mu = st.sidebar.number_input("Expected Return (Drift)", 0.0, 0.2, 0.06, 0.01)
sigma = st.sidebar.number_input("Volatility (Sigma)", 0.05, 0.5, 0.20, 0.01)
r = st.sidebar.number_input("Risk-Free Rate", 0.0, 0.1, 0.02, 0.01)
gamma = st.sidebar.slider("Risk Aversion (Gamma)", 0.5, 10.0, 2.0, 0.1)
T = st.sidebar.slider("Time Horizon (Years)", 5, 60, 30, 5)

# LOWER DEFAULT PATHS to prevent memory crashes during debugging
n_paths = st.sidebar.slider("Simulation Paths", 50, 5000, 500, 50) 

if st.button("Run Simulation"):
    status = st.empty() # Placeholder for status updates
    
    try:
        # --- STEP 1: SETUP ---
        status.write("Step 1/4: Initializing Parameters...")
        params = ModelParams(mu=mu, r=r, sigma=sigma, gamma=gamma, T=float(T))
        sim_cfg = SimConfig(n_paths=n_paths, dt=1/252)
        
        # --- STEP 2: TRAINING ---
        status.write("Step 2/4: Training AI Agent (Backward Induction)...")
        # Using coarse grid for speed
        solver = DPSolverFiniteHorizon(params, n_w_points=50, n_time_steps=int(T*4))
        solver.initialize_terminal_condition()
        solver.train_backward()
        smart_policy = solver.get_policy_function()
        
        def smart_wrapper(w, t, p):
            return smart_policy(w, t, p)
        
        # --- STEP 3: SIMULATION ---
        status.write("Step 3/4: Running Monte Carlo Simulations...")
        
        # Define Baseline
        merton_pi = (mu - r) / (gamma * sigma**2)
        merton_pi = max(0, min(merton_pi, 2.0))
        
        def baseline_wrapper(w, t, p):
            return constant_policy(w, t, p, pi_const=merton_pi, alpha_consume=0.04)
        
        # Run Sims
        res_base = simulate(params, sim_cfg, baseline_wrapper)
        res_smart = simulate(params, sim_cfg, smart_wrapper)
        
        # --- STEP 4: PLOTTING ---
        status.write("Step 4/4: Generating Plots...")
        
        col1, col2 = st.columns(2)
        u_base = res_base["DU"]
        u_smart = res_smart["DU"]
        imp = ((u_smart - u_base) / abs(u_base)) * 100
        
        with col1:
            st.metric(label="Baseline Utility", value=f"{u_base:.4f}")
        with col2:
            st.metric(label="Smart Agent Utility", value=f"{u_smart:.4f}", delta=f"{imp:.2f}%")

        st.subheader("Wealth Trajectories")
        
        # Create figure explicitly
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot with raw strings (r"") to fix warnings
        ax.plot(res_base["times"], res_base["W"].mean(axis=0), 'b--', label=rf"Baseline (Fixed $\pi$={merton_pi:.2f})")
        ax.plot(res_smart["times"], res_smart["W"].mean(axis=0), 'r-', label="Smart Agent (Dynamic)")
        
        ax.set_xlabel("Years")
        ax.set_ylabel("Average Wealth")
        ax.grid(alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
        plt.close(fig) 
        
        status.success("Simulation Complete!")

    except Exception as e:
        # If it crashes, this will show the error RED on screen instead of white screen
        st.error("An error occurred!")
        st.code(traceback.format_exc())