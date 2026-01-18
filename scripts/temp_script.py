import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.params import ModelParams
from src.models.policies import get_merton_values

params = ModelParams()  # Uses your defaults (mu=0.06, r=0.02, sigma=0.2, gamma=2)
pi, alpha = get_merton_values(params)

print(f"Theoretical Optimal Risky Weight (pi*): {pi:.4f}")
print(f"Theoretical Optimal Consumption (alpha*): {alpha:.4f}")