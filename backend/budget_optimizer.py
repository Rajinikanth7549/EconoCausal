import numpy as np
from scipy.optimize import LinearConstraint, milp, Bounds

def optimize_budget(ite_array: np.ndarray, order_values: np.ndarray, discount_costs: np.ndarray, budget: float):
    # Calculate expected incremental profit per customer
    expected_profit = (ite_array * order_values) - discount_costs
    num_customers = len(ite_array)
    
    # Require binary decision variables (0 or 1 for each customer)
    integrality = np.ones(num_customers)
    
    # Constraint: Total spent on discounts <= budget
    constraints = LinearConstraint(discount_costs.reshape(1, -1), lb=0, ub=budget)
    
    # Bounds: Each variable is between 0 and 1
    bounds = Bounds(lb=0, ub=1)

    # Scipy MILP minimizes objective, so we minimize -expected_profit
    res = milp(c=-expected_profit, integrality=integrality, constraints=constraints, bounds=bounds)
    
    selected_indices = np.where(res.x > 0.5)[0]
    total_spent = float(np.sum(discount_costs[selected_indices]))
    total_expected_profit = float(np.sum(expected_profit[selected_indices]))
    
    return {
        "selected_customer_count": len(selected_indices),
        "total_budget_spent": round(total_spent, 2),
        "expected_incremental_profit": round(total_expected_profit, 2),
        "selected_customer_ids": selected_indices.tolist()
    }