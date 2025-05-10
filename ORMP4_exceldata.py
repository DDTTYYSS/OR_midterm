import numpy as np
import gurobipy as gp
from gurobipy import GRB
from heuristic import read_data
import pandas as pd

# Function to generate random instances
def generate_instance(scale, container_cost, holding_cost):
    if scale == "small":
        N, T = 10, 6
    elif scale == "medium":
        N, T = 100, 20
    elif scale == "large":
        N, T = 500, 50

    # Generate random parameters
    D = np.random.uniform(0, 200, (N, T))  # Demand
    C_P = np.random.uniform(1000, 10000, N)  # Purchasing cost
    C_V1 = np.random.uniform(40, 100, N)  # Variable shipping cost for method 1
    alpha = np.random.uniform(0.4, 0.6, N)
    C_V2 = alpha * C_V1  # Variable shipping cost for method 2
    I_2 = np.where(np.random.rand(N) < 0.5, 0, np.random.uniform(0, 50, N))  # In-transit inventory (method 2)
    I_1 = np.where(np.random.rand(N) < 0.5, 0, np.random.uniform(0, 200, N))  # In-transit inventory (method 1)
    I_0 = np.random.uniform(D[:, 0], 400)  # Initial inventory
    V = np.random.uniform(0, 1, N)  # Volume

    # Container cost
    if container_cost == "low":
        C_C = 1375
    elif container_cost == "medium":
        C_C = 2750
    elif container_cost == "high":
        C_C = 5500

    # Holding cost
    if holding_cost == "low":
        C_H = 0.01 * C_P
    elif holding_cost == "medium":
        C_H = 0.02 * C_P
    elif holding_cost == "high":
        C_H = 0.04 * C_P

    # Fixed shipping cost
    C_F = np.array([100, 80, 50])

    # Lead times
    T_lead = np.array([1, 2, 3])

    return N, T, D, C_P, C_V1, C_V2, I_0, I_1, I_2, V, C_C, C_H, C_F, T_lead

def naive_solution(N, T, D, C_P, C_V1, I_0, I_1, I_2, C_H, C_F):
    total_cost = 0
    inventory = I_0.copy()  # Initial inventory

    for t in range(T):
        # Calculate ending inventory for this period
        if t == 0:
            ending_inventory = inventory + I_1 - D[:, t] # First period includes in-transit inventory from method 1
        elif t == 1:
            ending_inventory = inventory + I_2 - D[:, t] # Second period includes in-transit inventory from method 2
        else:
            ending_inventory = inventory - D[:, t]  # No additional in-transit inventory for later periods

        # Check if there is a shortage for the next month's demand
        if t < T - 1:  # No need to check for the last month
            shortages = np.maximum(0, D[:, t + 1] - np.maximum(0, ending_inventory))  # Shortages for next month
            if np.sum(shortages) > 0:
                # Place an order using shipping method 1
                order_cost = C_F[0] + np.sum(shortages * C_V1) + np.sum(shortages * C_P)
                total_cost += order_cost

                # Update inventory after the order arrives
                inventory = np.maximum(0, ending_inventory) + shortages
            else:
                # No shortages, update inventory
                inventory = np.maximum(0, ending_inventory)
        else:
            # For the last month, just update inventory
            inventory = np.maximum(0, ending_inventory)
            
        # Calculate holding cost for this period
        holding_cost = np.sum(C_H * np.maximum(0, inventory))  # Only positive inventory incurs holding cost
        total_cost += holding_cost
            
    return total_cost


# Function to solve the instance using the optimization model
def solve_instance(N, T, D, C_P, C_V1, C_V2, I_0, I_1, I_2, V, C_C, C_H, C_F, T_lead):
    # Create the Gurobi model
    model = gp.Model("InventoryManagement")

    # Set error parameter
    model.setParam('MIPGap', 0.0)

    # Define sets
    S_I = range(N)  # Products
    S_T = range(T)  # Time periods
    S_J = range(3)  # Shipping methods

    # Variables
    x = model.addVars(S_I, S_J, S_T, vtype=GRB.CONTINUOUS, name="x")  # Order quantity
    v = model.addVars(S_I, S_T, vtype=GRB.CONTINUOUS, name="v")  # Ending inventory
    y = model.addVars(S_J, S_T, vtype=GRB.CONTINUOUS, name="y")  # Relaxed binary for shipping method usage
    z = model.addVars(S_T, vtype=GRB.CONTINUOUS, name="z")  # Relaxed integer for number of containers

    # Objective function
    holding_cost = gp.quicksum(C_H[i] * v[i, t] for i in S_I for t in S_T)
    purchasing_and_shipping_cost = gp.quicksum(
        (C_P[i] + C_V1[i]) * x[i, 0, t] + (C_P[i] + C_V2[i]) * x[i, 1, t] + C_P[i] * x[i, 2, t]
        for i in S_I for t in S_T
    ) + gp.quicksum(C_F[j] * y[j, t] for j in S_J for t in S_T)
    container_cost = gp.quicksum(C_C * z[t] for t in S_T)

    model.setObjective(holding_cost + purchasing_and_shipping_cost + container_cost, GRB.MINIMIZE)

    # Constraints
    # Inventory balance
    J_in_inventory = np.array([1, 2, 3, 3, 3, 3])  # Number of shipping methods available at each time period

    for i in S_I:
        for t in S_T:
            # Compute the in-transit quantity arriving at time t
            in_inventory = 0
            for j in range(J_in_inventory[t]):
                in_inventory += x[i, j, t - T_lead[j] + 1]
            
            # Add the constraint for inventory balance
            if t == 0:
                model.addConstr(v[i, t] == in_inventory + I_0[i] + I_1[i] - D[i, t], name=f"InvBalance_{i}_{t}")
            elif t == 1:
                model.addConstr(v[i, t] == v[i, t-1] + in_inventory + I_2[i] - D[i, t], name=f"InvBalance_{i}_{t}")
                model.addConstr(v[i, t-1] >= D[i, t], name=f"Demand_{i}_{t}")
            else:
                model.addConstr(v[i, t] == v[i, t-1] + in_inventory - D[i, t], name=f"InvBalance_{i}_{t}")
                model.addConstr(v[i, t-1] >= D[i, t], name=f"Demand_{i}_{t}")

    # Relate order quantity and shipping method
    M = sum(sum(D[i, t] for t in S_T) for i in S_I)  # Large number M
    for j in S_J:
        for t in S_T:
            model.addConstr(gp.quicksum(x[i, j, t] for i in S_I) <= M * y[j, t], name=f"ShippingMethod_{j}_{t}")

    # Container constraint
    for t in S_T:
        model.addConstr(
            gp.quicksum(V[i] * x[i, 2, t] for i in S_I) <= 30 * z[t],
            name=f"Container_{t}"
        )

    # Non-negativity and binary constraints
    for i in S_I:
        for j in S_J:
            for t in S_T:
                model.addConstr(x[i, j, t] >= 0, name=f"NonNeg_x_{i}_{j}_{t}")
    for i in S_I:
        for t in S_T:
            model.addConstr(v[i, t] >= 0, name=f"NonNeg_v_{i}_{t}")
    for j in S_J:
        for t in S_T:
            model.addConstr(y[j, t] >= 0, name=f"Binary_y_{j}_{t}")
            model.addConstr(y[j, t] <= 1, name=f"Binary_y_upper_{j}_{t}")
    for t in S_T:
        model.addConstr(z[t] >= 0, name=f"NonNeg_z_{t}")

    # Optimize the model
    model.optimize()

    # Print the solution
    if model.status == GRB.OPTIMAL:
        print("\nOptimal objective value:", model.objVal)
        print("\nOrder quantities (x_ijt):")
        for t in S_T:
            for i in S_I:
                for j in S_J:
                    if x[i, j, t].x > 0:
                        print(f"x[{i+1},{j+1},{t+1}] = {x[i, j, t].x}")
        
        print("\nEnding inventory (v_it):")
        for t in S_T:
            for i in S_I:
                if v[i, t].x > 0:
                    print(f"v[{i+1},{t+1}] = {v[i, t].x}")
        
        print("\nShipping method usage (y_jt):")
        for t in S_T:
            for j in S_J:
                if y[j, t].x > 0:
                    print(f"y[{j+1},{t+1}] = {y[j, t].x}")
        
        print("\nNumber of containers (z_t):")
        for t in S_T:
            if z[t].x > 0:
                print(f"z[{t+1}] = {z[t].x}")
        
        return model.objVal
    else:
        print("No optimal solution found.")
        return None

# Main function to run the experiments
def main():
    # Read data from Excel file
    file_path = "OR113-2_midtermProject_data.xlsx"
    try:
        # Read data using heuristic's read_data function
        N, T, J, D, I_0, C, V, V_C, lead_times, in_transit = read_data(file_path)
        
        print("\nRunning Excel Data Test")
        print(f"Number of products (N): {N}")
        print(f"Number of time periods (T): {T}")
        
        # Read inventory cost data for holding costs
        df_inventory_cost = pd.read_excel(file_path, sheet_name="Inventory cost")
        
        # Convert data format to match solve_instance parameters
        C_P = C["P"]
        C_V1 = C["V"][:, 1]  # Air freight costs
        C_V2 = C["V"][:, 2]  # Express shipping costs
        C_C = C["C"]
        C_H = np.array([float(df_inventory_cost.iloc[i, 3]) for i in range(N)])  # Holding costs from column 4
        C_F = np.array([100, 80, 50])  # Fixed costs for express, air, ocean
        
        # Convert lead times to array format
        T_lead = np.array([lead_times["express"], lead_times["air"], lead_times["ocean"]])
        
        # Convert in-transit inventory
        I_1 = in_transit["march"]
        I_2 = in_transit["april"]
        
        # Solve using optimization model
        optimal_cost = solve_instance(N, T, D, C_P, C_V1, C_V2, I_0, I_1, I_2, V, C_C, C_H, C_F, T_lead)
        
        if optimal_cost is not None:
            print(f"\nOptimal Cost: {optimal_cost:.2f}")
        else:
            print("Optimization failed.")
            
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
