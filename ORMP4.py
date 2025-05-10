import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Function to generate random instances adapted to the example code format
def generate_instance(scale, container_cost, holding_cost):
    if scale == "small":
        N, T = 10, 6
    elif scale == "medium":
        N, T = 100, 20
    elif scale == "large":
        N, T = 500, 50
    
    J = 3  # Number of shipping methods
    
    # Generate random parameters
    D = np.random.uniform(0, 200, (N, T))  # Demand
    I_0 = np.random.uniform(D[:, 0], 400)  # Initial inventory
    
    # In-transit inventory
    I = np.zeros([N, T])
    I[:, 0] = np.where(np.random.rand(N) < 0.5, 0, np.random.uniform(0, 50, N))  # March
    I[:, 1] = np.where(np.random.rand(N) < 0.5, 0, np.random.uniform(0, 200, N))  # April
    
    # Volume per unit
    V = np.random.uniform(0, 1, N)
    V_C = 30  # Container volume capacity
    
    # Cost parameters following example code structure
    C = {
        "H": np.zeros([N]),     # Holding cost
        "P": np.zeros([N]),     # Purchasing cost
        "V": np.zeros([N, J]),  # Variable shipping cost
        "F": np.array([100, 80, 50]),  # Fixed shipping cost for express, air, ocean
        "C": 0,                 # Container cost (will be set below)
    }
    
    # Set purchasing costs
    C["P"] = np.random.uniform(1000, 10000, N)
    
    # Set shipping costs
    for i in range(N):
        # Express shipping cost
        C["V"][i, 0] = np.random.uniform(40, 100)
        # Air freight cost
        C["V"][i, 1] = C["V"][i, 0] * np.random.uniform(0.4, 0.6)
        # Ocean shipping cost is 0 in variable cost since it's handled through container cost
        C["V"][i, 2] = 0
    
    # Set container cost based on parameter
    if container_cost == "low":
        C["C"] = 1375
    elif container_cost == "medium":
        C["C"] = 2750
    elif container_cost == "high":
        C["C"] = 5500
    
    # Set holding cost based on parameter
    if holding_cost == "low":
        C["H"] = 0.01 * C["P"]
    elif holding_cost == "medium":
        C["H"] = 0.02 * C["P"]
    elif holding_cost == "high":
        C["H"] = 0.04 * C["P"]
    
    # Lead times
    T_lead = np.array([1, 2, 3])  # For express, air, ocean
    
    return N, T, D, I_0, I, C, V, V_C, T_lead


def naive_solution(N, T, D, I_0, I, C):
    """
    A simple heuristic approach that:
    1. Uses in-transit inventory for the first two periods
    2. Orders just enough to meet next period demand when shortages are predicted
    3. Uses air shipping as a default method (index 1)
    4. Calculates holding costs based on ending inventory
    """
    total_cost = 0
    inventory = I_0.copy()  # Initial inventory
    
    print("\nRunning naive solution:")
    
    for t in range(T):
        print(f"\nPeriod {t+1}:")
        # Calculate ending inventory for this period
        if t == 0:
            ending_inventory = inventory + I[:, 0] - D[:, t]  # First period includes in-transit inventory
            print(f"  Using in-transit inventory for March")
        elif t == 1:
            ending_inventory = inventory + I[:, 1] - D[:, t]  # Second period includes in-transit inventory
            print(f"  Using in-transit inventory for April")
        else:
            ending_inventory = inventory - D[:, t]  # No additional in-transit inventory for later periods
        
        print(f"  Starting inventory: {np.sum(inventory):.2f}")
        print(f"  Demand: {np.sum(D[:, t]):.2f}")
        print(f"  Ending inventory before ordering: {np.sum(np.maximum(0, ending_inventory)):.2f}")
        
        # Check if there is a shortage for the next month's demand
        if t < T - 1:  # No need to check for the last month
            shortages = np.maximum(0, D[:, t + 1] - np.maximum(0, ending_inventory))  # Shortages for next month
            
            if np.sum(shortages) > 0:
                # Place an order using express shipping (method index 0)
                fixed_cost = C["F"][0]
                variable_cost = np.sum(shortages * C["V"][:, 0])
                purchase_cost = np.sum(shortages * C["P"])
                order_cost = fixed_cost + variable_cost + purchase_cost
                
                total_cost += order_cost
                
                print(f"  Shortage detected for next period: {np.sum(shortages):.2f}")
                print(f"  Ordering using air shipping:")
                print(f"    Fixed cost: {fixed_cost:.2f}")
                print(f"    Variable shipping cost: {variable_cost:.2f}")
                print(f"    Purchase cost: {purchase_cost:.2f}")
                print(f"    Total order cost: {order_cost:.2f}")
                
                # Update inventory after the order arrives
                inventory = np.maximum(0, ending_inventory) + shortages
            else:
                print("  No shortages predicted for next period, no ordering needed")
                # No shortages, update inventory
                inventory = np.maximum(0, ending_inventory)
        else:
            print("  Last period, no ordering needed")
            # For the last month, just update inventory
            inventory = np.maximum(0, ending_inventory)
        
        # Calculate holding cost for this period
        holding_cost = np.sum(C["H"] * np.maximum(0, inventory))  # Only positive inventory incurs holding cost
        total_cost += holding_cost
        
        print(f"  Ending inventory after ordering: {np.sum(inventory):.2f}")
        print(f"  Holding cost: {holding_cost:.2f}")
    
    print(f"\nNaive solution total cost: {total_cost:.2f}")
    return total_cost


# Function to solve the instance using the optimization model - exactly matching the example code
def solve_instance(N, T, D, I_0, I, C, V, V_C, T_lead):
    # Create the Gurobi model
    model = gp.Model("InventoryManagement")
    
    # Set error parameter
    model.setParam('MIPGap', 0.0)
    
    # Define sets
    S_I = range(N)  # Products i in {0,  ..., N-1}
    S_T = range(T)  # Time periods t in {0, ..., T-1}
    S_J = range(len(T_lead))  # Shipping methods j in {0, ..., J-1}
    
    # Variables
    x = model.addVars(S_I, S_J, S_T, vtype=GRB.CONTINUOUS, name="x")  # Order quantity x_ijt
    v = model.addVars(S_I, S_T, vtype=GRB.CONTINUOUS, name="v")  # Ending inventory v_it
    y = model.addVars(S_J, S_T, vtype=GRB.CONTINUOUS, name="y")  # Binary for shipping method y_jt
    z = model.addVars(S_T, vtype=GRB.CONTINUOUS, name="z")  # Number of containers z_t
    
    # Objective function (1)
    # Holding cost + (Purchasing cost + Variable shipping cost + Fixed shipping cost) + Container cost
    holding_cost = gp.quicksum(C["H"][i] * v[i, t] for i in S_I for t in S_T)
    purchasing_and_shipping_cost = gp.quicksum(
        (C["P"][i] + C["V"][i, j]) * x[i, j, t]
        for i in S_I for j in S_J for t in S_T
    ) + gp.quicksum(C["F"][j] * y[j, t] for t in S_T for j in S_J)
    container_cost = gp.quicksum(C["C"] * z[t] for t in S_T)
    
    model.setObjective(holding_cost + purchasing_and_shipping_cost + container_cost, GRB.MINIMIZE)
    
    # Constraints
    # Inventory balance (2)
    
    for i in S_I:
        for t in S_T:
            # Compute the in-transit quantity arriving at time t
            in_inventory = 0
            in_inventory = gp.quicksum(
                x[i, j, t - T_lead[j] + 1] for j in S_J if t - T_lead[j] >= -1
            )
            # Add the constraint for inventory balance
            if t == 0:
                model.addConstr(v[i, t] == in_inventory + I_0[i] + I[i, t] - D[i, t], name=f"InvBalance_{i}_{t}")
            else:
                model.addConstr(v[i, t] == v[i, t-1] + in_inventory + I[i, t] - D[i, t], name=f"InvBalance_{i}_{t}")
                model.addConstr(v[i, t-1] >= D[i, t], name=f"Demand_{i}_{t}")
    
    # Relate order quantity and shipping method (4)
    M = sum(sum(D[i, t] for t in S_T) for i in S_I)  # Large number M as per problem statement
    for j in S_J:
        for t in S_T:
            model.addConstr(gp.quicksum(x[i, j, t] for i in S_I) <= M * y[j, t], name=f"ShippingMethod_{j}_{t}")
    
    # Container constraint (5)
    for t in S_T:
        model.addConstr(
            gp.quicksum(V[i] * x[i, 2, t] for i in S_I) <= V_C * z[t],
            name=f"Container_{t}"
        )
    
    # Non-negativity and binary constraints (6)
    for i in S_I:
        for j in S_J:
            for t in S_T:
                model.addConstr(x[i, j, t] >= 0, name=f"NonNeg_x_{i}_{j}_{t}")
    for i in S_I:
        for t in S_T:
            model.addConstr(v[i, t] >= 0, name=f"NonNeg_v_{i}_{t}")
    for j in S_J:
        for t in S_T:
            model.addConstr(y[j, t] >= 0, name=f"Binary_y_{j}_{t}")  # Already binary due to vtype
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
                        print(f"x[{i+1},{j+1},{t+1}] = {x[i, j, t].x}")  # +1 to make the index consistent
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
        
        # Calculate individual costs for reporting
        total_holding_cost = sum(C["H"][i] * v[i, t].x for i in S_I for t in S_T)
        total_purchasing_cost = sum((C["P"][i]) * x[i, j, t].x for i in S_I for j in S_J for t in S_T)
        total_variable_shipping_cost = sum(C["V"][i, j] * x[i, j, t].x for i in S_I for j in S_J for t in S_T)
        total_fixed_shipping_cost = sum(C["F"][j] * y[j, t].x for j in S_J for t in S_T)
        total_container_cost = sum(C["C"] * z[t].x for t in S_T)
        
        print("\nCost Breakdown:")
        print(f"Holding Cost: {total_holding_cost:.2f}")
        print(f"Purchasing Cost: {total_purchasing_cost:.2f}")
        print(f"Variable Shipping Cost: {total_variable_shipping_cost:.2f}")
        print(f"Fixed Shipping Cost: {total_fixed_shipping_cost:.2f}")
        print(f"Container Cost: {total_container_cost:.2f}")
        print(f"Total Cost: {model.objVal:.2f}")
        
        return model.objVal
    else:
        print("No optimal solution found.")
        return None


# Main function to run the experiments
def main():
    # Read data from Excel file using the approach from example code
    file_path = "OR113-2_midtermProject_data.xlsx"
    try:
        # Read the Excel file from the 'Demand' sheet
        df_demand = pd.read_excel(file_path, sheet_name="Demand")
        N = df_demand.shape[0] - 1   # -1 because of the first row, +1 for indices' consistency
        T = df_demand.shape[1] - 2  # -2 because of the first two columns, +1 for indices' consistency  
        print("N:", N, "T:", T)
        
        # Display the dataframe to verify the data
        I = np.zeros([N, T])
        D = np.zeros([N, T])
        I_0 = np.zeros([N])
        
        for i in range(N):
            I_0[i] = df_demand.iloc[i+1, 1]
            for t in range(T):
                D[i, t] = df_demand.iloc[i+1, t+2]
        
        print("I_0:", I_0)
        print("D:", D)
        
        # Read the Excel file from the 'In-transit' sheet
        df_in_transit = pd.read_excel(file_path, sheet_name="In-transit")
        for i in range(N):
            for t in range(df_in_transit.shape[1] - 1):
                I[i, t] = df_in_transit.iloc[i+1, t+1]
        print("I:", I)
        
        # Read the Excel file from the 'Shipping cost' sheet
        df_shipping_cost = pd.read_excel(file_path, sheet_name="Shipping cost")
        J = df_shipping_cost.shape[1] - 1 # -1 because of the first column
        df_inventory_cost = pd.read_excel(file_path, sheet_name="Inventory cost")
        
        C = {
            "H": np.zeros([N]),
            "P": np.zeros([N]),
            "V": np.zeros([N, J]),
            "F": np.array([100, 80, 50]),
            "C": 2750,
        }
        V = np.zeros([N])
        V_C = 30
        for i in range(N):
            C["H"][i] = df_inventory_cost.iloc[i, 3]
            C["P"][i] = df_inventory_cost.iloc[i, 2]
            V[i] = df_shipping_cost.iloc[i, 3]
            for j in range(J):
                if j == J - 1:
                    C["V"][i, j] = 0
                else:
                    C["V"][i, j] = df_shipping_cost.iloc[i, j+1]
        
        print("C:", C)
        print("V:", V)
        T_lead = np.array([1, 2, 3]) # T_j
        
        # Solve using optimization model
        optimal_cost = solve_instance(N, T, D, I_0, I, C, V, V_C, T_lead)
        
        if optimal_cost is not None:
            print(f"\nOptimal Cost: {optimal_cost:.2f}")
            
            # Run naive solution for comparison
            naive_cost = naive_solution(N, T, D, I_0, I, C)
            print(f"Naive Solution Cost: {naive_cost:.2f}")
            print(f"Improvement: {(naive_cost - optimal_cost) / naive_cost * 100:.2f}%")
        else:
            print("Optimization failed.")
            
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
