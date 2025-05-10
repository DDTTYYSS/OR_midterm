import numpy as np
import gurobipy as gp
from gurobipy import GRB

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
    for i in S_I:
        for t in S_T:
            in_inventory = gp.quicksum(
                x[i, j, t - T_lead[j]] for j in S_J if t - T_lead[j] >= 0
            )
            if t == 0:
                model.addConstr(v[i, t] == I_0[i] + I_1[i] + in_inventory - D[i, t])
            elif (t == 1):
                model.addConstr(v[i, t] == I_0[i] + I_2[i] + in_inventory - D[i, t])
            else:
                model.addConstr(v[i, t] == v[i, t-1] + in_inventory - D[i, t])

    M = sum(sum(D[i, t] for t in S_T) for i in S_I)  # Large number M
    for j in S_J:
        for t in S_T:
            model.addConstr(gp.quicksum(x[i, j, t] for i in S_I) <= M * y[j, t], name=f"ShippingMethod_{j}_{t}")

    for t in S_T:
        model.addConstr(
            gp.quicksum(V[i] * x[i, 2, t] for i in S_I) <= 30 * z[t]
        )

    model.addConstrs((0 <= y[j, t] for j in S_J for t in S_T))
    model.addConstrs((y[j, t] <= 1 for j in S_J for t in S_T))
    model.addConstrs((z[t] >= 0 for t in S_T))
    model.addConstrs((x[i, j, t] >= 0 for i in S_I for j in S_J for t in S_T))
    model.addConstrs((v[i, t] >= 0 for i in S_I for t in S_T))

    # Optimize the model
    model.optimize()

    # Return the objective value
    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        return None

# Main function to run the experiments
def main():
    scenarios = [
        ("medium", "medium", "medium"),
        ("small", "medium", "medium"),
        ("large", "medium", "medium"),
        ("medium", "low", "medium"),
        ("medium", "high", "medium"),
        ("medium", "medium", "low"),
        ("medium", "medium", "high"),
    ]

    results = []

    for scenario_id, (scale, container_cost, holding_cost) in enumerate(scenarios, 1):
        print(f"Running Scenario {scenario_id}: Scale={scale}, Container Cost={container_cost}, Holding Cost={holding_cost}")
        gaps = []  # Store optimality gaps for this scenario
        for instance_id in range(30):
            # Generate random instance
            instance = generate_instance(scale, container_cost, holding_cost)
            N, T, D, C_P, C_V1, C_V2, I_0, I_1, I_2, V, C_C, C_H, C_F, T_lead = instance

            # Solve using naive solution
            naive_cost = naive_solution(N, T, D, C_P, C_V1, I_0, I_1, I_2, C_H, C_F)

            # Solve using optimization model
            optimal_cost = solve_instance(N, T, D, C_P, C_V1, C_V2, I_0, I_1, I_2, V, C_C, C_H, C_F, T_lead)

            # Calculate optimality gap
            if optimal_cost is not None:
                gap = (naive_cost - optimal_cost) / naive_cost * 100
                gaps.append(gap)
                results.append((scenario_id, instance_id + 1, naive_cost, optimal_cost, gap))
                print(f"  Instance {instance_id + 1}: Naive Cost = {naive_cost}, Optimal Cost = {optimal_cost}, Gap = {gap:.2f}%")
            else:
                print(f"  Instance {instance_id + 1}: Optimization failed.")

        # Calculate average and standard deviation of gaps for this scenario
        avg_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        print(f"Scenario {scenario_id} - Average Gap: {avg_gap:.2f}%, Standard Deviation: {std_gap:.2f}%\n")

    import csv 
    # Export results to CSV
    with open("results.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Scenario ID", "Instance ID", "Naive Cost", "Optimal Cost", "Naive Time", "Optimal Time", "Optimality Gap (%)"])
        csvwriter.writerows(results)

    # Print summary
    print("\nSummary of Results:")
    for scenario_id, instance_id, naive_cost, optimal_cost, gap in results:
        print(f"Scenario {scenario_id}, Instance {instance_id}: Naive Cost = {naive_cost}, Optimal Cost = {optimal_cost}, Gap = {gap:.2f}%")

if __name__ == "__main__":
    main()