import pandas as pd
import numpy as np

def read_data(file_path):
    # Read the Excel file from the 'Demand' sheet
    df_demand = pd.read_excel(file_path, sheet_name="Demand")
    N = df_demand.shape[0] - 1
    T = df_demand.shape[1] - 2
    
    # Initialize arrays
    D = np.zeros([N, T])
    I_0 = np.zeros([N])
    
    # Read demand data
    for i in range(N):
        I_0[i] = df_demand.iloc[i+1, 1]
        for t in range(T):
            D[i, t] = df_demand.iloc[i+1, t+2]
    
    # Read shipping cost data
    df_shipping_cost = pd.read_excel(file_path, sheet_name="Shipping cost")
    J = 2  # Only considering ocean (j=0) and air (j=1) freight
    df_inventory_cost = pd.read_excel(file_path, sheet_name="Inventory cost")
    
    # Initialize cost parameters
    C = {
        "P": np.zeros([N]),  # Purchasing cost
        "V": np.zeros([N, J]),  # Variable shipping cost (only used for air freight)
        "C": 2750,  # Container cost
    }
    V = np.zeros([N])  # Volume per unit
    V_C = 30  # Container volume capacity
    
    for i in range(N):
        C["P"][i] = df_inventory_cost.iloc[i, 2]
        V[i] = df_shipping_cost.iloc[i, 3]
        # Only set air freight cost
        C["V"][i, 1] = df_shipping_cost.iloc[i, 2]  # Air freight cost
        # Ocean freight cost will be calculated based on volume ratio
        C["V"][i, 0] = (V[i] / V_C) * C["C"]  # Cost per unit for ocean based on volume ratio
    
    lead_times = {"ocean": 3, "air": 1}  # Lead times in months
    
    return N, T, J, D, I_0, C, V, V_C, lead_times

def heuristic_solution(N, T, J, D, I_0, C, V, V_C, lead_times):
    # Initialize solution arrays
    x = np.zeros((N, J, T))  # Order quantities
    inventory = np.zeros((N, T))  # Ending inventory for each period
    z = np.zeros(T)  # Number of containers
    
    # Initialize inventory with initial values
    for i in range(N):
        if T > 0:
            inventory[i, 0] = I_0[i]
    
    # For each product
    for i in range(N):
        print(f"\nProduct {i+1}:")
        # For each period where we need to consider ordering
        for t in range(T):
            # Calculate what will arrive this period from previous orders
            arriving_ocean = 0
            arriving_air = 0
            
            # Check for ocean shipments arriving
            if t >= lead_times["ocean"]:
                arriving_ocean = x[i, 0, t - lead_times["ocean"]]
            
            # Check for air shipments arriving
            if t >= lead_times["air"]:
                arriving_air = x[i, 1, t - lead_times["air"]]
            
            # Calculate current inventory before demand
            if t == 0:
                current_inventory = I_0[i] + arriving_ocean + arriving_air
            else:
                current_inventory = inventory[i, t-1] + arriving_ocean + arriving_air
            
            # Calculate required quantity to prevent negative inventory
            required_qty = D[i, t]  # We need at least enough for current period's demand
            
            if current_inventory < required_qty:
                # We need to order the difference to prevent negative inventory
                order_qty = required_qty - current_inventory
                
                print(f"\nPeriod {t+1}:")
                print(f"Demand: {D[i, t]}")
                print(f"Current inventory before demand: {current_inventory}")
                print(f"Need to order: {order_qty}")
                
                # Calculate order costs for both methods
                ocean_cost = C["V"][i, 0] * order_qty
                air_cost = C["V"][i, 1] * order_qty
                
                print(f"Ocean cost: {ocean_cost:.2f}")
                print(f"Air cost: {air_cost:.2f}")
                
                # Determine when we need to place the order
                ocean_order_time = max(0, t - lead_times["ocean"])
                air_order_time = max(0, t - lead_times["air"])
                
                # Choose shipping method based on cost and timing
                if t >= lead_times["ocean"]:  # Can use either method
                    if ocean_cost <= air_cost:
                        x[i, 0, ocean_order_time] += order_qty
                        print(f"Using ocean shipping, order in period {ocean_order_time + 1}")
                    else:
                        x[i, 1, air_order_time] += order_qty
                        print(f"Using air shipping, order in period {air_order_time + 1}")
                else:  # Must use air due to lead time
                    x[i, 1, air_order_time] += order_qty
                    print(f"Must use air shipping due to lead time, order in period {air_order_time + 1}")
            
            # Update ending inventory after demand
            inventory[i, t] = max(0, current_inventory - D[i, t])  # Ensure non-negative inventory
            print(f"Ending inventory for period {t+1}: {inventory[i, t]}")
    
    # Calculate containers needed for each period
    for t in range(T):
        total_volume = sum(V[i] * x[i, 0, t] for i in range(N))
        if total_volume > 0:
            z[t] = np.ceil(total_volume / V_C)
            print(f"\nPeriod {t+1} ocean shipping:")
            print(f"Total volume: {total_volume:.2f}")
            print(f"Containers needed: {z[t]:.0f}")
    
    return x, inventory, z

def calculate_total_cost(x, z, C, N, T, J):
    # Calculate purchasing cost
    purchasing_cost = sum(C["P"][i] * sum(x[i, j, t] for j in range(J) for t in range(T))
                         for i in range(N))
    
    # Calculate air shipping cost
    air_shipping_cost = sum(C["V"][i, 1] * x[i, 1, t]
                           for i in range(N) for t in range(T))
    
    # Calculate ocean shipping cost (based on actual containers needed)
    ocean_shipping_cost = sum(C["C"] * z[t] for t in range(T))
    
    return purchasing_cost + air_shipping_cost + ocean_shipping_cost

def save_results_to_excel(x, inventory, z, C, N, T, J, total_cost, V, output_file="heuristic_results.xlsx"):
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Create summary sheet
        summary_data = {
            'Metric': ['Total Cost', 'Number of Products', 'Number of Periods'],
            'Value': [total_cost, N, T]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Create orders sheet
        orders_data = []
        for t in range(T):
            for i in range(N):
                for j in range(J):
                    if x[i, j, t] > 0:
                        shipping_method = "Ocean" if j == 0 else "Air"
                        orders_data.append({
                            'Product': i + 1,
                            'Order Period': t + 1,
                            'Shipping Method': shipping_method,
                            'Order Quantity': x[i, j, t],
                            'Total Volume': x[i, j, t] * V[i]
                        })
        pd.DataFrame(orders_data).to_excel(writer, sheet_name='Orders', index=False)
        
        # Create inventory sheet
        inventory_data = []
        for t in range(T):
            for i in range(N):
                inventory_data.append({
                    'Product': i + 1,
                    'Period': t + 1,
                    'Ending Inventory': inventory[i, t]
                })
        pd.DataFrame(inventory_data).to_excel(writer, sheet_name='Inventory', index=False)
        
        # Create containers sheet
        containers_data = []
        for t in range(T):
            if z[t] > 0:
                containers_data.append({
                    'Period': t + 1,
                    'Number of Containers': z[t],
                    'Total Cost': C["C"] * z[t]
                })
        pd.DataFrame(containers_data).to_excel(writer, sheet_name='Containers', index=False)

def main():
    # Read data
    file_path = "OR113-2_midtermProject_data.xlsx"
    N, T, J, D, I_0, C, V, V_C, lead_times = read_data(file_path)
    
    # Get heuristic solution
    x, inventory, z = heuristic_solution(N, T, J, D, I_0, C, V, V_C, lead_times)
    
    # Calculate total cost
    total_cost = calculate_total_cost(x, z, C, N, T, J)
    
    # Print results to console
    print("\nSimplified Just-in-Time Solution Results:")
    print(f"Total Cost: {total_cost:.2f}")
    
    # Save results to Excel
    save_results_to_excel(x, inventory, z, C, N, T, J, total_cost, V)
    print(f"\nResults have been saved to 'heuristic_results.xlsx'")

if __name__ == "__main__":
    main() 