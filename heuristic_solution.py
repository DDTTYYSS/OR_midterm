import pandas as pd
import numpy as np

def read_data(file_path):
    # Read the Excel file from the 'Demand' sheet
    df_demand = pd.read_excel(file_path, sheet_name="Demand")
    N = df_demand.shape[0] - 1  # Number of products
    T = df_demand.shape[1] - 2  # Number of periods
    
    print(f"\nData dimensions:")
    print(f"Number of products (N): {N}")
    print(f"Number of periods (T): {T}")
    print("\nDemand Data:")
    print(df_demand)
    
    # Initialize arrays
    D = np.zeros([N, T])
    I_0 = np.zeros([N])
    
    # Read demand data
    for i in range(N):
        I_0[i] = df_demand.iloc[i+1, 1]  # Initial inventory is in column 1
        for t in range(T):
            D[i, t] = df_demand.iloc[i+1, t+2]  # Demand starts from column 2
        print(f"\nProduct {i+1}:")
        print(f"  Initial inventory: {I_0[i]}")
        print(f"  Demand: {D[i]}")
    
    # Read shipping cost data
    df_shipping_cost = pd.read_excel(file_path, sheet_name="Shipping cost")
    J = 2  # Only considering ocean (j=0) and air (j=1) freight
    df_inventory_cost = pd.read_excel(file_path, sheet_name="Inventory cost")
    
    print("\nDataFrame shapes:")
    print(f"Shipping cost shape: {df_shipping_cost.shape}")
    print(f"Inventory cost shape: {df_inventory_cost.shape}")
    print("\nInventory Cost Data:")
    print(df_inventory_cost)
    
    # Initialize cost parameters
    C = {
        "P": np.zeros([N]),  # Purchasing cost
        "V": np.zeros([N, J]),  # Variable shipping cost (only used for air freight)
        "C": 2750,  # Container cost
    }
    V = np.zeros([N])  # Volume per unit
    V_C = 30  # Container volume capacity
    
    # Read costs and volumes
    for i in range(N):
        try:
            # Note: using i instead of i+1 since we're reading from row i in the cost data
            C["P"][i] = float(df_inventory_cost.iloc[i, 2])  # Unit cost from inventory cost sheet
            V[i] = float(df_shipping_cost.iloc[i, 3])  # Volume from shipping cost sheet
            C["V"][i, 1] = float(df_shipping_cost.iloc[i, 2])  # Air freight cost
            C["V"][i, 0] = (V[i] / V_C) * C["C"]  # Ocean cost per unit based on volume ratio
            
            print(f"Product {i+1}:")
            print(f"  Purchasing cost: {C['P'][i]}")
            print(f"  Volume: {V[i]}")
            print(f"  Air freight cost: {C['V'][i, 1]}")
            print(f"  Ocean freight cost per unit: {C['V'][i, 0]}")
        except Exception as e:
            print(f"Error reading data for product {i+1}: {e}")
            print(f"Current row in inventory cost data: {df_inventory_cost.iloc[i]}")
            print(f"Current row in shipping cost data: {df_shipping_cost.iloc[i]}")
            raise
    
    lead_times = {"ocean": 3, "air": 1}  # Lead times in months
    
    # Read in-transit inventory data
    df_transit = pd.read_excel(file_path, sheet_name="In-transit")
    print(f"\nIn-transit data shape: {df_transit.shape}")
    print("\nIn-transit Data:")
    print(df_transit)
    
    in_transit = {
        "march": np.zeros([N]),  # Arriving in March
        "april": np.zeros([N])   # Arriving in April
    }
    
    # Read in-transit amounts for each product
    for i in range(N):
        try:
            # Skip the header row by using i+1
            march_value = df_transit.iloc[i+1, 1] if pd.notna(df_transit.iloc[i+1, 1]) else 0
            april_value = df_transit.iloc[i+1, 2] if pd.notna(df_transit.iloc[i+1, 2]) else 0
            
            # Convert to float, handling any non-numeric values
            try:
                in_transit["march"][i] = float(march_value)
            except (ValueError, TypeError):
                print(f"Warning: Non-numeric value '{march_value}' for March in-transit, product {i+1}. Using 0.")
                in_transit["march"][i] = 0
                
            try:
                in_transit["april"][i] = float(april_value)
            except (ValueError, TypeError):
                print(f"Warning: Non-numeric value '{april_value}' for April in-transit, product {i+1}. Using 0.")
                in_transit["april"][i] = 0
            
            print(f"Product {i+1} in-transit: March={in_transit['march'][i]}, April={in_transit['april'][i]}")
        except Exception as e:
            print(f"Error reading in-transit data for product {i+1}: {e}")
            print(f"Current row in transit data: {df_transit.iloc[i+1]}")
            raise
    
    # Print final data summary
    print("\nFinal Data Summary:")
    for i in range(N):
        print(f"\nProduct {i+1}:")
        print(f"  Initial inventory: {I_0[i]}")
        print(f"  Demand: {D[i]}")
        print(f"  Purchase cost: {C['P'][i]}")
        print(f"  Volume: {V[i]}")
        print(f"  Air cost: {C['V'][i, 1]}")
        print(f"  Ocean cost per unit: {C['V'][i, 0]}")
    
    return N, T, J, D, I_0, C, V, V_C, lead_times, in_transit

def heuristic_solution(N, T, J, D, I_0, C, V, V_C, lead_times, in_transit):
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
                
                # Choose shipping method based on period requirements
                if t == 1:  # For period 2 demand
                    # Must use express (air) shipping
                    x[i, 1, air_order_time] += order_qty
                    print(f"Period 2 demand: Must use express shipping, order in period {air_order_time + 1}")
                elif t == 2:  # For period 3 demand
                    # Must use air shipping
                    x[i, 1, air_order_time] += order_qty
                    print(f"Period 3 demand: Must use air shipping, order in period {air_order_time + 1}")
                else:  # For other periods
                    if t >= lead_times["ocean"]:  # Can use either method if timing allows
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
            # Add in-transit inventory for March (t=2) and April (t=3)
            in_transit_arriving = 0
            if t == 2:  # March
                in_transit_arriving = in_transit["march"][i]
            elif t == 3:  # April
                in_transit_arriving = in_transit["april"][i]
            
            inventory[i, t] = max(0, current_inventory - D[i, t] + in_transit_arriving)  # Add in-transit to ending inventory
            print(f"Ending inventory for period {t+1}: {inventory[i, t]}")
            if in_transit_arriving > 0:
                print(f"Including in-transit arrival for period {t+1}: {in_transit_arriving}")
    
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
    # Calculate purchasing cost for each product
    purchasing_costs = np.zeros(N)
    for i in range(N):
        total_quantity = sum(x[i, j, t] for j in range(J) for t in range(T))
        purchasing_costs[i] = C["P"][i] * total_quantity
        print(f"\nProduct {i+1} cost calculation:")
        print(f"Unit cost: {C['P'][i]}")
        print(f"Total quantity ordered: {total_quantity}")
        print(f"Total purchasing cost: {purchasing_costs[i]}")
    
    total_purchasing_cost = sum(purchasing_costs)
    print(f"\nTotal purchasing cost: {total_purchasing_cost}")
    
    # Calculate air shipping cost
    air_shipping_cost = sum(C["V"][i, 1] * x[i, 1, t]
                           for i in range(N) for t in range(T))
    
    # Calculate ocean shipping cost (based on actual containers needed)
    ocean_shipping_cost = sum(C["C"] * z[t] for t in range(T))
    
    total_cost = total_purchasing_cost + air_shipping_cost + ocean_shipping_cost
    
    return total_cost, purchasing_costs, air_shipping_cost, ocean_shipping_cost

def save_results_to_excel(x, inventory, z, C, N, T, J, total_cost, purchasing_costs, air_cost, ocean_cost, V, output_file="heuristic_results.xlsx"):
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Create summary sheet with detailed costs
        summary_data = {
            'Metric': [
                'Total Cost',
                'Total Purchasing Cost',
                'Total Air Shipping Cost',
                'Total Ocean Shipping Cost',
                'Number of Products',
                'Number of Periods'
            ],
            'Value': [
                total_cost,
                sum(purchasing_costs),
                air_cost,
                ocean_cost,
                N,
                T
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Create purchasing costs sheet
        purchasing_data = {
            'Product': list(range(1, N + 1)),
            'Unit Cost': [C["P"][i] for i in range(N)],
            'Total Quantity': [sum(x[i, j, t] for j in range(J) for t in range(T)) for i in range(N)],
            'Total Purchasing Cost': purchasing_costs
        }
        pd.DataFrame(purchasing_data).to_excel(writer, sheet_name='Purchasing Costs', index=False)
        
        # Create orders sheet
        orders_data = []
        for t in range(T):
            for i in range(N):
                for j in range(J):
                    if x[i, j, t] > 0:
                        shipping_method = "Ocean" if j == 0 else "Air"
                        shipping_cost = C["V"][i, j] * x[i, j, t] if j == 1 else 0  # Only for air shipping
                        orders_data.append({
                            'Product': i + 1,
                            'Order Period': t + 1,
                            'Shipping Method': shipping_method,
                            'Order Quantity': x[i, j, t],
                            'Total Volume': x[i, j, t] * V[i],
                            'Unit Cost': C["P"][i],
                            'Purchasing Cost': C["P"][i] * x[i, j, t],
                            'Shipping Cost': shipping_cost
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
    N, T, J, D, I_0, C, V, V_C, lead_times, in_transit = read_data(file_path)
    
    # Get heuristic solution
    x, inventory, z = heuristic_solution(N, T, J, D, I_0, C, V, V_C, lead_times, in_transit)
    
    # Calculate total cost and detailed costs
    total_cost, purchasing_costs, air_cost, ocean_cost = calculate_total_cost(x, z, C, N, T, J)
    
    # Print results to console
    print("\nSimplified Just-in-Time Solution Results:")
    print(f"Total Cost: {total_cost:.2f}")
    print(f"Total Purchasing Cost: {sum(purchasing_costs):.2f}")
    print(f"Total Air Shipping Cost: {air_cost:.2f}")
    print(f"Total Ocean Shipping Cost: {ocean_cost:.2f}")
    
    # Save results to Excel
    save_results_to_excel(x, inventory, z, C, N, T, J, total_cost, purchasing_costs, air_cost, ocean_cost, V)
    print(f"\nResults have been saved to 'heuristic_results.xlsx'")

if __name__ == "__main__":
    main() 