import pandas as pd
import numpy as np
import os
from datetime import datetime

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
    J = 3  # ocean (j=0), air (j=1), and express (j=2)
    df_inventory_cost = pd.read_excel(file_path, sheet_name="Inventory cost")
    
    print("\nDataFrame shapes:")
    print(f"Shipping cost shape: {df_shipping_cost.shape}")
    print(f"Inventory cost shape: {df_inventory_cost.shape}")
    print("\nInventory Cost Data:")
    print(df_inventory_cost)
    
    # Initialize cost parameters
    C = {
        "P": np.zeros([N]),  # Purchasing cost
        "V": np.zeros([N, J]),  # Variable shipping cost
        "C": 2750,  # Container cost
    }
    V = np.zeros([N])  # Volume per unit
    V_C = 30  # Container volume capacity
    
    # Read costs and volumes
    for i in range(N):
        try:
            C["P"][i] = float(df_inventory_cost.iloc[i, 2])  # Unit cost from inventory cost sheet
            V[i] = float(df_shipping_cost.iloc[i, 3])  # Volume from shipping cost sheet
            C["V"][i, 0] = (V[i] / V_C) * C["C"]  # Ocean cost per unit based on volume ratio
            C["V"][i, 1] = float(df_shipping_cost.iloc[i, 2])  # Air freight cost
            C["V"][i, 2] = float(df_shipping_cost.iloc[i, 1])  # Express shipping cost
            
            print(f"Product {i+1}:")
            print(f"  Purchasing cost: {C['P'][i]}")
            print(f"  Volume: {V[i]}")
            print(f"  Ocean freight cost per unit: {C['V'][i, 0]}")
            print(f"  Air freight cost: {C['V'][i, 1]}")
            print(f"  Express shipping cost: {C['V'][i, 2]}")
        except Exception as e:
            print(f"Error reading data for product {i+1}: {e}")
            print(f"Current row in inventory cost data: {df_inventory_cost.iloc[i]}")
            print(f"Current row in shipping cost data: {df_shipping_cost.iloc[i]}")
            raise
    
    lead_times = {"ocean": 3, "air": 2, "express": 1}  # Lead times in months
    
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
            march_value = df_transit.iloc[i+1, 1] if pd.notna(df_transit.iloc[i+1, 1]) else 0
            april_value = df_transit.iloc[i+1, 2] if pd.notna(df_transit.iloc[i+1, 2]) else 0
            
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
    x = np.zeros((N, J, T))  # Order quantities (J=3: ocean, air, express)
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
            # Calculate current inventory before demand
            if t == 0:
                current_inventory = I_0[i]
            else:
                current_inventory = inventory[i, t-1]
            
            # Calculate what will arrive this period from previous orders
            arriving_ocean = 0
            arriving_air = 0
            arriving_express = 0
            
            # Check for ocean shipments arriving (ordered 3 periods ago)
            if t >= lead_times["ocean"]:
                arriving_ocean = x[i, 0, t - lead_times["ocean"]]
            
            # Check for air shipments arriving (ordered 1 period ago)
            if t >= lead_times["air"]:
                arriving_air = x[i, 1, t - lead_times["air"]]
            if t >= lead_times["express"]:
                arriving_express = x[i, 2, t - lead_times["express"]]
            
            # Add in-transit inventory for March (t=0) and April (t=1)
            in_transit_arriving = 0
            if t == 0:  # March
                in_transit_arriving = in_transit["march"][i]
            elif t == 1:  # April
                in_transit_arriving = in_transit["april"][i]
            
            # Add all arriving shipments to current inventory
            current_inventory += arriving_ocean + arriving_air + arriving_express + in_transit_arriving
            
            print(f"\nPeriod {t+1}:")
            print(f"Starting inventory: {current_inventory}")
            print(f"Arriving ocean: {arriving_ocean}")
            print(f"Arriving air: {arriving_air}")
            print(f"Arriving express: {arriving_express}")
            print(f"In-transit arriving: {in_transit_arriving}")
            print(f"Total inventory before demand: {current_inventory}")
            print(f"Demand: {D[i, t]}")
            
            # Calculate required quantity
            required_qty = D[i, t]
            
            # Calculate how much we need to order
            shortage = max(0, required_qty - current_inventory)

            # Calculate order costs for all methods
            ocean_cost = C["V"][i, 0] * shortage
            air_cost = C["V"][i, 1] * shortage
            express_cost = C["V"][i, 2] * shortage
            
            if shortage > 0:
                print(f"Need to order: {shortage}")
                
                print(f"Ocean cost: {ocean_cost:.2f}")
                print(f"Air cost: {air_cost:.2f}")
                print(f"Express cost: {express_cost:.2f}")
                
                # Determine when we need to place the order
                ocean_order_time = max(0, t - lead_times["ocean"])
                air_order_time = max(0, t - lead_times["air"])
                express_order_time = max(0, t - lead_times["express"])
                
                # Choose shipping method based on period requirements
                if t == 1:  # For period 2 demand
                    # Must use express shipping
                    x[i, 2, express_order_time] += shortage
                    print(f"Period 2 demand: Must use express shipping, order in period {express_order_time + 1}")
                elif t == 2:  # For period 3 demand
                    # Must use air shipping
                    x[i, 1, air_order_time] += shortage
                    print(f"Period 3 demand: Must use air shipping, order in period {air_order_time + 1}")
                else:  # For other periods
                    if ocean_cost <= air_cost and ocean_cost <= express_cost:
                        x[i, 0, ocean_order_time] += shortage
                        print(f"Using ocean shipping, order in period {ocean_order_time + 1}")
                    elif air_cost <= express_cost:
                        x[i, 1, air_order_time] += shortage
                        print(f"Using air shipping, order in period {air_order_time + 1}")
                    else:
                        x[i, 2, express_order_time] += shortage
                        print(f"Using express shipping, order in period {express_order_time + 1}")
            
            # Calculate ending inventory (after demand)
            inventory[i, t] = max(0, current_inventory - required_qty)
            
            
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

def calculate_total_cost(x, z, C, N, T, J, inventory, df_inventory_cost, lead_times):
    # Calculate purchasing cost for each product
    purchasing_costs = np.zeros(N)
    for i in range(N):
        total_quantity = sum(x[i, j, t] for j in range(J) for t in range(T))
        purchasing_costs[i] = C["P"][i] * total_quantity
        print(f"\nProduct {i+1} cost calculation:")
        print(f"  Unit cost: {C['P'][i]}")
        print(f"  Total quantity ordered: {total_quantity}")
        print(f"  Total purchasing cost: {purchasing_costs[i]}")
    
    total_purchasing_cost = sum(purchasing_costs)
    print(f"\nTotal purchasing cost: {total_purchasing_cost}")
    
    # Calculate shipping costs
    air_shipping_cost = 0
    express_shipping_cost = 0
    for i in range(N):
        for t in range(T):
            air_shipping_cost += C["V"][i, 1] * x[i, 1, t]  # Regular air freight
            express_shipping_cost += C["V"][i, 2] * x[i, 2, t]  # Express shipping
    
    print(f"Total air shipping cost: {air_shipping_cost}")
    print(f"Total express shipping cost: {express_shipping_cost}")
    
    # Calculate ocean shipping cost (based on actual containers needed)
    ocean_shipping_cost = sum(C["C"] * z[t] for t in range(T))
    print(f"Total ocean shipping cost: {ocean_shipping_cost}")
    
    # Calculate holding cost for each product and period
    holding_costs = np.zeros(N)
    for i in range(N):
        holding_cost_rate = float(df_inventory_cost.iloc[i, 3])  # Get holding cost rate from inventory cost sheet
        print(f"Holding cost rate for product {i+1}: {holding_cost_rate}")
        # For each period, calculate holding cost for inventory excluding in-transit
        for t in range(T):
            # Calculate actual inventory excluding in-transit
            period_inventory = inventory[i, t]  # Base ending inventory
            
            arriving_ocean = 0
            arriving_air = 0
            arriving_express = 0
            
            # Calculate arriving quantities in this period and add one unit of holding cost for each
            if t >= lead_times["ocean"]:
                arriving_ocean = x[i, 0, t - lead_times["ocean"]]
                # Add one unit of holding cost for ocean shipments arriving
                holding_costs[i] += arriving_ocean * holding_cost_rate
            
                
            if t >= lead_times["air"]:
                arriving_air = x[i, 1, t - lead_times["air"]]
                holding_costs[i] += arriving_air * holding_cost_rate
            

            if t >= lead_times["express"]:
                arriving_express = x[i, 2, t - lead_times["express"]]
                # Add one unit of holding cost for air and express shipments arriving
                holding_costs[i] += arriving_express * holding_cost_rate
           
            
            # Calculate holding cost for this period's ending inventory
            period_cost = holding_cost_rate * period_inventory
            holding_costs[i] += period_cost
            
            print(f"\nProduct {i+1}, Period {t+1} holding cost calculation:")
            print(f"  Base ending inventory: {period_inventory}")
            print(f"  Arriving ocean: {arriving_ocean}")
            print(f"  Arriving air: {arriving_air}")
            print(f"  Arriving express: {arriving_express}")
            print(f"  Holding cost for arrivals: {(arriving_ocean + arriving_air + arriving_express) * holding_cost_rate}")
            print(f"  Period holding cost for inventory: {period_cost}")
            print(f"  Total period holding cost: {period_cost + (arriving_ocean + arriving_air + arriving_express) * holding_cost_rate}")
    
    total_holding_cost = sum(holding_costs)
    print(f"\nTotal holding cost: {total_holding_cost}")
    
    # Calculate fixed costs (setup costs) for each order
    fixed_costs = np.zeros(N)
    ocean_fixed_cost = 0
    for t in range(T):
        # For ocean shipping, check if any product uses it in this period
        if any(x[i, 0, t] > 0 for i in range(N)):
            ocean_fixed_cost += 50  # $50 per period where ocean shipping is used
    
    for i in range(N):
        # Fixed costs for express and air shipping methods (per product)
        express_orders = sum(1 for t in range(T) if x[i, 2, t] > 0) * 100  # $100 per express order
        air_orders = sum(1 for t in range(T) if x[i, 1, t] > 0) * 80      # $80 per air freight order
        fixed_costs[i] = express_orders + air_orders
        print(f"\nProduct {i+1} fixed cost:")
        print(f"  Express orders: {sum(1 for t in range(T) if x[i, 2, t] > 0)} × $100")
        print(f"  Air freight orders: {sum(1 for t in range(T) if x[i, 1, t] > 0)} × $80")
        print(f"  Total fixed cost for product: {fixed_costs[i]}")
    
    total_fixed_cost = sum(fixed_costs) + ocean_fixed_cost
    print(f"\nOcean shipping fixed cost (across all products): {ocean_fixed_cost}")
    print(f"Total fixed cost: {total_fixed_cost}")
    
    # Calculate total cost
    total_cost = (total_purchasing_cost + 
                 air_shipping_cost + 
                 express_shipping_cost + 
                 ocean_shipping_cost + 
                 total_holding_cost + 
                 total_fixed_cost)
    
    print("\nCost Summary:")
    print(f"Purchasing Cost: {total_purchasing_cost:.2f}")
    print(f"Air Shipping Cost: {air_shipping_cost:.2f}")
    print(f"Express Shipping Cost: {express_shipping_cost:.2f}")
    print(f"Ocean Shipping Cost: {ocean_shipping_cost:.2f}")
    print(f"Holding Cost: {total_holding_cost:.2f}")
    print(f"Fixed Cost: {total_fixed_cost:.2f}")
    print(f"Total Cost: {total_cost:.2f}")
    
    return total_cost, purchasing_costs, air_shipping_cost, express_shipping_cost, ocean_shipping_cost, total_holding_cost, total_fixed_cost

def save_results_to_excel(x, inventory, z, C, N, T, J, total_cost, purchasing_costs, air_cost, express_cost, ocean_cost, holding_cost, fixed_cost, V, lead_times, output_file="heuristic_results.xlsx"):
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename with timestamp
    filename = f"heuristic_results_{timestamp}.xlsx"
    output_path = os.path.join(results_dir, filename)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Calculate total quantities by product and shipping method
        total_ocean_qty = np.array([sum(x[i, 0, t] for t in range(T)) for i in range(N)])
        total_air_qty = np.array([sum(x[i, 1, t] for t in range(T)) for i in range(N)])
        total_express_qty = np.array([sum(x[i, 2, t] for t in range(T)) for i in range(N)])
        total_qty = total_ocean_qty + total_air_qty + total_express_qty

        # Create summary sheet with detailed costs and quantities
        summary_data = {
            'Metric': [
                'Total Cost',
                'Total Purchasing Cost',
                'Total Air Shipping Cost',
                'Total Express Shipping Cost',
                'Total Ocean Shipping Cost',
                'Total Holding Cost',
                'Total Fixed Cost',
                'Number of Products',
                'Number of Periods',
                'Total Quantity Ordered (All Products)',
                'Total Ocean Shipping Quantity',
                'Total Air Shipping Quantity',
                'Total Express Shipping Quantity'
            ],
            'Value': [
                total_cost,
                sum(purchasing_costs),
                air_cost,
                express_cost,
                ocean_cost,
                holding_cost,
                fixed_cost,
                N,
                T,
                sum(total_qty),
                sum(total_ocean_qty),
                sum(total_air_qty),
                sum(total_express_qty)
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Create quantities summary sheet
        quantities_data = {
            'Product': list(range(1, N + 1)),
            'Total Quantity': total_qty,
            'Ocean Shipping Quantity': total_ocean_qty,
            'Air Shipping Quantity': total_air_qty,
            'Express Shipping Quantity': total_express_qty,
            'Unit Cost': [C["P"][i] for i in range(N)],
            'Total Purchasing Cost': purchasing_costs,
            'Total Volume': [V[i] * total_qty[i] for i in range(N)],
            'Ocean Volume': [V[i] * total_ocean_qty[i] for i in range(N)],
            'Air Volume': [V[i] * total_air_qty[i] for i in range(N)],
            'Express Volume': [V[i] * total_express_qty[i] for i in range(N)]
        }
        pd.DataFrame(quantities_data).to_excel(writer, sheet_name='Quantities Summary', index=False)
        
        # Create purchasing costs sheet
        purchasing_data = {
            'Product': list(range(1, N + 1)),
            'Unit Cost': [C["P"][i] for i in range(N)],
            'Total Quantity': total_qty,
            'Ocean Quantity': total_ocean_qty,
            'Air Quantity': total_air_qty,
            'Express Quantity': total_express_qty,
            'Total Purchasing Cost': purchasing_costs,
            'Total Holding Cost': [holding_cost / N for _ in range(N)],  # Approximate per-product holding cost
            'Total Fixed Cost': [fixed_cost / N for _ in range(N)]  # Approximate per-product fixed cost
        }
        pd.DataFrame(purchasing_data).to_excel(writer, sheet_name='Purchasing Costs', index=False)
        
        # Create detailed orders sheet
        orders_data = []
        for t in range(T):
            for i in range(N):
                for j in range(J):
                    if x[i, j, t] > 0:
                        shipping_method = "Ocean" if j == 0 else "Air" if j == 1 else "Express"
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
    
    print(f"\nResults have been saved to '{output_path}'")

def main():
    # Read data
    file_path = "OR113-2_midtermProject_data.xlsx"
    N, T, J, D, I_0, C, V, V_C, lead_times, in_transit = read_data(file_path)
    
    # Get heuristic solution
    x, inventory, z = heuristic_solution(N, T, J, D, I_0, C, V, V_C, lead_times, in_transit)
    
    # Read inventory cost data for holding and fixed costs
    df_inventory_cost = pd.read_excel(file_path, sheet_name="Inventory cost")
    
    # Calculate total cost and detailed costs
    total_cost, purchasing_costs, air_cost, express_cost, ocean_cost, holding_cost, fixed_cost = calculate_total_cost(
        x, z, C, N, T, J, inventory, df_inventory_cost, lead_times)
    
    # Print results to console
    print("\nSimplified Just-in-Time Solution Results:")
    print(f"Total Cost: {total_cost:.2f}")
    print(f"Total Purchasing Cost: {sum(purchasing_costs):.2f}")
    print(f"Total Air Shipping Cost: {air_cost:.2f}")
    print(f"Total Express Shipping Cost: {express_cost:.2f}")
    print(f"Total Ocean Shipping Cost: {ocean_cost:.2f}")
    print(f"Total Holding Cost: {holding_cost:.2f}")
    print(f"Total Fixed Cost: {fixed_cost:.2f}")
    
    # Save results to Excel
    save_results_to_excel(x, inventory, z, C, N, T, J, total_cost, purchasing_costs, 
                         air_cost, express_cost, ocean_cost, holding_cost, fixed_cost, V, lead_times)

if __name__ == "__main__":
    main() 
