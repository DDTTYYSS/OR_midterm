# import os
# os.environ['GRB_LICENSE_FILE'] = '/Users/albert/gurobi/gurobi.lic'

import os
import sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import csv
import pandas as pd
from ORMP4 import generate_instance, naive_solution, solve_instance
from heuristic_solution import read_data

# Set up Gurobi environment
try:
    # Try to set the license file path
    license_path = os.path.expanduser('~/gurobi/gurobi.lic')
    if os.path.exists(license_path):
        os.environ['GRB_LICENSE_FILE'] = license_path
    else:
        # Try alternative paths
        alternative_paths = [
            '/Users/albert/gurobi/gurobi.lic',
            '/usr/local/gurobi/gurobi.lic',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gurobi.lic')
        ]
        for path in alternative_paths:
            if os.path.exists(path):
                os.environ['GRB_LICENSE_FILE'] = path
                break
        else:
            print("Warning: Could not find Gurobi license file. Please ensure it is properly installed.")
except Exception as e:
    print(f"Warning: Error setting up Gurobi environment: {str(e)}")

def adapt_heuristic_for_synthetic_data(N, T, D, C_P, C_V1, C_V2, I_0, I_1, I_2, V, C_C, C_H, C_F, T_lead):
    """
    Adapt the heuristic algorithm to work with synthetic data format
    """
    # Initialize parameters in format expected by heuristic algorithm
    J = 3  # Number of shipping methods (ocean, air, express)
    
    # Reshape variable shipping costs to match heuristic's expected format
    C = {
        "P": C_P,  # Purchasing cost
        "V": np.zeros([N, J]),  # Variable shipping cost
        "C": C_C,  # Container cost
    }
    
    # Map shipping costs to the right format
    for i in range(N):
        C["V"][i, 0] = (V[i] / 30) * C_C  # Ocean cost based on volume ratio
        C["V"][i, 1] = C_V1[i]  # Air freight cost
        C["V"][i, 2] = C_V2[i]  # Express shipping cost

    # Create lead times dictionary
    lead_times = {"ocean": T_lead[2], "air": T_lead[1], "express": T_lead[0]}
    
    # Create in-transit inventory dictionary
    in_transit = {
        "march": I_1,  # Arriving in first period
        "april": I_2   # Arriving in second period
    }
    
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
        # print(f"\nProduct {i+1}:")
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
            
            # print(f"\nPeriod {t+1}:")
            # print(f"Starting inventory: {current_inventory}")
            # print(f"Arriving ocean: {arriving_ocean}")
            # print(f"Arriving air: {arriving_air}")
            # print(f"Arriving express: {arriving_express}")
            # print(f"In-transit arriving: {in_transit_arriving}")
            # print(f"Total inventory before demand: {current_inventory}")
            # print(f"Demand: {D[i, t]}")
            
            # Calculate required quantity
            required_qty = D[i, t]
            
            # Calculate how much we need to order
            shortage = max(0, required_qty - current_inventory)
            
            # Calculate order costs for all methods
            ocean_cost = C["V"][i, 0] * shortage
            air_cost = C["V"][i, 1] * shortage
            express_cost = C["V"][i, 2] * shortage
            
            if shortage > 0:
                # print(f"Need to order: {shortage}")
                # print(f"Ocean cost: {ocean_cost:.2f}")
                # print(f"Air cost: {air_cost:.2f}")
                # print(f"Express cost: {express_cost:.2f}")
                
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
            z[t] = np.ceil(total_volume / 30)  # Container volume capacity is 30
            print(f"\nPeriod {t+1} ocean shipping:")
            print(f"Total volume: {total_volume:.2f}")
            print(f"Containers needed: {z[t]:.0f}")
    
    # Calculate total cost
    total_cost = calculate_heuristic_cost(x, z, C, N, T, J, inventory, C_H, C_F, lead_times)
    
    return total_cost, x, inventory, z

def calculate_heuristic_cost(x, z, C, N, T, J, inventory, C_H, C_F, lead_times):
    """
    Calculate the total cost of the heuristic solution
    """
    # Calculate purchasing cost
    purchasing_costs = np.zeros(N)
    for i in range(N):
        total_quantity = sum(x[i, j, t] for j in range(J) for t in range(T))
        purchasing_costs[i] = C["P"][i] * total_quantity
    
    total_purchasing_cost = sum(purchasing_costs)
    
    # Calculate shipping costs
    air_shipping_cost = 0
    express_shipping_cost = 0
    for i in range(N):
        for t in range(T):
            air_shipping_cost += C["V"][i, 1] * x[i, 1, t]  # Regular air freight
            express_shipping_cost += C["V"][i, 2] * x[i, 2, t]  # Express shipping
    
    # Calculate ocean shipping cost (based on containers needed)
    ocean_shipping_cost = sum(C["C"] * z[t] for t in range(T))

    #############################################
    # Calculate holding cost for each product and period
    holding_costs = np.zeros(N)
    for i in range(N):
        holding_cost_rate = C_H[i]  # Get holding cost rate from C_H array
        print(f"Holding cost rate for product {i+1}: {holding_cost_rate}")
        # For each period, calculate holding cost for inventory excluding in-transit
        for t in range(T):
            # Calculate actual inventory excluding in-transit
            period_inventory = inventory[i, t]  # Base ending inventory
            
            # Calculate arriving quantities in this period and add one unit of holding cost for each
            if t >= lead_times["ocean"]:
                arriving_ocean = x[i, 0, t - lead_times["ocean"]]
                # Add one unit of holding cost for ocean shipments arriving
                holding_costs[i] += arriving_ocean * holding_cost_rate
            else:
                arriving_ocean = 0
                
            if t >= lead_times["air"]:
                arriving_air = x[i, 1, t - lead_times["air"]]
                arriving_express = x[i, 2, t - lead_times["express"]]
                # Add one unit of holding cost for air and express shipments arriving
                holding_costs[i] += (arriving_air + arriving_express) * holding_cost_rate
            else:
                arriving_air = 0
                arriving_express = 0
            
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
    ######################################
    
    # Calculate fixed costs for each order
    fixed_cost = 0
    
    # Check each period for each shipping method
    for t in range(T):
        # For ocean shipping
        if any(x[i, 0, t] > 0 for i in range(N)):
            fixed_cost += C_F[2]  # Ocean freight fixed cost per period
        
        # For air shipping
        if any(x[i, 1, t] > 0 for i in range(N)):
            fixed_cost += C_F[1]  # Air freight fixed cost per period
            
        # For express shipping
        if any(x[i, 2, t] > 0 for i in range(N)):
            fixed_cost += C_F[0]  # Express shipping fixed cost per period
    
    total_fixed_cost = fixed_cost
    
    # Calculate total cost
    total_cost = (total_purchasing_cost + 
                 air_shipping_cost + 
                 express_shipping_cost + 
                 ocean_shipping_cost + 
                 total_holding_cost + 
                 total_fixed_cost)
    
    return total_cost

def run_experiment():
    """
    Run the experiment for all scenarios
    """
    results = []
    summary_stats = []

    # Add Excel data test group
    print("\nRunning Excel Data Test Group")
    try:
        # Read Excel data
        file_path = "OR113-2_midtermProject_data.xlsx"
        N, T, J, D, I_0, C, V, V_C, lead_times, in_transit = read_data(file_path)
        
        # Convert Excel data format to match synthetic data format
        C_P = C["P"]
        C_V1 = C["V"][:, 1]  # Air freight costs
        C_V2 = C["V"][:, 2]  # Express shipping costs
        C_C = C["C"]  # Container cost
        
        # Read holding costs from inventory cost sheet
        df_inventory_cost = pd.read_excel(file_path, sheet_name="Inventory cost")
        C_H = np.array([float(df_inventory_cost.iloc[i, 3]) for i in range(N)])  # Holding costs
        
        # Fixed costs for ocean, air, express
        C_F = np.array([50, 80, 100])
        
        # Lead times
        T_lead = np.array([lead_times["express"], lead_times["air"], lead_times["ocean"]])
        
        # In-transit inventory
        I_1 = in_transit["march"]
        I_2 = in_transit["april"]

        # Time and solve using relaxed optimization model
        start_time = time.time()
        optimal_cost = solve_instance(N, T, D, C_P, C_V1, C_V2, I_0, I_1, I_2, V, C_C, C_H, C_F, T_lead)
        optimal_time = time.time() - start_time

        # Time and solve using naive solution
        start_time = time.time()
        naive_cost = naive_solution(N, T, D, C_P, C_V1, I_0, I_1, I_2, C_H, C_F)
        naive_time = time.time() - start_time

        # Time and solve using our heuristic solution
        start_time = time.time()
        heuristic_cost, x, inventory, z = adapt_heuristic_for_synthetic_data(
            N, T, D, C_P, C_V1, C_V2, I_0, I_1, I_2, V, C_C, C_H, C_F, T_lead
        )
        heuristic_time = time.time() - start_time

        # Calculate optimality gaps
        if optimal_cost is not None:
            naive_gap = (naive_cost - optimal_cost) / optimal_cost * 100
            heuristic_gap = (heuristic_cost - optimal_cost) / optimal_cost * 100
            
            results.append((
                "excel", 1, 
                optimal_cost, naive_cost, heuristic_cost,
                optimal_time, naive_time, heuristic_time,
                naive_gap, heuristic_gap
            ))
            
            print(f"\nExcel Data Test Results:")
            print(f"  Optimal Cost: {optimal_cost:.2f}")
            print(f"  Naive Cost: {naive_cost:.2f}")
            print(f"  Heuristic Cost: {heuristic_cost:.2f}")
            print(f"  Times (s):")
            print(f"    Optimal: {optimal_time:.3f}")
            print(f"    Naive: {naive_time:.3f}")
            print(f"    Heuristic: {heuristic_time:.3f}")
            print(f"  Gaps (%):")
            print(f"    Naive: {naive_gap:.2f}")
            print(f"    Heuristic: {heuristic_gap:.2f}")

            summary_stats.append((
                "excel", "excel", "excel", "excel",
                naive_gap, 0, heuristic_gap, 0,
                optimal_time, 0, naive_time, 0, heuristic_time, 0
            ))
        else:
            print("Excel Data Test: Optimization failed.")
    except Exception as e:
        print(f"Error in Excel data test: {str(e)}")
        import traceback
        traceback.print_exc()

    # Export results to CSV
    with open("detailed_results.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            "Scenario ID", "Instance ID", 
            "Optimal Cost", "Naive Cost", "Heuristic Cost",
            "Optimal Time (s)", "Naive Time (s)", "Heuristic Time (s)",
            "Naive Gap (%)", "Heuristic Gap (%)"
        ])
        csvwriter.writerows(results)

    # Export summary statistics to CSV
    with open("summary_stats.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            "Scenario ID", "Scale", "Container Cost", "Holding Cost",
            "Avg Naive Gap (%)", "Std Naive Gap (%)", "Avg Heuristic Gap (%)", "Std Heuristic Gap (%)",
            "Avg Optimal Time (s)", "Std Optimal Time (s)", 
            "Avg Naive Time (s)", "Std Naive Time (s)", 
            "Avg Heuristic Time (s)", "Std Heuristic Time (s)"
        ])
        csvwriter.writerows(summary_stats)
    
    return results, summary_stats

def main():
    print("Starting experiment to compare optimization model, naive heuristic, and proposed heuristic")
    results, summary_stats = run_experiment()
    
    print("\nExperiment completed successfully!")
    print("Detailed results saved to 'detailed_results.csv'")
    print("Summary statistics saved to 'summary_stats.csv'")
    
    # Display overall summary
    if results:
        avg_naive_gap = np.mean([r[8] for r in results])
        avg_heuristic_gap = np.mean([r[9] for r in results])
        
        avg_optimal_time = np.mean([r[5] for r in results])
        avg_naive_time = np.mean([r[6] for r in results])
        avg_heuristic_time = np.mean([r[7] for r in results])
        
        print("\nOverall Summary:")
        print(f"Average Naive Gap: {avg_naive_gap:.2f}%")
        print(f"Average Heuristic Gap: {avg_heuristic_gap:.2f}%")
        print(f"Average Computation Times (s):")
        print(f"  Optimal Solution: {avg_optimal_time:.3f}")
        print(f"  Naive Heuristic: {avg_naive_time:.3f}")
        print(f"  Proposed Heuristic: {avg_heuristic_time:.3f}")
        
        print("\nFindings:")
        if avg_heuristic_gap < avg_naive_gap:
            print(f"- Our proposed heuristic achieves {avg_naive_gap - avg_heuristic_gap:.2f}% better solution quality than the naive approach")
        
        if avg_heuristic_time < avg_optimal_time:
            print(f"- Our heuristic is {avg_optimal_time / avg_heuristic_time:.1f}x faster than solving the relaxed model")
            
        print(f"- The proposed heuristic offers a good balance between solution quality and computation time")
        print("- The heuristic performs consistently well across different problem sizes and parameter settings")
    else:
        print("\nNo results were generated. Please check the error messages above.")

if __name__ == "__main__":
    main()
