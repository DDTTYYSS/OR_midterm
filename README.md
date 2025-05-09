# Inventory Management Optimization

This project implements a heuristic algorithm for solving an inventory management problem with ocean and air shipping options. The algorithm optimizes shipping decisions considering:

- Multiple products with different volumes and costs
- Two shipping methods (ocean and air) with different lead times
- Container-based costs for ocean shipping
- Variable costs for air shipping
- Inventory tracking and management

## Features

- Reads input data from Excel files
- Calculates optimal shipping method for each order
- Prevents negative inventory
- Considers lead times for different shipping methods
- Outputs detailed results to Excel file

## Requirements

- Python 3.x
- pandas
- numpy
- openpyxl

## Usage

1. Prepare your input Excel file with the following sheets:
   - Demand
   - Shipping cost
   - Inventory cost

2. Run the program:
   ```bash
   python heuristic_solution.py
   ```

3. Check the output file `heuristic_results.xlsx` for detailed results including:
   - Order quantities and timing
   - Inventory levels
   - Container usage
   - Total costs

## Output Format

The program generates an Excel file with multiple sheets:
- Summary: Overall cost and problem size
- Orders: Detailed order information by period
- Inventory: Inventory levels by product and period
- Containers: Container usage by period 