import matplotlib.pyplot as plt
import eval  # Module containing risk and upperbound results
from param import k_values, reg_values  # Module containing parameter lists
from collections import Counter
import pandas as pd
import os # Import os module to handle paths

# --- Data Loading and Preparation ---

# Load risk and upperbound data from the eval module
# Assumes eval.risk and eval.upperbound are dictionaries with tuple keys like {(k, reg): value}
risk_dict = eval.risk
upperbound_dict = eval.upperbound

# --- Create Pandas DataFrames using Series and unstack ---
# Convert the dictionaries with tuple keys to Pandas Series for easier manipulation
risk_series = pd.Series(risk_dict)
upperbound_series = pd.Series(upperbound_dict)

# Set MultiIndex names for clarity before unstacking
risk_series.index.names = ['k', 'reg']
upperbound_series.index.names = ['k', 'reg']

# Unstack the 'reg' level of the MultiIndex to become columns.
# This transforms the Series into a DataFrame where 'k' is the index and 'reg' values are the columns.
risk_df = risk_series.unstack(level='reg')
upperbound_df = upperbound_series.unstack(level='reg')

# Ensure the columns (reg_values) and index (k_values) are sorted according to the imported lists.
# This is important if the original dictionaries weren't ordered or if specific ordering is desired.
risk_df = risk_df.reindex(index=k_values, columns=reg_values)
upperbound_df = upperbound_df.reindex(index=k_values, columns=reg_values)

# --- Removed original DataFrame population code ---

# Print the DataFrames to the console for verification
# Note: Terminal display might wrap lines if the table is wide.
print("\nRisk values (rows=k, columns=reg):")
print(risk_df)
print("\nUpperbound values (rows=k, columns=reg):")
print(upperbound_df)

# --- Save DataFrames to CSV ---
output_dir = "results" # Define a subdirectory to save results
os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist

csv_path_risk = os.path.join(output_dir, "risk_results.csv")
csv_path_upperbound = os.path.join(output_dir, "upperbound_results.csv")

try:
    # Save the DataFrames to CSV files in the specified directory
    risk_df.to_csv(csv_path_risk)
    upperbound_df.to_csv(csv_path_upperbound)
    print(f"\nRisk data saved to: {csv_path_risk}")
    print(f"Upperbound data saved to: {csv_path_upperbound}")
except Exception as e:
    print(f"Error saving DataFrames to CSV: {str(e)}")


# --- Analysis: Find Optimal Regularization ---

# Determine the regularization strength ('reg') that minimizes risk and upperbound for each 'k'
least_reg_dict_risk = {}
least_reg_dict_upperbound = {}

# Iterate through each k value
for k in k_values:
    # For the current k, find the 'reg' value from reg_values that yields the minimum risk value in the original risk_dict.
    # The `key` argument of `min` specifies a function (lambda) to determine the value to compare for minimization.
    min_reg_risk = min(reg_values, key=lambda reg: risk_dict[(k, reg)])
    # Similarly, find the 'reg' value that yields the minimum upperbound value in the original upperbound_dict.
    min_reg_upperbound = min(reg_values, key=lambda reg: upperbound_dict[(k, reg)])
    # Store the optimal 'reg' for the current 'k'
    least_reg_dict_risk[k] = min_reg_risk
    least_reg_dict_upperbound[k] = min_reg_upperbound

# --- Analysis: Frequency of Optimal Regularization ---

# Calculate how often each regularization value appears as the optimal one for minimizing risk
min_reg_values_risk = list(least_reg_dict_risk.values())
frequency_risk = Counter(min_reg_values_risk) # Counter efficiently counts occurrences

# Calculate how often each regularization value appears as the optimal one for minimizing the upperbound
min_reg_values_upperbound = list(least_reg_dict_upperbound.values())
frequency_upperbound = Counter(min_reg_values_upperbound) # Counter efficiently counts occurrences

# --- Plotting (Reordered and Saving to Files) ---
try:
    # --- Plot 1: Minimum Risk/Upperbound Value vs. k ---
    # Shows the actual minimum risk/upperbound value achieved for each k, using its corresponding optimal regularization.
    plt.figure(figsize=(10, 6))
    # Plot minimum risk values found for each k
    plt.scatter(list(least_reg_dict_risk.keys()), [risk_dict[(k, least_reg_dict_risk[k])] for k in least_reg_dict_risk.keys()], marker='o', color='b', label='Minimum Risk')
    # Plot minimum upperbound values found for each k
    plt.scatter(list(least_reg_dict_upperbound.keys()), [upperbound_dict[(k, least_reg_dict_upperbound[k])] for k in least_reg_dict_upperbound.keys()], marker='x', color='r', label='Minimum Upperbound')
    plt.xlabel('k (Correlation Parameter)')
    plt.ylabel('Corresponding Minimum Value')
    plt.title('Minimum Achieved Risk/Upperbound vs. Correlation Parameter (k)')
    plt.grid(True)
    plt.legend()
    plot1_filename = os.path.join(output_dir, "plot1_min_value_vs_k.png")
    plt.savefig(plot1_filename) # Save the plot to a file
    print(f"Plot 1 saved to: {plot1_filename}")
    plt.close() # Close the figure to free memory

    # --- Combined Plot 2: Frequency of Optimal Regularization Strengths ---
    # Shows how many times each regularization value was found to be optimal across all k values.
    plt.figure(figsize=(10, 6))
    # Plot frequency counts for optimal 'reg' in risk minimization
    plt.scatter(list(frequency_risk.keys()), list(frequency_risk.values()), marker='o', color='b', label='Risk Frequency')
    # Plot frequency counts for optimal 'reg' in upperbound minimization
    plt.scatter(list(frequency_upperbound.keys()), list(frequency_upperbound.values()), marker='x', color='r', label='Upperbound Frequency')
    plt.xlabel('Regularization Strength (log scale)')
    plt.ylabel('Frequency (Count)')
    plt.title('Frequency of Optimal Regularization Strengths (Risk & Upperbound)')
    plt.xscale('log') # Use log scale for x-axis as 'reg' values might span orders of magnitude
    plt.grid(True)
    plt.legend() # Add legend to distinguish risk and upperbound frequencies
    plot2_filename = os.path.join(output_dir, "plot2_frequency.png")
    plt.savefig(plot2_filename) # Save the plot to a file
    print(f"Plot 2 saved to: {plot2_filename}")
    plt.close() # Close the figure

    # --- Plot 1: Optimal Regularization vs. k ---
    # Shows the best regularization strength ('reg') found for each correlation parameter 'k'.
    plt.figure(figsize=(10, 6))
    # Plot optimal 'reg' for risk minimization for each k
    plt.scatter(list(least_reg_dict_risk.keys()), list(least_reg_dict_risk.values()), marker='o', color='b', label='Risk')
    # Plot optimal 'reg' for upperbound minimization for each k
    plt.scatter(list(least_reg_dict_upperbound.keys()), list(least_reg_dict_upperbound.values()), marker='x', color='r', label='Upperbound')
    plt.xlabel('k (Correlation Parameter)')
    plt.ylabel('Optimal Regularization Strength (log scale)')
    plt.title('Optimal Regularization Strength vs. Correlation Parameter (k)')
    plt.yscale('log') # Use log scale for y-axis as 'reg' values might span orders of magnitude
    plt.grid(True)
    plt.legend()
    plot3_filename = os.path.join(output_dir, "plot3_optimal_reg_vs_k.png")
    plt.savefig(plot3_filename) # Save the plot to a file
    print(f"Plot 3 saved to: {plot3_filename}")
    plt.close() # Close the figure

    # --- Removed original separate frequency plots (Plot 3 and Plot 4) ---

except KeyError as e:
    # Handle cases where a specific (k, reg) combination might be missing from the input dictionaries
    print(f"Error generating plot: Missing key {e}. Check if risk_dict/upperbound_dict cover all k/reg combinations from param.py.")
except Exception as e:
    # Catch any other unexpected errors during plotting
    print(f"An unexpected error occurred during plotting: {str(e)}")

