import matplotlib.pyplot as plt
import eval  # Module containing risk and upperbound results
from param import k_values, reg_values  # Module containing parameter lists
from collections import Counter
import pandas as pd

# --- Data Loading and Preparation ---

# Load risk and upperbound data from the eval module
# Assumes eval.risk and eval.upperbound are dictionaries like {(k, reg): value}
risk = eval.risk
upperbound = eval.upperbound

# Create Pandas DataFrames for better visualization and analysis of risk and upperbound values
risk_df = pd.DataFrame(index=k_values, columns=reg_values)
upperbound_df = pd.DataFrame(index=k_values, columns=reg_values)

# Populate the DataFrames
for (k, reg), value in risk.items():
    risk_df.at[k, reg] = value

for (k, reg), value in upperbound.items():
    upperbound_df.at[k, reg] = value

# Print the DataFrames to the console
print("\nRisk values for different k and regularization strengths:")
print(risk_df)
print("\nUpperbound values for different k and regularization strengths:")
print(upperbound_df)

# --- Analysis: Find Optimal Regularization ---

# Determine the regularization strength ('reg') that minimizes risk and upperbound for each 'k'
least_reg_dict_risk = {}
least_reg_dict_upperbound = {}

for k in k_values:
    # Find the 'reg' that gives the minimum value in the risk dictionary for the current 'k'
    min_reg_risk = min(reg_values, key=lambda reg: risk[(k, reg)])
    # Find the 'reg' that gives the minimum value in the upperbound dictionary for the current 'k'
    min_reg_upperbound = min(reg_values, key=lambda reg: upperbound[(k, reg)])
    least_reg_dict_risk[k] = min_reg_risk
    least_reg_dict_upperbound[k] = min_reg_upperbound

# --- Analysis: Frequency of Optimal Regularization ---

# Calculate how often each regularization value appears as the minimum for risk
min_reg_values_risk = list(least_reg_dict_risk.values())
frequency_risk = Counter(min_reg_values_risk)

# Calculate how often each regularization value appears as the minimum for upperbound
min_reg_values_upperbound = list(least_reg_dict_upperbound.values())
frequency_upperbound = Counter(min_reg_values_upperbound)

# --- Plotting ---
try:
    # Plot 1: Optimal Regularization vs. k
    # Shows the best regularization strength for each correlation parameter k.
    plt.figure(figsize=(10, 6))
    plt.scatter(list(least_reg_dict_risk.keys()), list(least_reg_dict_risk.values()), marker='o', color='b', label='Risk')
    plt.scatter(list(least_reg_dict_upperbound.keys()), list(least_reg_dict_upperbound.values()), marker='x', color='r', label='Upperbound')
    plt.xlabel('k (Correlation Parameter)')
    plt.ylabel('Least Regularization Strength (log scale)')
    plt.title('Optimal Regularization Strength vs. Correlation Parameter (k)')
    plt.yscale('log') # Use log scale for potentially wide range of reg values
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot 2: Minimum Risk/Upperbound Value vs. k
    # Shows the actual minimum risk/upperbound value achieved for each k using its optimal regularization.
    plt.figure(figsize=(10, 6))
    plt.scatter(list(least_reg_dict_risk.keys()), [risk[(k, least_reg_dict_risk[k])] for k in least_reg_dict_risk.keys()], marker='o', color='b', label='Minimum Risk')
    plt.scatter(list(least_reg_dict_upperbound.keys()), [upperbound[(k, least_reg_dict_upperbound[k])] for k in least_reg_dict_upperbound.keys()], marker='x', color='r', label='Minimum Upperbound')
    plt.xlabel('k (Correlation Parameter)')
    plt.ylabel('Corresponding Minimum Value')
    plt.title('Minimum Achieved Risk/Upperbound vs. Correlation Parameter (k)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Combined Plot 3 & 4: Frequency of Optimal Regularization Strengths
    # Shows how many times each regularization value was found to be optimal for minimizing risk and upperbound.
    plt.figure(figsize=(10, 6))
    plt.scatter(list(frequency_risk.keys()), list(frequency_risk.values()), marker='o', color='b', label='Risk Frequency')
    plt.scatter(list(frequency_upperbound.keys()), list(frequency_upperbound.values()), marker='x', color='r', label='Upperbound Frequency') # Use different marker/color
    plt.xlabel('Regularization Strength (log scale)')
    plt.ylabel('Frequency (Count)')
    plt.title('Frequency of Optimal Regularization Strengths (Risk & Upperbound)')
    plt.xscale('log') # Use log scale for potentially wide range of reg values
    plt.grid(True)
    plt.legend() # Add legend to distinguish the plots
    plt.show()

    # --- Removed original Plot 3 and Plot 4 ---
    # # Plot 3: Frequency of Optimal Regularization Strengths (Risk)
    # # Shows how many times each regularization value was found to be optimal for minimizing risk.
    # plt.figure(figsize=(10, 6))
    # plt.scatter(list(frequency_risk.keys()), list(frequency_risk.values()), marker='o', color='b')
    # plt.xlabel('Regularization Strength (log scale)')
    # plt.ylabel('Frequency (Count)')
    # plt.title('Frequency of Optimal Regularization Strengths (Risk)')
    # plt.xscale('log') # Use log scale for potentially wide range of reg values
    # plt.grid(True)
    # plt.show()
    #
    # # Plot 4: Frequency of Optimal Regularization Strengths (Upperbound)
    # # Shows how many times each regularization value was found to be optimal for minimizing the upperbound.
    # plt.figure(figsize=(10, 6))
    # plt.scatter(list(frequency_upperbound.keys()), list(frequency_upperbound.values()), marker='o', color='r')
    # plt.xlabel('Regularization Strength (log scale)')
    # plt.ylabel('Frequency (Count)')
    # plt.title('Frequency of Optimal Regularization Strengths (Upperbound)')
    # plt.xscale('log') # Use log scale for potentially wide range of reg values
    # plt.grid(True)
    # plt.show()

except KeyError as e:
    print(f"Error generating plot: Missing key {e}. Check if risk/upperbound dictionaries cover all k/reg combinations.")
except Exception as e:
    print(f"An unexpected error occurred during plotting: {str(e)}")

