import matplotlib.pyplot as plt
import eval
from param import k_values, reg_values
from collections import Counter
import pandas as pd

# Find least reg for each k
risk = eval.risk  # results is a dict {(k, reg): value}

# Create a DataFrame with k as rows and reg as columns
risk_df = pd.DataFrame(index=k_values, columns=reg_values)

# Populate the DataFrame with risk values
for (k, reg), value in risk.items():
    risk_df.at[k, reg] = value

print(risk_df)



least_reg_dict = {}
for k in k_values:
    min_reg = min(reg_values, key=lambda reg: risk[(k, reg)])
    least_reg_dict[k] = min_reg

# Calculate frequency of each minimum regularization value
min_reg_values = list(least_reg_dict.values())
frequency = Counter(min_reg_values)

# Plotting
try:
    plt.figure(figsize=(10, 6))
    plt.scatter(list(frequency.keys()), list(frequency.values()), marker='o', color='b')
    plt.xlabel('Minimum Regularization Strength')
    plt.ylabel('Frequency')
    plt.title('Frequency of Minimum Regularization Strengths')
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Error generating plot: {str(e)}")

try:
    plt.figure(figsize=(10, 6))
    plt.scatter(list(least_reg_dict.keys()), list(least_reg_dict.values()), 
             marker='o', linestyle='--', color='b')
    plt.xlabel('k (Correlation Parameter)')
    plt.ylabel('Optimal Regularization Strength')
    plt.title('Optimal Regularization vs. Correlation Parameter')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Error generating plot: {str(e)}")

