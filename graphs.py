import matplotlib.pyplot as plt
import eval
from param import k_values, reg_values
from collections import Counter
import pandas as pd

# Find least reg for each k
risk = eval.risk  # results is a dict {(k, reg): value}
upperbound = eval.upperbound

# Create DataFrames
risk_df = pd.DataFrame(index=k_values, columns=reg_values)
upperbound_df = pd.DataFrame(index=k_values, columns=reg_values)

# Populate the DataFrames with risk and upperbound values
for (k, reg), value in risk.items():
    risk_df.at[k, reg] = value

for (k, reg), value in upperbound.items():
    upperbound_df.at[k, reg] = value

print("\nRisk values for different k and regularization strengths:")
print(risk_df)
print("\nUpperbound values for different k and regularization strengths:")
print(upperbound_df)


# Calculate least reg for risk and upperbound
least_reg_dict_risk = {}
least_reg_dict_upperbound = {}

for k in k_values:
    min_reg_risk = min(reg_values, key=lambda reg: risk[(k, reg)])
    min_reg_upperbound = min(reg_values, key=lambda reg: upperbound[(k, reg)])
    least_reg_dict_risk[k] = min_reg_risk
    least_reg_dict_upperbound[k] = min_reg_upperbound

# Calculate frequency of each minimum regularization value
min_reg_values_risk = list(least_reg_dict_risk.values())
frequency_risk = Counter(min_reg_values_risk)

min_reg_values_upperbound = list(least_reg_dict_upperbound.values())
frequency_upperbound = Counter(min_reg_values_upperbound)

# Plotting
try:
    # First Graph: k vs least reg for risk and upperbound
    plt.figure(figsize=(10, 6))
    plt.scatter(list(least_reg_dict_risk.keys()), list(least_reg_dict_risk.values()), marker='o', color='b', label='Risk')
    plt.scatter(list(least_reg_dict_upperbound.keys()), list(least_reg_dict_upperbound.values()), marker='x', color='r', label='Upperbound')
    plt.xlabel('k (Correlation Parameter)')
    plt.ylabel('Least Regularization Strength')
    plt.title('Least Regularization vs. Correlation Parameter')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Second Graph: k vs corresponding value of least reg for risk and upperbound
    plt.figure(figsize=(10, 6))
    plt.scatter(list(least_reg_dict_risk.keys()), [risk[(k, least_reg_dict_risk[k])] for k in least_reg_dict_risk.keys()], marker='o', color='b', label='Risk')
    plt.scatter(list(least_reg_dict_upperbound.keys()), [upperbound[(k, least_reg_dict_upperbound[k])] for k in least_reg_dict_upperbound.keys()], marker='x', color='r', label='Upperbound')
    plt.xlabel('k (Correlation Parameter)')
    plt.ylabel('Corresponding Value of Least Regularization')
    plt.title('Corresponding Value vs. Correlation Parameter')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Third Graph: Frequency of least reg for risk
    plt.figure(figsize=(10, 6))
    plt.scatter(list(frequency_risk.keys()), list(frequency_risk.values()), marker='o', color='b')
    plt.xlabel('Regularization Strength')
    plt.ylabel('Frequency')
    plt.title('Frequency of Least Regularization Strengths (Risk)')
    plt.xscale('log')
    plt.grid(True)
    plt.show()

    # Fourth Graph: Frequency of least reg for upperbound
    plt.figure(figsize=(10, 6))
    plt.scatter(list(frequency_upperbound.keys()), list(frequency_upperbound.values()), marker='o', color='r')
    plt.xlabel('Regularization Strength')
    plt.ylabel('Frequency')
    plt.title('Frequency of Least Regularization Strengths (Upperbound)')
    plt.xscale('log')
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"Error generating plot: {str(e)}")

