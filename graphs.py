import matplotlib.pyplot as plt
import eval 
from param import k_values, reg_values
from collections import Counter

# Find least reg for each k
results = eval.results  # results is a dict {(k, reg): value}
least_reg_dict = {}
for k in k_values:
    min_reg = min(reg_values, key=lambda reg: results[(k, reg)])
    least_reg_dict[k] = min_reg

# Calculate frequency of each minimum regularization value
min_reg_values = list(least_reg_dict.values())
frequency = Counter(min_reg_values)

# Plotting
try:
    plt.figure(figsize=(10, 6))
    plt.bar(frequency.keys(), frequency.values(), color='b', width=1.0)  
    plt.xlabel('Minimum Regularization Strength')
    plt.ylabel('Frequency')
    plt.title('Frequency of Minimum Regularization Strengths')
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
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Error generating plot: {str(e)}")

