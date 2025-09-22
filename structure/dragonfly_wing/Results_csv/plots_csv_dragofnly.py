import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

############### TIMOSHENKO #############################

# Read the CSV files
df = pd.read_csv('timo_displ_over_line.csv')

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Extract data for plotting
timodisplacement_0 = df['Displacements_0']
timodisplacement_1 = df['Displacements_1']
xtimo = df["Points_Magnitude"]

############ LINEAR ELASTICITY ###########################

df = pd.read_csv('displacement_over_center_line.csv')

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Extract data for plotting
lindisplacement_0 = df['Displacements_0']
lindisplacement_1 = df['Displacements_1']
xlin = df["Points_Magnitude"]

############ PLOTTING #################################

# Create figure with subplots
plt.figure(figsize=(18, 14))

# Plot Displacement_0 comparison
plt.subplot(2, 1, 1)
plt.plot(xtimo, timodisplacement_0,  color='purple', label='Timoshenko', linewidth=3)
plt.plot(xlin, lindisplacement_0, color='green', label='Linear Elasticity', linewidth=3)
plt.title('X Displacement Comparison', fontsize=20)
plt.ylabel('ux (cm)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.legend(prop={'size': 18})

# Plot Displacement_1 comparison
plt.subplot(2, 1, 2)
plt.plot(xtimo, timodisplacement_1, color='purple', label='Timoshenko', linewidth=3)
plt.plot(xlin, lindisplacement_1, color='green', label='Linear Elasticity', linewidth=3)
plt.title('Y Displacement Comparison', fontsize=20)
plt.ylabel('uy (cm)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.legend(prop={'size': 18})

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()

# Optional: Save the figure
plt.savefig('displacement_comparison_linear_timo.png', dpi=300, bbox_inches='tight')

# Print some basic statistics for comparison
print("=== DISPLACEMENT COMPARISON STATISTICS ===")
print("\nTimoshenko Theory:")
print(f"Displacement_0 - Min: {timodisplacement_0.min():.6f}, Max: {timodisplacement_0.max():.6f}")
print(f"Displacement_1 - Min: {timodisplacement_1.min():.6f}, Max: {timodisplacement_1.max():.6f}")

print("\nLinear Elasticity:")
print(f"Displacement_0 - Min: {lindisplacement_0.min():.6f}, Max: {lindisplacement_0.max():.6f}")
print(f"Displacement_1 - Min: {lindisplacement_1.min():.6f}, Max: {lindisplacement_1.max():.6f}")

# Calculate and display differences
if len(timodisplacement_0) == len(lindisplacement_0):
    diff_0 = np.abs(timodisplacement_0 - lindisplacement_0)
    diff_1 = np.abs(timodisplacement_1 - lindisplacement_1)
    print(f"\nAverage absolute difference:")
    print(f"Displacement_0: {diff_0.mean():.6f}")
    print(f"Displacement_1: {diff_1.mean():.6f}")
else:
    print(f"\nNote: Different number of data points - Timoshenko: {len(timodisplacement_0)}, Linear: {len(lindisplacement_0)}")