import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV files
df = pd.read_csv('timoshenko_disp.csv')
df_rot = pd.read_csv("timosjenko_rot.csv")

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()
df_rot.columns = df_rot.columns.str.strip()

# Extract data for plotting
point_ids = df['Point ID']
displacement_0 = df['Displacements_0']
displacement_1 = df['Displacements_1']
displacement_2 = df['Displacements_2']
rot = df_rot['Rotations']

# Select every two nodes (every other data point)
# Using slice notation [::2] to get every 2nd element starting from index 0
point_ids_filtered = point_ids[::2]
displacement_0_filtered = displacement_0[::2]
displacement_1_filtered = displacement_1[::2]
displacement_2_filtered = displacement_2[::2]
rot_filtered = rot[::2]
points = []
print(f"Original data points: {len(point_ids)}")
print(f"Filtered data points (every 2 nodes): {len(point_ids_filtered)}")

# Create figure with subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

# Plot 1: Displacement_0 vs Point_ID (every 2 nodes)
ax1.plot(point_ids_filtered, displacement_0_filtered, 'b-o', linewidth=2, markersize=6, label='Displacement 0')
ax1.set_xlabel('Point ID')
ax1.set_ylabel('Displacement 0')
ax1.set_title('Displacement 0 vs Point ID (Every 2 Nodes)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Displacement_1 vs Point_ID (every 2 nodes)
ax2.plot(point_ids_filtered, displacement_1_filtered, 'r-o', linewidth=2, markersize=6, label='Displacement 1')
ax2.set_xlabel('Point ID')
ax2.set_ylabel('Displacement 1')
ax2.set_title('Displacement 1 vs Point ID (Every 2 Nodes)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Rotation vs Point_ID (every 2 nodes)
ax3.plot(point_ids_filtered, rot_filtered, 'g-o', linewidth=2, markersize=6, label='Rotation')
ax3.set_xlabel('Point ID')
ax3.set_ylabel('Rotation')
ax3.set_title('Rotation vs Point ID (Every 2 Nodes)')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

# Optional: Create a comparison plot with all three variables (every 2 nodes)
plt.figure(figsize=(14, 8))
plt.subplot(1, 1, 1)
plt.plot(point_ids_filtered, displacement_0_filtered, 'b-o', linewidth=2, markersize=4, label='Displacement 0')
plt.plot(point_ids_filtered, displacement_1_filtered, 'r-s', linewidth=2, markersize=4, label='Displacement 1')
plt.plot(point_ids_filtered, rot_filtered, 'g-^', linewidth=2, markersize=4, label='Rotation')
plt.xlabel('Point ID')
plt.ylabel('Value')
plt.title('Comparison of Displacements and Rotation (Every 2 Nodes)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print some information about the filtered data
print("\n--- Filtered Data Info (Every 2 Nodes) ---")
print(f"Point IDs plotted: {list(point_ids_filtered)}")
print(f"Displacement 0 - Min: {displacement_0_filtered.min():.6f}, Max: {displacement_0_filtered.max():.6f}")
print(f"Displacement 1 - Min: {displacement_1_filtered.min():.6f}, Max: {displacement_1_filtered.max():.6f}")
print(f"Rotation - Min: {rot_filtered.min():.6f}, Max: {rot_filtered.max():.6f}")