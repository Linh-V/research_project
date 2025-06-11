# 2D Cantilever Linear Elastic Beam

# Analytical solution: Max deflection = -PL^3/3EI

import getfem as gf 
import numpy as np

E = 70e9  # Young's Modulus in Pa (N/m²)
A = 0.0005 * 0.0005  # Cross-sectional area in m²
I = 0.0005*(0.0005**3)/12  # Second moment of inertia in m⁴
L = 0.1 # Beam length in m
nu = 0.3    # Poisson ratio
F = -0.01   # Force at the right Boundary in N
n_dof = 3

############## Analytical solution #########################################################
# Calculate analytical solution for comparison
analytical_tip_deflection = abs(F * L**3) / (3 * E * I)
print(f"Expected analytical tip deflection: {analytical_tip_deflection:.6e} meters")


