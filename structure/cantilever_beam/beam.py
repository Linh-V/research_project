# 1D Cantilever Beam

# Analytical solution: Max deflection = -PL^3/3EI

import getfem as gf 
import numpy as np

E = 70e9  # Young's Modulus in Pa (N/m²)
H = 0.005 # Height in m
B = 0.005 # Base in m
A = B * H  # Cross-sectional area in m²
I = H*(B**3)/12  # Second moment of inertia in m⁴
L = 0.1 # Beam length in m
nu = 0.3    # Poisson ratio
F = -0.01   # Force at the right Boundary in N
n_dof = 3

############## Analytical solution #########################################################
# Calculate analytical solution for comparison
analytical_tip_deflection = abs(F * L**3) / (3 * E * I)
print(f"Expected analytical tip deflection: {analytical_tip_deflection:.6e} meters")

############## Mesh #########################################################

mesh = gf.Mesh("cartesian", np.linspace(0, L, 41)) # Mesh with 2 elements and 3 nodes

############## Boundary Identification #########################################################

fleft = mesh.outer_faces_with_direction(v=[-1.0], angle=0.01) #returns faces not shared by convex with outward vector v
fright = mesh.outer_faces_with_direction(v=[+1.0], angle=0.01)
NEUMANN_BOUNDARY = 1
DIRICHLET_BOUNDARY = 2
mesh.set_region(NEUMANN_BOUNDARY, fright) #flag faces
mesh.set_region(DIRICHLET_BOUNDARY, fleft)

############## Finite Element / Integration #################################

mfu = gf.MeshFem(mesh, 1)  # Finite element for the elastic displacement
mfu.set_classical_fem(2)

mfv = gf.MeshFem(mesh, 1)  # Finite element for the elastic displacement
mfv.set_classical_fem(2)

mftheta = gf.MeshFem(mesh, 1)
mftheta.set_classical_fem(2)  # One degree less for rotation

mim = gf.MeshIm(mesh, 17)   # Integration method

############## The model (weak form of the PDE) #############################
# empty real model
md = gf.Model('real')

# declare that the unknowns of the system on the finite element method "mf"
md.add_fem_variable("u", mfu)        # axial displacement
md.add_fem_variable("v", mfv)        # transverse displacement
md.add_fem_variable("theta", mftheta)   # bending angle

# Adding a constant scalar value
md.add_initialized_data("EA", E*A)
md.add_initialized_data("EI", E*I)

# Building the weak form for both axial and transverse displacement
# Axial stiffness term (axial displacement) from PDE EA * d²u/dx² = 0
md.add_linear_term(mim, "EA * Grad_u . Grad_Test_u")

# # Bending stiffness term (transverse displacement) from PDE EI d⁴v/dx⁴ = 0
md.add_linear_term(mim, "EI * Grad(Grad_v) : Grad(Grad_Test_v)")

# Constraint: theta = dv/dx (correct weak form)
# ∫ (theta - dv/dx) * test_function dx = 0
# This gives us: ∫ theta * test_theta dx - ∫ (dv/dx) * test_theta dx = 0
# Integration by parts: ∫ theta * test_theta dx + ∫ v * d(test_theta)/dx dx = boundary terms
large_penalty = 1e8
md.add_initialized_data("constraint_penalty", large_penalty)
md.add_linear_term(mim, "constraint_penalty * (theta - Grad_v) . (Test_theta - Grad_Test_v)")

# add force terms to the RHS (mim_dirac method)
md.add_initialized_data('VerticalForce', F)
md.add_source_term_brick(mim, 'v', 'VerticalForce', NEUMANN_BOUNDARY)

############## Boundary Conditions ########################################################

# Fix the left side
md.add_Dirichlet_condition_with_multipliers(mim, "u", 0, DIRICHLET_BOUNDARY)
md.add_Dirichlet_condition_with_multipliers(mim, "v", 0, DIRICHLET_BOUNDARY)
md.add_Dirichlet_condition_with_multipliers(mim, "theta", 0, DIRICHLET_BOUNDARY)

############### Solve and Export #############################################

# solve the linear system
md.solve()

# main unknown
U = md.variable("u")
V = md.variable("v")

max_u = np.max(np.abs(U))
max_v = np.max(np.abs(V))
print(f"Maximum axial displacement: {max_u} meters")
print(f"Maximum transverse displacement: {max_v} meters")

print(U)
print(V)

############### Plotting the solution #############################################
import matplotlib.pyplot as plt

# 1D beam nodes (same as used for mesh creation)
x0 = np.linspace(0, L, 41)  # 41 points from 0 to L
y0 = np.zeros_like(x0)      # flat beam (undeformed)

# Extract displacement results: every other DOF (P3 elements → more DOFs)
# We assume the primary DOFs are at nodes (safe if classical FEM is used)
v_nodes = np.array(V)[::2]
u_nodes = np.array(U)[::2]

# Apply displacement scaling for visualization
scale = 1e4  # adjust for visual clarity (no physical meaning)
x_disp = x0 + scale * u_nodes
y_disp = y0 + scale * v_nodes

# Plot setup
plt.figure(figsize=(10, 4))
plt.plot(x0, y0, color='white', linewidth=1, label="Original shape")
plt.plot(x_disp, y_disp, color='cyan', linewidth=1.5, label="Deformed shape")
plt.gca().set_facecolor('#2c2f3a')  # dark background
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.legend(loc="upper left", facecolor="lightgray")
plt.title("Deformation of 1D Beam (Scaled)", color='white')

# Save as image
plt.savefig("Beam_1D.png", dpi=300, facecolor='#2c2f3a')