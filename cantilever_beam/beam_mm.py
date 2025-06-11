# 1D Cantilever Beam

# Analytical solution: Max deflection = -PL^3/3EI

import getfem as gf 
import numpy as np

E = 70e3  # Young's Modulus in MPa
H = 5 # Height in mm
B = 5 # Base in mm
A = B * H  # Cross-sectional area in mm²
I = H*(B**3)/12  # Second moment of inertia in mm⁴
L = 100 # Beam length in mm
nu = 0.3    # Poisson ratio
F = -1   # Force at the right Boundary in N
n_dof = 3

############## Analytical solution #########################################################
# Calculate analytical solution for comparison
analytical_tip_deflection = abs(F * L**3) / (3 * E * I)
print(f"Expected analytical tip deflection: {analytical_tip_deflection:.6e} mm")

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
mfv.set_classical_fem(3)

mftheta = gf.MeshFem(mesh, 1)
mftheta.set_classical_fem(2)  # One degree less for rotation

mim = gf.MeshIm(mesh, 10)   # Integration method

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
print(f"Maximum axial displacement: {max_u} mm")
print(f"Maximum transverse displacement: {max_v} mm")

# export computed solution using vtk format (you can load it with Paraview)
# mfu.export_to_vtk("displacement_u.vtk", U, "Axial displacement")
# mfv.export_to_vtk("displacement_v.vtk", V, "Transverse displacement")

print(U)
print(V)

