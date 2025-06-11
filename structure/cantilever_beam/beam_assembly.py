# 1D Cantilever Beam Assembly at 90°. >> doesn't work in a 1D space

import getfem as gf 
import numpy as np

E = 70e9  # Young's Modulus in Pa (N/m²)
A = 0.0005 * 0.0005  # Cross-sectional area in m²
I = 0.0005*(0.0005**3)/12  # Second moment of inertia in m⁴
L = 0.1 # Beam length in m
nu = 0.3    # Poisson ratio
F = -0.01   # Force at the right Boundary in N
n_dof = 3

############## Mesh #########################################################
points = np.array([
    [0],           # P1
    [L],        # P2
    [L],    # P3

])

# creating the mesh with all the beam's node points
Mesh = gf.Mesh('empty', 1)
ind = []
for point in points: 
    ind.append(Mesh.add_point(point))

# now it's needed to connect all this points:
GT = gf.GeoTrans('GT_PK(1,1)')  # 1D element in 1D ref space
for i in range(len(ind)-1):# we pass all the points since they are in line we can pass them with a for cycle
    # This Geometric Transformation defines how the element is mapped to my mesh 
    # hence, is connectining the reference elements [-1,1] or [0,1] to the segment of my mesh
    # the element in mapped to P = 1 so a 1d dimension and with a K=1 hence a linear transformation (K=2 quadratic transformation)
    Mesh.add_convex(GT, np.array([points[i], points[i+1]]).T)

    # add convex means that we are adding a finite element in the mesh, 
    # in this case a linear 1d element (line) which is between the two points defined above

############## Boundary Identification #########################################################

# Defining different regions of the mesh
# (including the different node index of each region to the name)
Nodes_bottom = Mesh.pid_from_coords(points[0])    # correspond to 1st point of the mesh
Nodes_top = Mesh.pid_from_coords(points[1])
Nodes_right = Mesh.pid_from_coords(points[-1])  # correspond to the last point of the mesh

# Defining "sides" from all external nodes
# Is it required here for a 1D mesh with only 1 node on the side not a surface ?
fbottom = Mesh.faces_from_pid(Nodes_bottom)     # defining a "face" containing all the left node 
ftop = Mesh.faces_from_pid(Nodes_top) 
fright = Mesh.faces_from_pid(Nodes_right)   # defining a "face" with all the right node

# Associating a flag to each surface
NEUMANN_BOUNDARY = 1
DIRICHLET_BOUNDARY = 2
CONNECT_BOUNDARY = 3
Mesh.set_region(NEUMANN_BOUNDARY, fright) 
Mesh.set_region(DIRICHLET_BOUNDARY, fbottom)
Mesh.set_region(CONNECT_BOUNDARY, fbottom)

############## Finite Element / Integration #################################

mfu = gf.MeshFem(Mesh, 1)  # Finite element for the elastic displacement
mfu.set_classical_fem(2)

mfv = gf.MeshFem(Mesh, 1)  # Finite element for the elastic displacement
mfv.set_classical_fem(2)

mftheta = gf.MeshFem(Mesh, 1)
mftheta.set_classical_fem(2)  # One degree less for rotation

mim = gf.MeshIm(Mesh, 17)   # Integration method

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

# Impose a 90° angle at point 2
md.add_initialized_data("theta_90", 1.57079633)
md.add_Dirichlet_condition_with_multipliers(mim, "theta", 1, CONNECT_BOUNDARY, "theta_90")

# Impose initial displacement at point 3
md.add_initialized_data("L_val", L)
md.add_Dirichlet_condition_with_multipliers(mim, "v", 1, NEUMANN_BOUNDARY, "L_val")

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

# export computed solution using vtk format (you can load it with Paraview)
# mfu.export_to_vtk("displacement_u.vtk", U, "Axial displacement")
# mfv.export_to_vtk("displacement_v.vtk", V, "Transverse displacement")

print(U)
print(V)