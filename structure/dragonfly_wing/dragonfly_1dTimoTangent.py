# DRAGONFLY WING BEAM #

#Something weird that will allow cheking

import getfem as gf 
import numpy as np
import matplotlib.pyplot as plt

# Paramters need to change according to the Dragonfly paramters:

E = 70e5  # Young's Modulus in Pa (N/m²)
L = 10 # Beam length in cm
b = 0.005*L
h = 0.005*L
A = b * h  # Cross-sectional area in m²
I = b*(h**3)/12  # Second moment of inertia in m⁴
nu = 0.3    # Poisson ratio
k = 5/6     # Shear correction factor
G = E/(2*(1+nu))    # Shear modulus
F = -0.01   # Force at the right Boundary in N


############## Mesh #########################################################

# Dragonfly airfoil:
points = np.array([
    [0, 0],           # P1
    [0.09, 0],        # P2
    [0.14, 0.055],    # P3
    [0.19, 0],        # P4
    [0.24, 0.055],    # P5
    [0.29, 0],        # P6
    [0.34, 0],        # P7
    [0.5, 0.02],      # P8
    [0.6, 0.06],      # P9
    [0.7, 0.08],      # P10
    [0.75, 0.085],    # P11
    [0.8, 0.08],      # P12 
    [0.85, 0.07],     # P13
    [0.9, 0.055],     # P14
    [0.95, 0.04],     # P15
    [1, 0.025]        # P16
])

points = points*L

print(points)

# creating the mesh with all the wing's node points
Mesh = gf.Mesh('empty', 2)
ind = []
for point in points: 
    ind.append(Mesh.add_point(point))

# now it's needed to connect all this points:
beam_angles = []
GT = gf.GeoTrans('GT_PK(1,1)')  # 1D linear element
# After creating the mesh, store the tangent vectors for each element
tangent_vectors = []
for i in range(len(ind)-1):
    dx = points[i+1][0] - points[i][0]
    dy = points[i+1][1] - points[i][1]
    length = np.sqrt(dx**2 + dy**2)
    # Unit tangent vector
    tangent_vectors.append([dx/length, dy/length])
    Mesh.add_convex(GT, np.array([points[i], points[i+1]]).T)
############## Boundary Identification #########################################################

# Defining different regions of the mesh
# (including the different node index of each region to the name)
Nodes_left = Mesh.pid_from_coords(points[0])    # correspond to 1st point of the mesh
Nodes_right = Mesh.pid_from_coords(points[-1])  # correspond to the 2nd point of the mesh

# Defining "sides" from all external nodes
# Is it required here for a 1D mesh with only 1 node on the side not a surface ?
fleft = Mesh.faces_from_pid(Nodes_left)     # defining a "face" containing all the left node 
fright = Mesh.faces_from_pid(Nodes_right)   # defining a "face" with all the right node

# Associating a flag to each surface
NEUMANN_BOUNDARY = 1
DIRICHLET_BOUNDARY = 2
Mesh.set_region(NEUMANN_BOUNDARY, fright) 
Mesh.set_region(DIRICHLET_BOUNDARY, fleft)

############## Finite Element / Integration #################################
# For Timoshenko beam, we need:
# - 2 DOF for displacement (u, w)
# - 1 DOF for rotation (theta)

mfu = gf.MeshFem(Mesh, 2)  # For displacement (u, w)
mfu.set_fem(gf.Fem("FEM_PK(1,3)"))  # Quadratic elements for displacement

mftheta = gf.MeshFem(Mesh, 1)  # For rotation theta
mftheta.set_fem(gf.Fem("FEM_PK(1,3)"))  # Linear elements for rotation

mim = gf.MeshIm(Mesh, gf.Integ("IM_GAUSS1D(99)"))  # Integration method

############## Initialising the model #############################
md = gf.Model('real')

# Add variables
md.add_fem_variable("u", mfu)      # displacement field
md.add_fem_variable("theta", mftheta)  # rotation field

# Material properties

md.add_initialized_data("EA", E*A)
md.add_initialized_data("EI", E*I)
md.add_initialized_data("kGA", k*G*A)


############## Define regions for each element ####################################
# After creating the mesh elements
element_regions = []
for i in range(len(tangent_vectors)):
    region_id = 100 + i  # Start from 100 to avoid conflicts
    Mesh.set_region(region_id, [i])  # Set region for element i
    element_regions.append(region_id)
print(Mesh.regions())
############## Weak Form for Timoshenko Beam ####################################
# Now apply terms element by element

normal_vector =[]

for i, region in enumerate(element_regions):
    tx, ty = tangent_vectors[i]
    nx, ny = -ty, tx
    normal_vector.append([nx,ny])
    md.add_initialized_data(f"tx_{i}", tx)
    md.add_initialized_data(f"ty_{i}", ty)
    md.add_initialized_data(f"nx_{i}", nx)
    md.add_initialized_data(f"ny_{i}", ny)
    
    # Create unit vectors as 2x1 matrices for proper multiplication
    md.add_initialized_data(f"t_vec_{i}", [[tx], [ty]])
    md.add_initialized_data(f"n_vec_{i}", [[nx], [ny]])
    
    # Axial term: EA * (du_tangent/ds) * (dv_tangent/ds)
    # du_tangent/ds = t^T * Grad_u * t (this gives a scalar)
    axial_term = f"EA*(t_vec_{i}'*Grad_u*t_vec_{i})*(t_vec_{i}'*Grad_Test_u*t_vec_{i})"
    
    # Bending term
    bending_term = f"EI*Grad_theta.Grad_Test_theta"
    
    # Shear term: kGA * (dw/ds - theta) * (dv/ds - psi)
    # where dw/ds = n^T * Grad_u * t (scalar)
    shear_term = f"kGA*((n_vec_{i}'*Grad_u*t_vec_{i}) - theta)*((n_vec_{i}'*Grad_Test_u*t_vec_{i}) - Test_theta)"
    
    md.add_linear_term(mim, axial_term, region)
    md.add_linear_term(mim, bending_term, region)
    md.add_linear_term(mim, shear_term, region)

############## Boundary Conditions ########################################################
# Fix displacement and rotation at the left side
md.add_Dirichlet_condition_with_multipliers(mim, "u", mfu, DIRICHLET_BOUNDARY)
md.add_Dirichlet_condition_with_multipliers(mim, "theta", mftheta, DIRICHLET_BOUNDARY)

# Option 1: Apply vertical force
md.add_initialized_data('F_vertical', [0, F])  # [Fx, Fy]
md.add_source_term_brick(mim, 'u', 'F_vertical', NEUMANN_BOUNDARY)


############### Solve and Export #############################################
md.solve()


# Get results
u = md.variable("u")
theta = md.variable("theta")

# Separate x and y displacements
u_x = u[0::2]  # x-displacements
u_y = u[1::2]  # y-displacements

print(f"Maximum displacement: {max(abs(u_y))} (cm)")

# Export
mfu.export_to_vtk("timoshenko_disp.vtk", mfu, u, 'Displacements')
mftheta.export_to_vtk("timoshenko_rot.vtk", mftheta, theta, 'Rotations')