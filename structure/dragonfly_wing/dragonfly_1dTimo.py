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

# creating the mesh with all the wing's node points
Mesh = gf.Mesh('empty', 2)
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
# Test with beam code
mfu = gf.MeshFem(Mesh, 2)  # Finite element for the elastic displacement
mfu.set_fem(gf.Fem("FEM_PK(1,5)"))
 
mim = gf.MeshIm(Mesh, gf.Integ("IM_GAUSS1D(17)")) # Integration method  

############## Initialising the model #############################

# empty real model
md = gf.Model('real')

# declare that the unknowns of the system on the finite element method "mfu"
md.add_fem_variable("u", mfu)        # Displacement [w, theta]

# Adding a constant scalar value
md.add_initialized_data("kGA", k*G*A)
md.add_initialized_data("EI", E*I)

############## Weak Form for Timoshenko Beam ####################################

# Bending energy: EI * dtheta/dx * d(delta_theta)/dx
bending_term = "EI * Grad_u(2,1) * Grad_Test_u(2,1)"

# Shear energy: kGA * (dw/dx - theta) * (d(delta_w)/dx - delta_theta)
shear_term = "kGA * (Grad_u(1,1) - u(2)) * (Grad_Test_u(1, 1) - Test_u(2))"

# Add the complete bilinear form
md.add_linear_term(mim, bending_term + " + " + shear_term)

############## Applied Forces ################################################

# Apply transverse force at the tip in global coordinates
# Need to transform to local normal direction
md.add_initialized_data('F_global', [F, 0])

# Transform global force to local coordinates at the boundary
# F_local_normal = F_global · normal_vector
md.add_source_term_brick(mim, 'u', 'F_global', NEUMANN_BOUNDARY)

############## Boundary Conditions ########################################################

# Fix the left side
md.add_Dirichlet_condition_with_multipliers(mim, "u", 2, DIRICHLET_BOUNDARY)

############### Solve and Export #############################################

# solve the linear system
md.solve()

# main unknown
U = md.variable("u")

max_u = np.max(np.abs(U[0::2]))
print(f"Maximum transverse displacement: {max_u} cm")

# Extract displacements and rotations
w = U[0::2]  # Transverse displacements
theta = U[1::2]  # Rotations

U_2D = np.zeros((len(w), 2))
U_2D[:, 1] = w  # Transverse displacement in y-direction
U_2D_flat = U_2D.flatten()

mfu_2d = gf.MeshFem(Mesh, 2)  # 2D displacement field
mfu_2d.set_fem(gf.Fem("FEM_PK(1,5)"))
mfu_2d.export_to_vtk("Dragonfly_Timo.vtk", U_2D_flat, "Displacement")