import getfem as gf
import numpy as np
import matplotlib.pyplot as plt

# Parameters need to change according to the Dragonfly parameters:
E = 70e3  # Young's Modulus in MPa
H = 5 # Height in mm
B = 5 # Base in mm
A = B * H  # Cross-sectional area in m²
I = H*(B**3)/12  # Second moment of inertia in mm⁴
L = 100 # Beam length in mm
nu = 0.3    # Poisson ratio
F = -1   # Force at the right Boundary in N
n_dof = 3

############## 1D Mesh #####################################################

# Define the x-coordinates of the Dragonfly wing (nodes in 1D)
x_coords = np.array([
    0, 0.09, 0.14, 0.19, 0.24, 0.29, 0.34, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1
])

x_coords = x_coords*L

# Create the mesh with only x-coordinates (1D mesh)
Mesh = gf.Mesh('empty', 1)  # 1D Mesh, because only x-coordinates are needed
ind = []
for point in x_coords: 
    ind.append(Mesh.add_point([point]))  # We only need x-coordinates, and we initialize y = 0 for all nodes

# Now connect the points with 1D finite elements (1D mesh)
GT = gf.GeoTrans('GT_PK(1,1)')  # 1D element in reference space
for i in range(len(ind)-1):
    Mesh.add_convex(GT, np.array([ind[i], ind[i+1]]).T)

############## Boundary Identification #########################################################

# Defining different regions of the mesh
Nodes_left = Mesh.pid_from_coords([x_coords[0]])  # Left boundary node
Nodes_right = Mesh.pid_from_coords([x_coords[-1]])  # Right boundary node

# Faces for the left and right boundary
fleft = Mesh.faces_from_pid(Nodes_left)  # Left face
fright = Mesh.faces_from_pid(Nodes_right)  # Right face

# Region assignment for boundary conditions
NEUMANN_BOUNDARY = 1
DIRICHLET_BOUNDARY = 2
Mesh.set_region(NEUMANN_BOUNDARY, fright)
Mesh.set_region(DIRICHLET_BOUNDARY, fleft)

############## Finite Element / Integration #################################

mfu = gf.MeshFem(Mesh, 1)  # Finite element for axial displacement (1D)
# mfu.set_fem(gf.Fem('FEM_HERMITE(1)'))
mfu.set_classical_fem(2)

mfv = gf.MeshFem(Mesh, 1)  # Finite element for transverse displacement (1D)
# mfv.set_fem(gf.Fem('FEM_HERMITE(1)'))
# mfv.set_fem(gf.Fem("FEM_PK(1,1)"))
mfv.set_classical_fem(2)

mftheta = gf.MeshFem(Mesh, 1)  # Finite element for bending angle (1D)
# mftheta.set_fem(gf.Fem('FEM_HERMITE(1)'))
mftheta.set_classical_fem(2)

mim = gf.MeshIm(Mesh, gf.Integ("IM_GAUSS1D(5)"))  # Integration method

############## Initializing the Model #############################

# Create a real model
md = gf.Model('real')

# Adding variables to the model (1D displacement)
md.add_fem_variable("u", mfu)  # axial displacement
md.add_fem_variable("v", mfv)  # transverse displacement (y-coordinate displacement)
md.add_fem_variable("theta", mftheta)  # bending angle (not used in this 1D analysis, but kept for completeness)

# Material constants
md.add_initialized_data("EA", E*A)
md.add_initialized_data("EI", E*I)

############## Weak Form for Curved Beam ####################################

# Axial energy term
md.add_linear_term(mim, "EA * Grad_u * Grad_Test_u")

# Bending energy term 
md.add_linear_term(mim, "EI * Grad(Grad_v) : Grad(Grad_Test_v)")

# Constraint: theta = dv/ds (penalty method)
constraint_penalty = 1e12
md.add_initialized_data("constraint_penalty", constraint_penalty)
md.add_linear_term(mim, "constraint_penalty * (theta - Grad_v) . (Test_theta - Grad_Test_v)")

############## Applied Forces ################################################

# Apply transverse force at the tip in global coordinates (converted to local)
md.add_initialized_data('F_global_y', F)

# Transform the global force to local coordinates and apply it at the boundary
md.add_source_term_brick(mim, 'v', 'F_global_y', NEUMANN_BOUNDARY)

############## Boundary Conditions ########################################################

# Fix the left side (Dirichlet boundary condition)
md.add_Dirichlet_condition_with_multipliers(mim, "u", 0, DIRICHLET_BOUNDARY)
md.add_Dirichlet_condition_with_multipliers(mim, "v", 0, DIRICHLET_BOUNDARY) 
md.add_Dirichlet_condition_with_multipliers(mim, "theta", 0, DIRICHLET_BOUNDARY)

# Define the y-displacement (initial displacement, assumed function of x)
# For simplicity, define a linear displacement in the y-direction as an example
y_displacement = np.array([0.0, 0.0, 0.055, 0.0, 0.055, 0.0, 0.0, 0.02, 0.06, 0.08, 0.085, 0.08, 0.07, 0.055, 0.04, 0.025])
y_displacement = y_displacement*L
print(y_displacement)

# # Applying initial y-displacement at the nodes (we're using v for y displacement)
dof_pts = mfv.basic_dof_nodes().reshape(-1)
dof_pts = dof_pts[::2]

# Print or compare
print("DOF coordinates:")
print(dof_pts)

print("x_coords:")
print(x_coords)

for i, x in enumerate(dof_pts):
    region_id = 200 + i
    pid = Mesh.pid_from_coords([x])
    face = Mesh.faces_from_pid(pid)
    Mesh.set_region(region_id, face)
    val = y_displacement[i] if i < len(y_displacement) else 0.0 
    md.add_initialized_data(f"y_disp_{i}", val)
    md.add_Dirichlet_condition_with_multipliers(mim, "v", mfv, region_id, f"y_disp_{i}")


############### Solve and Export #############################################

# Solve the linear system
md.solve()

# Access solution variables (displacements)

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

points = points *L

# Building the tangential vector to each element
Tvec = []
Nvec = []
# Access the nodes for each element
for i in range(len(points)-1):

    x_start = points[i][0]
    y_start = points[i][1]
    x_end = points[i+1][0]
    y_end = points[i+1][0]
    # Compute the tangent vector (Tvec) and normal vector (Nvec)
    dx = x_end - x_start
    dy = y_end - y_start
    length = np.sqrt(dx**2 + dy**2)  # Magnitude of the tangent vector
    Tvec.append(np.array([dx / length, dy / length]))  # Normalize the tangent vector

    Nvec.append(([-dy / length, dx / length]))  # Perpendicular to Tvec

Tvec.append(Tvec[-1])
Nvec.append(Nvec[-1])

U = []
V = []

for i, point in enumerate(points):
    v = md.interpolation("v", point[0], mfu.mesh())
    print(point)
    print(v)
    if v.size > 0 :
        disp = v * Tvec[i]
        print(disp)
        U.append(disp[0])
        V.append(disp[1])
    else:
        v
        U.append(0)
        V.append(0)

max_u = np.max(np.abs(U))
max_v = np.max(np.abs(V))
print(f"Maximum axial displacement: {max_u} mm")
print(f"Maximum transverse displacement: {max_v} mm")

print(U)
print(len(U))
print(V)

############### Plotting the solution #############################################
import matplotlib.pyplot as plt

# Original points
x0 = points[:, 0]  # scale to mm if needed
y0 = points[:, 1]

# Displacement (you already extracted these)
u_nodes = np.array(U)
v_nodes = np.array(V)

# Deformed points (apply scaling for visibility)
scale = 1  # visual scaling factor
x_disp = x0 + scale * u_nodes
y_disp = y0 + scale * v_nodes

# Plot setup
plt.figure(figsize=(10, 4))
plt.plot(x0, y0, color='white', linewidth=1, label="Original shape")
plt.plot(x_disp, y_disp, color='cyan', linewidth=1.5, label="Deformed shape")
plt.gca().set_facecolor('#2c2f3a')  # background color like your image
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.legend(loc="upper left")
plt.title("Deformation of Dragonfly Wing (Scaled)", color='white')
plt.savefig("dragonfly_deformation_1D_Coord.png", dpi=300, facecolor='#2c2f3a')

