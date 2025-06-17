# 2D Dragonfly Wing using 

# import basic modules
import getfem as gf
import numpy as np

# Paramters need to change according to the Dragonfly paramters:

# geometric paramters:
L = 0.1
thickness = 0.005*L 

E = 70e9 # Young Modulus  ---> 70 Gpa
nu = 0.33      # Poisson ratio
clambda = E*nu/(1-nu**2)                # First Lame coefficient (N/cm^2)
cmu = E/(2*(1+nu))                              # Second Lame coefficient (N/cm^2)

F = -0.01/thickness**2  # Force at the right Boundary in N/m

############## Mesh #########################################################

# Importing the mesh from Gmsh
Mesh = gf.Mesh('import', 'gmsh', 'dragonfly.msh')

# Scale the mesh coordinates to physical dimensions
# This transforms the normalized mesh (0-1) to actual size (0-0.1 m)
pts = Mesh.pts()
pts_scaled = pts * L  # Scale all coordinates by L
Mesh.set_pts(pts_scaled)

############## Finite Element / Integration #################################

mu_ = gf.MeshFem(Mesh,2)    # the first input is the mesh the second input is the order of the field to be solved

mu_.set_fem(gf.Fem('FEM_PK(2,2)'))  # P_2 methods on triangles of dimension 2

mim = gf.MeshIm(Mesh,gf.Integ('IM_TRIANGLE(4)'))   # Integration method: Integratio on triangles

############## Boundary Identification #########################################################

LEADING = 10
TRAILING = 20
TOP = 30
BOTTOM = 40

############## Boundary Conditions ########################################################

#Fix the left side (leading edge) - clamp both axial and transverse displacement
# This represents a clamped boundary condition
# reduced problem by eliminating the DOF of the clamped end (leading edge)
kept_dofs = list(
                set(range(mu_.nbdof()))
                -set(mu_.basic_dof_on_region(LEADING)))

mu = gf.MeshFem('partial', mu_, kept_dofs)

############## The model (weak form of the PDE) #############################

# empty real model
md = gf.Model('real')

md.add_fem_variable("u", mu)  # u = [u_x, u_y] displacement vector

md.add_initialized_data('cmu', [cmu])
md.add_initialized_data('clambda', [clambda])

md.add_isotropic_linearized_elasticity_brick(mim, 'u', 'clambda', 'cmu') # Plane stress considered for thin wing with small cross-section

# Adding a vertical force on the RHS, trailing edge
md.add_initialized_data('ForceData',[0,F])
md.add_source_term_brick(mim, 'u', 'ForceData', TRAILING)


############### Solve and Export #############################################

# solve the linear system
md.solve()

# main unknown - 2D displacement field
U = md.variable("u")

# Separate x and y displacements
u_x = U[0::2]  # x-displacements
u_y = U[1::2]  # y-displacements

print(f"Maximum displacement: {max(abs(u_y)):.6e} meters")
mu.export_to_vtk('2D_dragonfly.vtk', mu, U, 'Displacements')
