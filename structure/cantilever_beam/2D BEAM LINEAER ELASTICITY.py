import getfem as gf
import numpy as np
import os
import json
# geometric paramters:
L = 10
thickness = 0.005*L

# Material properties
E = 70e5                                        # Young Modulus  ---> 70 Gpa
nu = 0.33                                       # Poisson ratio
clambda = E*nu/((1+nu)*(1-nu))                # First Lame coefficient (N/cm^2)
cmu = E/(2*(1+nu))                              # Second Lame coefficient (N/cm^2)

# Force
F = - 0.01 /(thickness**2)                                     # Force at the right Boundary (N)



# IMPORT OF THE DRAGONFLY GMSH 
# Note: it's possible to automated the refinement of the mesh by using gmsh on python 
# look into that by using GMSH api 

m = gf.Mesh('Import','gmsh','Beam.msh')
# #set region of the mesh: 
# # left_face = m.outer_faces_with_direction([-1., 0.], 0.01) # Left boundary
# # rigt_face = m.outer_faces_with_direction([1,0], 0.01) # right boundary 
# # LEFT = 40
# # RIGHT = 20  
# # m.set_region(LEFT, left_face)
# # m.set_region(RIGHT, rigt_face)
Beam_left = 1 
Beam_bottom = 2 
Beam_right = 3 
Beam_top = 4 

# selection of finite element
mu_ = gf.MeshFem(m,2)
#mu.set_fem(gf.Fem('FEM_PK(2,1)'))
mu_.set_fem(gf.Fem('FEM_QK(2,5)'))
#mu.set_fem(gf.Fem("FEM_Q2_INCOMPLETE(2)"))
# slection of integration method
#mim = gf.MeshIm(m, gf.Integ('IM_TRIANGLE(9)'))
mim = gf.MeshIm(m, gf.Integ('IM_QUAD(17)'))

kept_dofs = list(
                set(range(mu_.nbdof()))
                -set(mu_.basic_dof_on_region(Beam_left))
        
                 )

mu= gf.MeshFem('partial', mu_, kept_dofs)
# MODEL
md = gf.Model("real")
md.add_fem_variable("u",mu)
md.add_initialized_data('cmu', [cmu])
md.add_initialized_data('clambda', [clambda])

md.add_macro("eps(u)", "0.5*(Grad_u + Grad_u')")
md.add_macro("sigma(u)", "2*cmu*eps(u) + clambda*Id(2)*Div_u")

#md.add_isotropic_linearized_elasticity_brick(mim, 'u', 'clambda', 'cmu')

#md.add_linear_term(mim, "(cmu * (Grad_u + Grad_u') : Grad_Test_u + clambda * Div_u * Div_Test_u)") # wrong
# md.add_linear_generic_assembly_brick(mim,"[Grad_Test_u(1,1),Grad_Test_u(2,2),Grad_Test_u(1,2)+Grad_Test_u(2,1)]'." \
# "[[clambda + 2*cmu, clambda, 0],[clambda,clambda + 2*cmu,0],[0, 0, cmu]]."
# "[Grad_u(1,1),Grad_u(2,2),Grad_u(1,2)+Grad_u(2,1)]")
md.add_linear_term(mim, "sigma(u):eps(Test_u)")
md.add_initialized_data('ForceData',[0,F])
#md.add_initialized_data('ForceData',[10,0])

md.add_source_term_brick(mim, 'u', 'ForceData', Beam_right)



# BOUNDARY CONDITIONS:

#md.add_Dirichlet_condition_with_multipliers(mim,'u', mu, Beam_left)


md.solve()

##Print the stiffness matrix
#print(md.brick_list())
#print('stiffness_matrix',md.matrix_term(0,0))

U = md.variable("u")
# Separate x and y displacements
u_x = U[0::2]  # x-displacements
u_y = U[1::2]  # y-displacements


print(f"Maximum displacement: {max(abs(u_x))} (cm)")
print(f"Maximum displacement: {max(abs(u_y))} (cm)")
if max(abs(u_y))/L > 0.1:
    print("WARNING: Large deflection - linear theory may not be valid")

# FolderName = "LinearElasticityResults"
# os.system(f'mkdir {FolderName}')
# SaveFile = os.path.join(FolderName,'2d_BEAM_flux_4.vtk')
# print(gf.compute_error_estimate(mu,U,mim))
# mu.export_to_vtk(SaveFile,  mu, U, 'Displacements')

