import getfem as gf 
import numpy as np
import os
from Functions import verify_regions, mesh_statistics_corrected




############################{####
# Cylinder in a Fluid:
#################################
'''
The goal is to solve the full Navier-Stokes equations for a cylinder immeresed in a fluid.
'''

##################
## PROBLEM DATA ##
##################
"""
The fluid will be the air
"""

scale_factor = 1 #m -> m  (it can be used if we want to pass from meter to anything else cm, dm etc...) 


# Fluid properties (air):
#ν_fluid = 1.516e-5 * scale_factor**2                # m²/s 
ν_fluid = 1.16667e-3 * scale_factor**2  
rho_fluid = 1.204 /(scale_factor**3  )               # kg/m³ 



# Boundaries values: 
# Inlet velocity parameters
U_mean = 0.1       # Mean inlet velocity m/s
H = 0.2          # Channel height m

# Transient paramters: 
T = 10.0           # Total simulation time
dt = 1e-3         # Time step
theta = 0.5      # Theta parameter (0.5 = Crank-Nicolson)


print(f"Reynolds number is: {U_mean*0.7/ν_fluid}")############
## MESH ##
#############

Mesh_fluid= gf.Mesh('Import', 'gmsh','fluid/cylinder.msh')
Mesh_fluid.export_to_vtk('fluid/Fluid.vtk') 
mesh_statistics_corrected(Mesh_fluid,name="Mesh")

#############
## REGIONS ##
#############
"""
regions flagging in gmsh does not work thus it's necessary to give a region for each line or physical surface. 
 
"""

Bottom_left = 1 
Bottom_right = 2
Top_left = 3
Top_right = 4 
INLET = 5 
OUTLET = 7  
Cylinedr_1 = 8
Cylinder_2 = 9
Cylinder_3 = 10
Cylinder_4 = 11

CYLINDER_INTERFACE = 100

Mesh_fluid.region_merge(CYLINDER_INTERFACE, Cylinedr_1)
Mesh_fluid.region_merge(CYLINDER_INTERFACE, Cylinder_2)
Mesh_fluid.region_merge(CYLINDER_INTERFACE, Cylinder_3)
Mesh_fluid.region_merge(CYLINDER_INTERFACE, Cylinder_4)

# physical surface: 
fluid1 = 6 #to check
fluid2 = 12
fluid3 = 13
fluid4 = 14
fluid5 = 15

FLUID = 101
Mesh_fluid.region_merge(FLUID,fluid1)
Mesh_fluid.region_merge(FLUID,fluid2)
Mesh_fluid.region_merge(FLUID,fluid3)
Mesh_fluid.region_merge(FLUID,fluid4)
Mesh_fluid.region_merge(FLUID,fluid5)

WALLS  = 102
Mesh_fluid.region_merge(WALLS,Bottom_left)
Mesh_fluid.region_merge(WALLS,Bottom_right)
Mesh_fluid.region_merge(WALLS,Top_left)
Mesh_fluid.region_merge(WALLS,Top_right)

#verify_regions(Mesh_fluid, 'fluid/meshfluid')

########################
## INTEGRATION METHOD ##
########################
"""
The integration method is quadrature with 7 points, which is suitable for QK elements.  
"""
mim_fluid = gf.MeshIm(Mesh_fluid, gf.Integ('IM_QUAD(7)'))


#########################
## FEM ELEMENTS METHOD ##
########################1
"""
THe meshes are quadrilater, hence QK elements are used. 
"""

## FEM ELEMENTS:

mfv_fluid = gf.MeshFem(Mesh_fluid, 2)
mfv_fluid.set_fem(gf.Fem('FEM_QK(2,2)'))

mfp_fluid = gf.MeshFem(Mesh_fluid, 1)
mfp_fluid.set_fem(gf.Fem('FEM_QK(2,1)'))


###########
## MODEL ##
###########

md = gf.Model("real")


###################
## FEM VARIABLES ##
###################
"""
Fem variable are defined on the MeshFEM with removed degree of freedom. 

"""


md.add_fem_variable("v_f", mfv_fluid)       # Fluid velocity  
md.add_fem_variable("p", mfp_fluid)       # Pressure

# Previous time step data
md.add_fem_data("Previous_v_f", mfv_fluid)


###########################
## INITIALIZED CONSTANTS ##
###########################
"""
The material properties and the times variable are initiualized in the model 
"""
md.add_initialized_data("rho_f", rho_fluid)
md.add_initialized_data("nu_f", ν_fluid)
md.add_initialized_data("dt", dt)
md.add_initialized_data("theta", theta)
md.add_initialized_data("H", H)
md.add_initialized_data("U_mean", U_mean)
md.add_initialized_data("p_out", 0)
# Time variable
md.add_initialized_data("t", 0.0)
# Define inlet profile as a macro (parabolic profile)


########################
## INITIAL CONDITIONS ##
########################
"""
Initialize all to zero (at rest), since we're using filtered fem variable, it's not possibble to 
use md.interpolation. 
"""

# Initialize with zero vectors of the correct size
md.set_variable("v_f", np.zeros(mfv_fluid.nbdof()))
md.set_variable("p", np.zeros(mfp_fluid.nbdof()))

# Also set previous values
md.set_variable("Previous_v_f", np.zeros(mfv_fluid.nbdof()))

######################
## WEAK FORMULATION ##
# ######################
"""
The problem is formulated in weak form, and for the time the theta method is used.
"""


# FLUID : 
# stress tensors: 
md.add_macro('Stress_vu(v)', "rho_f*nu_f*2*Sym(Grad_v) ")
md.add_macro('Stress_p(p)', '-p*Id(2)')
md.add_macro("Convection(v)", "(rho_f*v.Grad_v)")
md.add_macro('Incompressibility(v)', "Trace(Grad_v)")


Transient_fluid = 'rho_f*(v_f - Previous_v_f). Test_v_f' \
            
md.add_nonlinear_term(mim_fluid, Transient_fluid, FLUID)


md.add_linear_term(mim_fluid, ' dt*Incompressibility(v_f)*Test_p', FLUID) 

md.add_nonlinear_term(mim_fluid,"dt*theta*(Convection(v_f)).Test_v_f +" \
 "dt*(1-theta)*(Convection(Previous_v_f)).Test_v_f", FLUID)

md.add_linear_term(mim_fluid,"dt*theta*Stress_vu(v_f):Grad_Test_v_f +" \
                                "dt*(1-theta)*(Stress_vu(Previous_v_f):Grad_Test_v_f)", FLUID)
md.add_linear_term(mim_fluid, 'dt*Stress_p(p):Grad_Test_v_f', FLUID)




#####################
# BOUNDARY CONDITIONS
# Inlet velocity profile (parabolic, time-dependent with smooth ramp)

# DIRICHLET CONDITIONS
md.add_Dirichlet_condition_with_multipliers(mfv_fluid, WALLS, "v_f", [U_mean, 0], FLUID)  
md.add_Dirichlet_condition_with_multipliers(mfv_fluid, INLET, "v_f", [U_mean, 0], FLUID)  
md.add_Dirichlet_condition_with_multipliers(mfv_fluid, CYLINDER_INTERFACE, "v_f", [0, 0], FLUID)  


####################
## MODEL SOLOTION ##
####################

# Create output directory
output_dir = "fluid/Results_fluid"
os.makedirs(output_dir, exist_ok=True)

# TIME STEPPING LOOP

print(f"Time step: {dt}, Total time: {T}")
print(f"Theta: {theta} (Crank-Nicolson)" if theta == 0.5 else f"Theta: {theta}")

t = 0.0
step = 0
export_every = int(0.01/dt)  # Export every 0.01s



# Main time loop
while t < T:
    
    md.set_variable("t", t)
    if t < 2.0:
        ramp_factor = 0.5 * (1 - np.cos(np.pi * t / 2.0))
    else:
        ramp_factor = 1.0
    V_inlet_expr = f"{ramp_factor}*[inlet_profile(X(2)), 0]"
    V_inlet_full = md.interpolation(V_inlet_expr, mfv_fluid_)
    V_inlet = V_inlet_full[kept_dofs_v_f]
    md.set_variable("V_inlet", V_inlet)

    if step > 0:  # After first solve
        print(f'time is: {t} and ramp_factor is {ramp_factor} and velocity is {V_inlet}')
    

    check_mesh_quality(Mesh_fluid, U_mean, 0.07071078, ν_fluid, dt)
    # More robust solver parameters
    md.solve("noisy", 
         "max_iter", 100,
         "max_res", 1e-6,  
         "lsolver", "superlu",  
         "alpha min", 1e-4,  
         "alpha mult", 0.5)    

    # Extract current solution

    v_f = md.variable("v_f")
    p = md.variable("p")
   
    
    mfv_fluid.export_to_vtu(f"{output_dir}/fluid_{step:05d}.vtu",
                            mfv_fluid, v_f, "Velocity", 
                            mfp_fluid, p, "Pressure")
    
    
    md.set_variable("Previous_v_f", v_f)
    
    
    # Advance time
    t += dt
    step += 1
    
