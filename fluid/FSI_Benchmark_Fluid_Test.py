import getfem as gf 
from Functions import verify_regions, mesh_statistics_corrected
import numpy as np
import os

print("Number of elements in the fluid mesh:")
π = np.pi
################################
# Beam in a Fluid:
#################################
'''
The goal is to solve the full Navier-Stokes equations for a beam immeresed in a fluid.
'''

##################
## PROBLEM DATA ##
##################
"""
The fluid will be the air
"""

scale_factor = 1 #m -> m  (it can be used if we want to pass from meter to anything else cm, dm etc...) 


# Fluid properties (air):
ν_fluid = 0.015e-4 * scale_factor**2                # m²/s 
rho_fluid = 1.23 /(scale_factor**3  )               # kg/m³ 



# Boundaries values: 
# Inlet velocity parameters
U_mean = 1.0     # Mean inlet velocity
H = 0.41           # Channel height

# Transient paramters: 
T = 10.0           # Total simulation time
dt = 1e-3         # Time step
theta = 0.5      # Theta parameter (0.5 = Crank-Nicolson)



#############
## MESH ##
#############

Mesh_fluid= gf.Mesh('Import', 'gmsh','fluid/cylinder.msh')
#Mesh_fluid.export_to_vtk('MESH_GMSH/Fluid.vtk') # Export mesh to VTK for visualization
print("Number of elements in the fluid mesh:")
mesh_statistics_corrected(Mesh_fluid,name="Mesh")

#############
## REGIONS ##
#############
"""
regions flagging in gmsh does not work thus it's necessary to give a region for each line or physical surface. 
 
"""

Beam_left = 1 
Beam_bottom = 2 
Beam_right = 3 
Beam_top = 4 
INLET = 5 
Wall_bottom = 6
OUTLET = 7  
Wall_top = 8 

BEAM_INTERFACE = 100

Mesh_fluid.region_merge(BEAM_INTERFACE, Beam_left)
Mesh_fluid.region_merge(BEAM_INTERFACE, Beam_right)
Mesh_fluid.region_merge(BEAM_INTERFACE, Beam_bottom)
Mesh_fluid.region_merge(BEAM_INTERFACE, Beam_top)

# physical surface: 
fluid1 = 9
fluid2 = 11
fluid3 = 10
fluid4 = 13


FLUID = 101
Mesh_fluid.region_merge(FLUID,fluid1)
Mesh_fluid.region_merge(FLUID,fluid2)
Mesh_fluid.region_merge(FLUID,fluid3)
Mesh_fluid.region_merge(FLUID,fluid4)

WALLS  = 102
Mesh_fluid.region_merge(WALLS,Wall_top)
Mesh_fluid.region_merge(WALLS,Wall_bottom)

#verify_regions(Mesh_solid, 'meshsolid')
########################
## INTEGRATION METHOD ##
########################
"""
The integration method is quadrature with 17 points, which is suitable for QK elements.  
"""
mim_fluid = gf.MeshIm(Mesh_fluid, gf.Integ('IM_QUAD(7)'))


#########################
## FEM ELEMENTS METHOD ##
########################1
"""
THe meshes are quadrilater, hence QK elements are used. 
"""

## FEM ELEMENTS:

mfv_fluid_ = gf.MeshFem(Mesh_fluid, 2)
mfv_fluid_.set_fem(gf.Fem('FEM_QK(2,2)'))

mfp_fluid = gf.MeshFem(Mesh_fluid, 1)
mfp_fluid.set_fem(gf.Fem('FEM_QK(2,1)'))

######################
## REMOVE FIXED DOF ##
######################
"""
Instead of applying dirichelt boundary coniditons, the degree of freedom are removed directly here.
"""

kept_dofs_v_f = list(
                set(range(mfv_fluid_.nbdof()))
                -set(mfv_fluid_.basic_dof_on_region(WALLS))
                -set(mfv_fluid_.basic_dof_on_region(BEAM_INTERFACE))
                )
                

mfv_fluid = gf.MeshFem('partial', mfv_fluid_, kept_dofs_v_f)


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
md.add_macro("inlet_profile(y)", "1.5*U_mean*4*y*(H-y)/(H*H)")


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
ramp_factor = 0.0
V_inlet_expr = f"{ramp_factor}*[inlet_profile(X(2)), 0]"
V_inlet_full = md.interpolation(V_inlet_expr, mfv_fluid_)
V_inlet = V_inlet_full[kept_dofs_v_f]

md.add_initialized_fem_data('V_inlet', mfv_fluid, V_inlet) 

# Replace penalty terms with:
md.add_Dirichlet_condition_with_penalization(mim_fluid, "v_f", 10**3, INLET, "V_inlet")
md.add_Dirichlet_condition_with_penalization(mim_fluid, 'p', 10**3, OUTLET)

####################
## MODEL SOLOTION ##
####################

# Create output directory
output_dir = "Results"
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
    
