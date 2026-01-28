import getfem as gf 
from Functions import verify_regions
import numpy as np
import os

gf.util_trace_level(1)
gf.util_warning_level(1)
π = np.pi


##################
## PROBLEM DATA ##
##################
"""
The values are taken form Turek hann paper
"""



# Artificial values for test purpose: 
# Solid properties:
E = 1e+2                        #kg/cms^2
nu_solid  = 0.42
clambda = E*nu_solid/((1+nu_solid)*(1-2*nu_solid))     # First Lame coefficient (N/cm^2)
cmu = E/(2*(1+nu_solid))                             # Second Lame coefficient (N/cm^2)
rho_solid = 1000                                    # kg/cm^3
mu_solid =  100                                   # kg/(cm*s^2)

# Fluid propeties: 
mu_fluid = 0.001                                   #cm^2/s
rho_fluid = 1                                       #kg/cm^3

 
# Boundaries values: 
# Inlet velocity parameters
U_mean = 1.0     # Mean inlet velocity
H = 0.41           # Channel height

# Transient paramters: 
T = 10.0           # Total simulation time
dt = 1e-5        # Time step
theta = 0.5      # Theta parameter (0.5 = Crank-Nicolson)



#############
## MESH ##
#############

Mesh = gf.Mesh('Import', 'gmsh', 'FSI/MESH/TF_1MESH_tri.msh')
#############
## REGIONS ##
#############
"""
defining region takes a bit of code, because regions flagging in gmsh
is not currently working, hence it give a region for each line or physical surface. 
Thanks to the function verify region it's possible to visulaize in paraview all the regions
and correctly merging them toghter. 
"""


INLET = 201
OUTLET = 202
WALLS = 203
CYLINDER = 204
BEAM_LEFT = 205
BEAM_INTERFACE = 206

FLUID = 207
BEAM = 208

Mesh.region_merge(FLUID, 28) 
Mesh.region_merge(FLUID, 29)
Mesh.region_merge(FLUID, 30)
Mesh.region_merge(FLUID, 31)
Mesh.region_merge(FLUID, 32)
Mesh.region_merge(FLUID, 33)
Mesh.region_merge(FLUID, 34)
Mesh.region_merge(FLUID, 35)
Mesh.region_merge(FLUID, 36)
Mesh.region_merge(FLUID, 37)
Mesh.region_merge(FLUID, 38)
Mesh.region_merge(FLUID, 39)
Mesh.region_merge(FLUID, 40)
Mesh.region_merge(FLUID, 41)
Mesh.region_merge(FLUID, 42)
Mesh.region_merge(FLUID, 43)
Mesh.region_merge(FLUID, 44)
Mesh.region_merge(FLUID, 46)
Mesh.region_merge(FLUID, 47)

Mesh.region_merge(BEAM, 45)
Mesh.region_merge(BEAM, 51)

# From 1,2,3,4, top wall 10,11,12,13 bottom wall
Mesh.region_merge(WALLS, 1)
Mesh.region_merge(WALLS, 10)
Mesh.region_merge(WALLS,11 )
Mesh.region_merge(WALLS,12 ) 
Mesh.region_merge(WALLS, 13)
Mesh.region_merge(WALLS, 2)
Mesh.region_merge(WALLS, 3)
Mesh.region_merge(WALLS, 4)

Mesh.region_merge(CYLINDER, 17)
Mesh.region_merge(CYLINDER, 18)
Mesh.region_merge(CYLINDER, 19)
Mesh.region_merge(CYLINDER, 20)
Mesh.region_merge(CYLINDER, 21)


Mesh.region_merge(BEAM_INTERFACE, 23 )
Mesh.region_merge(BEAM_INTERFACE,24 )
Mesh.region_merge(BEAM_INTERFACE, 25) 
Mesh.region_merge(BEAM_INTERFACE, 26)
Mesh.region_merge(BEAM_INTERFACE, 27)

Mesh.region_merge(BEAM_LEFT, 22) # this interface is with the cylinder


Mesh.region_merge(OUTLET, 5)
Mesh.region_merge(OUTLET, 6)
Mesh.region_merge(OUTLET, 7)
Mesh.region_merge(OUTLET, 8)
Mesh.region_merge(OUTLET, 9)

Mesh.region_merge(INLET, 14)
Mesh.region_merge(INLET, 15)
Mesh.region_merge(INLET, 16)

#verify_regions(Mesh) #it can be used verify regions againg to see if the regions are set up correctly.


########################
## INTEGRATION METHOD ##
########################
"""
Only one mesh is being used hence there is only one 
mesh integration method. However, it's possible to do this with two meshes one for fluid
and one for solid.  
"""
mim = gf.MeshIm(Mesh, 5)


#########################
## FEM ELEMENTS METHOD ##
########################

mfu_fluid = gf.MeshFem(Mesh, 2 )
mfu_fluid.set_fem(gf.Fem('FEM_PK(2,2)'))

mfv_fluid = gf.MeshFem(Mesh, 2)
mfv_fluid.set_fem(gf.Fem('FEM_PK(2,2)'))

mfp_fluid = gf.MeshFem(Mesh, 1)
mfp_fluid.set_fem(gf.Fem('FEM_PK(2,1)'))

mfu_solid = gf.MeshFem(Mesh, 2)
mfu_solid.set_fem(gf.Fem('FEM_PK(2,2)'))


mfv_solid= gf.MeshFem(Mesh, 2)
mfv_solid.set_fem(gf.Fem('FEM_PK(2,2)'))

###########
## MODEL ##
###########

md = gf.Model("real")


###################
## FEM VARIABLES ##
###################
"""
Filtered fem variable are used to only define the 
fem variable on the part of the mesh of interest. 

"""

md.add_filtered_fem_variable("u_f", mfu_fluid, FLUID)     # Fluid mesh displacement
md.add_filtered_fem_variable("v_f", mfv_fluid, FLUID)      # Fluid velocity  
md.add_filtered_fem_variable("p", mfp_fluid, FLUID)       # Pressure
md.add_filtered_fem_variable("u_s", mfu_solid, BEAM)     # Solid displacement
md.add_filtered_fem_variable("v_s",mfv_solid, BEAM)

# Previous time step data
md.add_filtered_fem_variable("Previous_u_f", mfu_fluid, FLUID)
md.add_filtered_fem_variable("Previous_v_f", mfv_fluid, FLUID)
md.add_filtered_fem_variable("Previous_u_s", mfu_solid, BEAM)
md.add_filtered_fem_variable("Previous_v_s", mfv_solid, BEAM) 

# Lagrange Multiplier
md.add_filtered_fem_variable("mult", mfu_fluid, BEAM_INTERFACE)
md.add_filtered_fem_variable("mult_v",  mfv_fluid, BEAM_INTERFACE)

###########################
## INITIALIZED CONSTANTS ##
###########################

md.add_initialized_data("rho_f", rho_fluid)
md.add_initialized_data("mu_f", mu_fluid)
md.add_initialized_data("lambda", clambda)
md.add_initialized_data("mu_s", cmu)
md.add_initialized_data("rho_s", rho_solid)
md.add_initialized_data("dt", dt)
md.add_initialized_data("theta", theta)
md.add_initialized_data("H", H)
md.add_initialized_data("U_mean", U_mean)
md.add_initialized_data("p_out", 0)



# Time variable
md.add_initialized_data("t", 0.0)


########################
## INITIAL CONDITIONS ##
########################
"""
Initialize all to zero (at rest), since we're using filtered fem variable, it's not possibble to 
use md.interpolation. 
"""

# Get the DOF indices for each region

fluid_u_dofs = mfu_fluid.basic_dof_on_region(FLUID)
fluid_v_dofs = mfv_fluid.basic_dof_on_region(FLUID)
fluid_p_dofs = mfp_fluid.basic_dof_on_region(FLUID)
solid_u_dofs = mfu_solid.basic_dof_on_region(BEAM)
solid_v_dofs = mfv_solid.basic_dof_on_region(BEAM)

# Interpolate on full mesh and extract region DOFs
u_f_full = md.interpolation("[0,0]", mfu_fluid)
v_f_full = md.interpolation("[0,0]", mfv_fluid)
p_full = md.interpolation("0", mfp_fluid)
u_s_full = md.interpolation("[0,0]", mfu_solid)
v_s_full = md.interpolation("[0,0]", mfv_solid)

# Set variables with extracted DOFs
md.set_variable("u_f", u_f_full[fluid_u_dofs])
md.set_variable("v_f", v_f_full[fluid_v_dofs])
md.set_variable("p", p_full[fluid_p_dofs])
md.set_variable("u_s", u_s_full[solid_u_dofs])
md.set_variable("v_s", v_s_full[solid_v_dofs])

# Set previous values
md.set_variable("Previous_u_f", u_f_full[fluid_u_dofs])
md.set_variable("Previous_v_f", v_f_full[fluid_v_dofs])
md.set_variable("Previous_u_s", u_s_full[solid_u_dofs])
md.set_variable("Previous_v_s", v_s_full[solid_v_dofs])

######################
## WEAK FORMULATION ##
# ######################
"""
The weak form is taken from Thomas Wick FSi lecture notes, equation 7.26.
It uses full N-S and elasticity with stvk materials in ALE framework.
Moreover, the time integration method is a theta method.  
"""
# MACROS: 
md.add_macro("F(u)", "Id(2)+Grad(u)")                 # deformation gradient
md.add_macro("J(u)", "Det(F(u))")                     # Volume change ratio

# FLUID : 
# stress tensors: 
md.add_macro('sigma_f_vu(v,u)', "2*mu_f*(Grad_v*Inv(F(u)) + (Inv(F(u)))' * (Grad_v)') ")
md.add_macro('sigma_f_p(p)', '-p*Id(2)')

md.add_macro("Convection(u,v)", "rho_f*J(u)*((Inv(F(u))*v).Grad_v)")
md.add_macro('Mesh_def(u)',"J(u)*Grad(u)" )
md.add_macro('Incompressibility(u,v)', "Trace(Inv(F(u))*Grad_v)")
md.add_macro("Stress_vu(u,v)", "J(u)*sigma_f_vu(v, u)*Inv(F(u))'")
md.add_macro("Stress_p(u,p)", "J(u)*sigma_f_p(p)*Inv(F(u))'")

#SOLID : 
md.add_macro('E(u)', "0.5*(F(u)'*F(u)- Id(2))")
#md.add_macro('E(u)', "0.5*(Grad_u + Grad_u')")
md.add_macro('Sigma(u)', 'lambda*Trace(E(u))*Id(2) + 2*mu_s*E(u)')



#A_T in wick notes
Transient_fluid = 'rho_f*((theta*J(u_f) + (1-theta)*J(Previous_u_f)) *(v_f - Previous_v_f). Test_v_f) -' \
            'rho_f*( (J(u_f)*Inv(F(u_f))*(u_f-Previous_u_f).Grad_v_f). Test_v_f)'

Transient_solid = "rho_s*(v_s - Previous_v_s).Test_v_s + " \
            "(u_s - Previous_u_s).Test_u_s"

md.add_nonlinear_term(mim, Transient_fluid, FLUID)
md.add_nonlinear_term(mim, Transient_solid,BEAM)

# A_I in wick notes
md.add_nonlinear_term(mim, ' dt*Incompressibility(u_f,v_f)*Test_p + dt*Mesh_def(u_f):Grad(Test_u_f)',FLUID) # i didn't multiply the Mesh_def for dt

# A_E in wick notes
md.add_nonlinear_term(mim,"dt*theta*(Convection(u_f,v_f).Test_v_f+ Stress_vu(u_f,v_f):Grad_Test_v_f) +" \
                                "dt*(1-theta)*(Convection(Previous_u_f,Previous_v_f).Test_v_f+ " \
                                "Stress_vu(Previous_u_f,Previous_v_f):Grad_Test_v_f)",FLUID)
md.add_nonlinear_term(mim, "dt*theta*Sigma(u_s):Grad(Test_v_s)+" \
                                "dt*(1-theta)*Sigma(Previous_u_s):Grad(Test_v_s)+" \
                                "(- theta*v_s - (1-theta)*Previous_v_s).Test_u_s" , BEAM)
# A_P in wick notes
md.add_nonlinear_term(mim, 'dt*Stress_p(u_f,p):Grad_Test_v_f', FLUID)



#####################
# COUPLING CONDITIONS
#####################


# Kinematic coupling: u_f = u_s on interface
md.add_linear_term(mim, "(u_f - u_s).Test_mult", BEAM_INTERFACE)

# Velocity coupling: v = v_s on interface  
md.add_linear_term(mim, "(v_f - v_s).Test_mult_v",BEAM_INTERFACE) ################# check 

# Dynamic coupling: stress balance
# The multiplier enforces the stress continuity
md.add_linear_term(mim, "mult.Test_u_f", BEAM_INTERFACE)
md.add_linear_term(mim, "mult.Test_u_s", BEAM_INTERFACE)

#####################
# BOUNDARY CONDITIONS
#####################

inlet_dofs = mfv_fluid.basic_dof_on_region(INLET)

ramp_factor = 0.0
V_inlet_expr = f"{ramp_factor}*[4*1.5*X(2)*(H-X(2))/(H*H), 0]" 
V_inlet= md.interpolation(V_inlet_expr, mfv_fluid)
md.add_initialized_fem_data('V_inlet', mfv_fluid, V_inlet)
V_noslip = md.interpolation( "[0,0]" , mfv_fluid)
md.add_initialized_fem_data('V_noslip', mfv_fluid, V_noslip)

# Apply Dirichlet conditions with multipliers
md.add_Dirichlet_condition_with_multipliers(mim, "v_f", mfv_fluid, INLET, "V_inlet")
md.add_Dirichlet_condition_with_multipliers(mim, "v_f", mfv_fluid, WALLS, "V_noslip")
md.add_Dirichlet_condition_with_multipliers(mim, "v_f", mfv_fluid, CYLINDER, "V_noslip")
md.add_Dirichlet_condition_with_multipliers(mim, "v_f", mfv_fluid, BEAM_INTERFACE, "V_noslip")
md.add_Dirichlet_condition_with_multipliers(mim, "v_f", mfv_fluid, BEAM_LEFT, "V_noslip")



md.add_Dirichlet_condition_with_multipliers(mim,'u_f', mfu_fluid, INLET)
md.add_Dirichlet_condition_with_multipliers(mim,'u_f', mfu_fluid, OUTLET)
md.add_Dirichlet_condition_with_multipliers(mim,'u_f', mfu_fluid, WALLS)
md.add_Dirichlet_condition_with_multipliers(mim,'u_f', mfu_fluid, CYLINDER)


md.add_Dirichlet_condition_with_multipliers(mim,'u_s', mfu_solid, BEAM_LEFT)
md.add_Dirichlet_condition_with_multipliers(mim,'v_s', mfv_solid, BEAM_LEFT)




####################
## MODEL SOLOTION ##
####################

# Create output directory
output_dir = "FSI/FSI_Benchmark_Results_1mesh"
os.makedirs(output_dir, exist_ok=True)

# TIME STEPPING LOOP
print("Starting FSI Benchmark simulation...")
print(f"Time step: {dt}, Total time: {T}")
print(f"Theta: {theta} (Crank-Nicolson)" if theta == 0.5 else f"Theta: {theta}")

t = 0.0
step = 0
export_every = int(1/dt)  # Export every 0.01s


# Main time loop
while t < T:
    
    # Update inlet velocity with ramp
    
    md.solve("very noisy", "max_iter", 50, "max_res", 1e-5,
             "lsearch", "simplest", "alpha max ratio", 2., "alpha min", 0.1)

    # Extract current solution
    u_f = md.variable("u_f")
    v_f = md.variable("v_f")
    p = md.variable("p")
    u_s = md.variable("u_s")
    v_s = md.variable("v_s")
    
    # Export solution
    
    # Alternative: Create region-specific meshes for export
    # Export solution
    if step % export_every == 0:
        # Create full-sized arrays for export
        u_f_full = np.zeros(mfu_fluid.nbdof())
        v_f_full = np.zeros(mfv_fluid.nbdof())
        p_full = np.zeros(mfp_fluid.nbdof())
        u_s_full = np.zeros(mfu_solid.nbdof())
        v_s_full = np.zeros(mfv_solid.nbdof())
        
        # Get DOF indices for each region
        fluid_u_dofs = mfu_fluid.basic_dof_on_region(FLUID)
        fluid_v_dofs = mfv_fluid.basic_dof_on_region(FLUID)
        fluid_p_dofs = mfp_fluid.basic_dof_on_region(FLUID)
        solid_u_dofs = mfu_solid.basic_dof_on_region(BEAM)
        solid_v_dofs = mfv_solid.basic_dof_on_region(BEAM)
        
        # Fill in the values at the appropriate DOF locations
        u_f_full[fluid_u_dofs] = u_f
        v_f_full[fluid_v_dofs] = v_f
        p_full[fluid_p_dofs] = p
        u_s_full[solid_u_dofs] = u_s
        v_s_full[solid_v_dofs] = v_s
        
        # Export with full-sized arrays
        mfu_fluid.export_to_vtu(f"{output_dir}/fsi_{step:05d}.vtu",
                            mfu_fluid, u_f_full, "MeshDisplacement",
                            mfv_fluid, v_f_full, "FLuidVelocity", 
                            mfp_fluid, p_full, "Pressure",
                            mfu_solid, u_s_full, "Displacement",
                            mfv_solid, v_s_full, "Velocity")
    # Update previous time step values
    md.set_variable("Previous_u_f", u_f)
    md.set_variable("Previous_v_f", v_f)
    md.set_variable("Previous_u_s", u_s)
    md.set_variable("Previous_v_s", v_s)
    
    # Advance time
    t += dt
    step += 1
    md.set_variable("t", t)
    
    ################################
    # INLET BOUNDARY CONTION UPDATE#
    ################################
    "The inlet dirichlet b.c has to be uptadated at each iteration; because there's a ramp factor for the firs 2 seconds. "
    if t < 2.0:
        ramp_factor = 0.5 * (1 - np.cos(np.pi * t / 2.0))
        V_inlet_expr = f"{ramp_factor}*[4*1.5*X(2)*(H-X(2))/(H*H), 0]"
        V_inlet= md.interpolation(V_inlet_expr, mfv_fluid)
        md.set_variable('V_inlet', V_inlet)
        md.add_Dirichlet_condition_with_multipliers(mim, "v_f", mfv_fluid, INLET, "V_inlet")



    
    print(f'time is: {t} and ramp_factor is {ramp_factor}')
    
    
        
        