import getfem as gf 
import numpy as np
import os



############################{####
# Cylinder in a Fluid:
#################################
'''
The goal is to solve the full Navier-Stokes equations for a cylinder immeresed in a fluid.
'''

##################
## PROBLEM DATA ##
##################
# Geometry parameters
L = 2.2      # Channel length
H = 0.41     # Channel height
c_x = 0.2    # Cylinder center x
c_y = 0.2    # Cylinder center y
r = 0.05     # Cylinder radius


# Fluid properties (artificial values for a Reynolds number of 60 ):
mu_fluid = 0.001   # Dynamic viscosity (Pa·s)
rho_fluid = 1.0    # Density (kg/m³)

# Boundaries values: 
# Inlet velocity parameters
U_max = 1.5      # Mean inlet velocity m/s

# Reference values for coefficients
D = 1  # Cylinder diameter (you need to set this based on your mesh)
A_ref = D   # Reference area (diameter × unit depth for 2D)
q_inf = 0.5 * rho_fluid * U_max**2  # Dynamic pressure


# Transient paramters: 
T = 10.0           # Total simulation time
dt = 1e-4       # Time step
theta = 0.5      # Theta parameter (0.5 = Crank-Nicolson)


print(f"Reynolds number is: {U_max*D/ν_fluid} and number of time steps: {int(T/dt)}")
print(f"Strouhal number is approximately: {0.2}, thus a peiod is T = {1/0.2*D/U_max}")

############
## MESH ##
#############

Mesh_fluid = gf.Mesh('Import', 'gmsh','fluid/Mesh/cylinder_channel_tri.msh')
h = min(Mesh_fluid.convex_radius())
print( f"Minimum mesh size h ={h}, and CFL = "f"{1}, thus dt = {dt} should be less than {1*h/U_max}" )

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
Cylinder_1 = 8
Cylinder_2 = 9
Cylinder_3 = 10
Cylinder_4 = 11
print("Regions in the mesh are:", Mesh_fluid.regions())

CYLINDER_INTERFACE = 2001

Mesh_fluid.region_merge(CYLINDER_INTERFACE, Cylinder_1)
Mesh_fluid.region_merge(CYLINDER_INTERFACE, Cylinder_2)
Mesh_fluid.region_merge(CYLINDER_INTERFACE, Cylinder_3)
Mesh_fluid.region_merge(CYLINDER_INTERFACE, Cylinder_4)

# physical surface: 
fluid1 = 6 
fluid2 = 12
fluid3 = 13
fluid4 = 14
fluid5 = 15

FLUID = 2002
Mesh_fluid.region_merge(FLUID,fluid1)
Mesh_fluid.region_merge(FLUID,fluid2)
Mesh_fluid.region_merge(FLUID,fluid3)
Mesh_fluid.region_merge(FLUID,fluid4)
Mesh_fluid.region_merge(FLUID,fluid5)

WALLS  = 2003
Mesh_fluid.region_merge(WALLS,Bottom_left)
Mesh_fluid.region_merge(WALLS,Bottom_right)
Mesh_fluid.region_merge(WALLS,Top_left)
Mesh_fluid.region_merge(WALLS,Top_right)

########################
## INTEGRATION METHOD ##
########################
"""
The integration method is quadrature with 5 points, which is suitable for QK elements.  
"""
mim_fluid = gf.MeshIm(Mesh_fluid, gf.Integ('IM_TRIANGLE(5)'))


#########################
## FEM ELEMENTS METHOD ##
########################1
"""
THe meshes are quadrilater, hence QK elements are used. 
"""

## FEM ELEMENTS:

mfv_fluid = gf.MeshFem(Mesh_fluid, 2)
mfv_fluid.set_fem(gf.Fem('FEM_PK(2,2)'))

mfp_fluid = gf.MeshFem(Mesh_fluid, 1)
mfp_fluid.set_fem(gf.Fem('FEM_PK(2,1)'))




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


md.add_fem_variable("v_f", mfv_fluid)        # Fluid velocity  
md.add_fem_variable("p", mfp_fluid)          # Pressure

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
md.add_initialized_data("U_max", U_max)

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
The problem is formulated in weak form and for the time the theta method is used.
"""


# FLUID : 
# stress tensors: 
md.add_macro('Stress_vu(v)', "rho_f*nu_f*2*Sym(Grad_v) ")
md.add_macro('Stress_p(p)', '-p*Id(2)')
md.add_macro("Convection(v)", "(rho_f*v.Grad_v)")
md.add_macro('Incompressibility(v)', "Trace(Grad_v)")
################################################################## provare a fare la simulazione senza macro


Transient_fluid = '(rho_f/dt)*(v_f - Previous_v_f). Test_v_f' 
            
md.add_nonlinear_term(mim_fluid, Transient_fluid, FLUID)


md.add_linear_term(mim_fluid, ' Incompressibility(v_f)*Test_p', FLUID) 

md.add_nonlinear_term(mim_fluid,"theta*(Convection(v_f)).Test_v_f +" \
 "(1-theta)*(Convection(Previous_v_f)).Test_v_f", FLUID)

md.add_linear_term(mim_fluid,"theta*Stress_vu(v_f):Grad_Test_v_f +" \
                                "(1-theta)*(Stress_vu(Previous_v_f):Grad_Test_v_f)", FLUID)
md.add_linear_term(mim_fluid, 'Stress_p(p):Grad_Test_v_f', FLUID)


#####################
# BOUNDARY CONDITIONS
#####################


# Boundary conditions
# Inlet velocity profile with ramp-up

inlet_dofs = mf_v.basic_dof_on_region(INLET)

ramp_factor = 1.5*np.sin(np.pi * t / 8) 
V_inlet_expr = f"{ramp_factor}*[4*1.5*X(2)*(H-X(2))/(H*H), 0]"
V_inlet= md1.interpolation(V_inlet_expr, mf_v)
md1.add_initialized_fem_data('V_inlet', mf_v, V_inlet)

V_noslip = md1.interpolation( "[0,0]" , mf_v)
md1.add_initialized_fem_data('V_noslip', mf_v, V_noslip)


# Apply Dirichlet conditions with multipliers
md.add_Dirichlet_condition_with_multipliers(mim_fluid, "v_f", mfv_fluid, INLET, "V_inlet")
md.add_Dirichlet_condition_with_multipliers(mim_fluid, "v_f", mfv_fluid, WALLS, "V_noslip")
md.add_Dirichlet_condition_with_multipliers(mim_fluid, "v_f", mfv_fluid, CYLINDER_INTERFACE, "V_noslip")



####################
## MODEL SOLOTION ##
####################

# Create output directory
output_dir = "fluid/Results"
os.makedirs(output_dir, exist_ok=True)

# TIME STEPPING LOOP

print(f"Time step: {dt}, Total time: {T}, number of steps: {int(T/dt)}")

t = 0.0
step = 0


time_history = []
cd_history = []
cl_history = []


# Main time loop
while t < T:
    
  md.set_variable("t", t)
  md.solve("noisy", 
         "max_iter", 100,
         "max_res", 1e-8,  
         "lsolver", "superlu",  
         "alpha min", 1e-4,  
         "alpha mult", 0.5)    

  # Extract current solution
  v_f = md.variable("v_f")
  p = md.variable("p")

  print(f'time is: {t} the velocity is {v_f}')

  
  time_ms = int(t * 1000)  # Convert to milliseconds
  mfv_fluid.export_to_vtu(f"{output_dir}/fluid_{time_ms:06d}.vtu",
                        mfv_fluid, v_f, "Velocity", 
                        mfp_fluid, p, "Pressure")
    


  md.set_variable("Previous_v_f", v_f)

  ###### CL and CD computation ######
  traction = gf.asm_generic(mim_fluid, 0, "Stress_vu(v_f)*Normal+ Stress_p(p)*Normal", CYLINDER_INTERFACE, md)
  Fx, Fy = -traction[0], -traction[1]
  Cd = Fx / (q_inf * A_ref)
  Cl = Fy / (q_inf * A_ref)
  print(f"Time: {t:.4f}, Cd: {Cd:.6f}, Cl: {Cl:.6f}")

  time_history.append(t)
  cd_history.append(Cd)
  cl_history.append(Fy)
  np.savetxt(f"fluid/force_coefficients.txt", 
        np.column_stack([time_history, cd_history, cl_history]),
        header="Time, Cd, Cl")
  
  ## some checks to see if the solution is resonable ##
  div_norm = np.sqrt(gf.asm_generic(mim_fluid, 0, 'pow((Trace(Grad_v_f)),2)', FLUID, md))
  print("‖div(v_f)‖ₗ₂ =", div_norm)

  # Advance time
  t += dt
  step += 1
  
print("simulation completed.")