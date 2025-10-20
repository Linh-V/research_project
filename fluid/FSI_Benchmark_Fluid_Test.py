import getfem as gf 
import numpy as np
import os
from Functions import verify_regions, mesh_statistics_corrected



def check_mesh_quality(mesh, U_mean, D, nu, dt):
    """Check mesh quality for cylinder flow simulation"""
    
    # 1. Get mesh element sizes using convex_radius
    radii = mesh.convex_radius()
    h_min = np.min(radii) * 2  # Diameter = 2 * radius
    h_max = np.max(radii) * 2
    h_mean = np.mean(radii) * 2
    
    print(f"Mesh element sizes:")
    print(f"  h_min = {h_min:.3e}, h_max = {h_max:.3e}, h_mean = {h_mean:.3e}")
    
    # 2. Reynolds number
    Re = U_mean * D / nu
    print(f"\nReynolds number: Re = {Re:.1f}")
    
    # 3. Boundary layer thickness estimate
    delta_bl = D / np.sqrt(Re) if Re > 0 else D
    print(f"Boundary layer thickness: δ ≈ {delta_bl:.3e}")
    
    # 4. Required resolution checks
    n_cells_in_bl = delta_bl / h_min
    print(f"Cells in boundary layer: ~{n_cells_in_bl:.1f}")
    
    # 5. CFL condition
    CFL = U_mean * dt / h_min
    print(f"\nCFL number: {CFL:.3f}")
    
    # 6. Check mesh quality
    quality = mesh.quality()
    q_min = np.min(quality)
    q_mean = np.mean(quality)
    poor_elements = np.sum(quality < 0.5)
    
    print(f"\nMesh quality:")
    print(f"  Min quality: {q_min:.3f}, Mean quality: {q_mean:.3f}")
    if poor_elements > 0:
        print(f"  ⚠ WARNING: {poor_elements} elements with quality < 0.5")
    
    # 7. Recommendations based on Re
    print("\nRecommendations:")
    if Re < 50:
        print("- Steady flow regime: Current mesh likely OK")
        cells_on_cylinder = np.pi * D / h_min
        print(f"  Estimated cells on cylinder: ~{cells_on_cylinder:.0f}")
    elif Re < 200:
        print("- Unsteady laminar: Need 100-200 cells on cylinder circumference")
        cells_on_cylinder = np.pi * D / h_min
        print(f"  Estimated cells on cylinder: ~{cells_on_cylinder:.0f}")
    else:
        print("- Higher Re regime: Need finer mesh")
        print(f"  Kolmogorov scale (if turbulent): η ≈ {D/Re**0.75:.3e}")
    
    if n_cells_in_bl < 10:
        print("\n⚠ WARNING: Insufficient boundary layer resolution (< 10 cells)")
        
    if CFL > 1:
        print("⚠ WARNING: CFL > 1, reduce time step or increase mesh size!")
    elif CFL < 0.1:
        print("⚠ Note: CFL < 0.1, you might be able to use larger time steps")

    return h_min, h_max, CFL


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
ν_fluid = 1.516e-5 * scale_factor**2                # m²/s 
rho_fluid = 1.204 /(scale_factor**3  )               # kg/m³ 



# Boundaries values: 
# Inlet velocity parameters
U_mean = 0.1       # Mean inlet velocity m/s
H = 0.2          # Channel height m

# Transient paramters: 
T = 10.0           # Total simulation time
dt = 1e-3         # Time step
theta = 0.5      # Theta parameter (0.5 = Crank-Nicolson)



#############
## MESH ##
#############

Mesh_fluid= gf.Mesh('Import', 'gmsh','fluid/cylinder.msh')
# Mesh_fluid.export_to_vtk('fluid/Fluid.vtk') 
# mesh_statistics_corrected(Mesh_fluid,name="Mesh")

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
                -set(mfv_fluid_.basic_dof_on_region(CYLINDER_INTERFACE))
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

# DIRICHLET CONDITIONS
md.add_Dirichlet_condition_with_penalization(mim_fluid, "v_f", 10**3, INLET, "V_inlet")


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
         "lsolver", "mumps",  
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
    
