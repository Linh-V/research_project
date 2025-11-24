import getfem as gf 
import numpy as np
import gmsh
import os
def create_dfg_mesh(mesh_file="dfg_benchmark.msh", visualize=False):
    """
    Create QUADRILATERAL mesh for DFG 2D-3 benchmark
    """
    
    gmsh.initialize()
    gmsh.model.add("DFG_2D3_Benchmark")
    
    # Geometry
    L = 2.2
    H = 0.41
    c_x = 0.2
    c_y = 0.2
    r = 0.05
    gdim = 2
    
    # Create geometry
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
    gmsh.model.occ.synchronize()
    
    # Physical groups
    FLUID = 1
    INLET= 2
    OUTLET = 3
    WALLS = 4
    OBSTACLE = 5
    
    volumes = gmsh.model.getEntities(dim=gdim)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], FLUID)
    gmsh.model.setPhysicalName(volumes[0][0], FLUID, "FLUID")
    
    # Boundary classification
    inflow, outflow, walls, obstacle = [], [], [], []
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, H/2, 0], atol=1e-3):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L, H/2, 0], atol=1e-3):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L/2, H, 0], atol=1e-3) or \
             np.allclose(center_of_mass, [L/2, 0, 0], atol=1e-3):
            walls.append(boundary[1])
        else:
            obstacle.append(boundary[1])
    
    gmsh.model.addPhysicalGroup(1, inflow,INLET)
    gmsh.model.setPhysicalName(1, INLET, "INLET")
    gmsh.model.addPhysicalGroup(1, outflow, OUTLET)
    gmsh.model.setPhysicalName(1, OUTLET, "OUTLET")
    gmsh.model.addPhysicalGroup(1, walls, WALLS)
    gmsh.model.setPhysicalName(1, WALLS, "WALLS")
    gmsh.model.addPhysicalGroup(1, obstacle, OBSTACLE)
    gmsh.model.setPhysicalName(1, OBSTACLE, "OBSTACLE")
    
    # Mesh size field
    res_min = r / 3
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
    
    # # IMPORTANT: Generate QUADRILATERAL mesh
    # gmsh.option.setNumber("Mesh.Algorithm", 8)
    # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    # gmsh.option.setNumber("Mesh.RecombineAll", 1)        
    # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)  # Second order elements
    gmsh.model.mesh.optimize("Netgen")
    
    # Save in MSH2 format (GetFEM compatible)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(mesh_file)
    
    print(f"Quadrilateral mesh saved to: {mesh_file}")
    
    if visualize:
        gmsh.fltk.run()
    
    gmsh.finalize()



################################
# Cylinder in a Fluid:
#################################
# Create output directory
output_dir = "fluid/Benchmark/results_cylinder_dfg"
os.makedirs(output_dir, exist_ok=True)

'''
The goal is to solve the full Navier-Stokes equations for a cylinder immeresed in a fluid for Re = 100.
nu  = 0.001
L_char = 0.1
U_mean = 1.0
U_max = 1.5
'''

##################
## PROBLEM DATA ##
##################

# Fluid properties (air):
ν_fluid = 0.001      # m²/s 
rho_fluid = 1        # kg/m³ 

# Transient paramters: 
T = 8           # Total simulation time
dt = 1/1600        # Time step
num_step = int(T/dt)

#############
## MESH ##
#############
create_dfg_mesh("dfg_benchmark.msh", visualize=False)

############
## MESH creation#
############

# Load mesh
mesh_file = 'dfg_benchmark.msh'  # or 'dfg_benchmark_tri.msh' for triangular
Mesh = gf.Mesh('import', 'gmsh', mesh_file)
# Mesh.export_to_vtk('mesh_dfg.vtk')

Mesh.export_to_vtk(f'{output_dir}/Cyl_mesh.vtk') 
# Print mesh info
print(f"Mesh dimension: {Mesh.dim()}")
print(f"Number of convexes: {Mesh.nbcvs()}")
print(f"Number of points: {Mesh.nbpts()}")

# Physical groups
FLUID = 1
INLET= 2
OUTLET = 3
WALLS = 4
OBSTACLE = 5

########################
## INTEGRATION METHOD ##
########################
"""
The integration method is quadrature with 7 points, which is suitable for QK elements.  
"""
mim_fluid = gf.MeshIm(Mesh, gf.Integ('IM_TRIANGLE(9)'))

#########################
## FEM ELEMENTS METHOD ##
########################1
"""
The meshes are trilaterals, hence PK elements are used. 
"""
## FEM ELEMENTS: Taylor-Hood stable 

mfv_fluid = gf.MeshFem(Mesh, 2)
mfv_fluid.set_fem(gf.Fem('FEM_PK(2,4)')) #Lagrangian P2

mfp_fluid = gf.MeshFem(Mesh, 1)
mfp_fluid.set_fem(gf.Fem('FEM_PK(2,2)')) #Lagrangian P1

########################
## INITIAL CONDITIONS ##
########################

u_n = np.zeros(mfv_fluid.nbdof())      # u^n
u_n1 = np.zeros(mfv_fluid.nbdof())    
p_n = np.zeros(mfp_fluid.nbdof())    

####################
## MODEL SOLOTION ##
####################


# Storage for results
time_history = []
cd_history = []
cl_history = []
p_diff_history = []


# Points for pressure evaluation
p_front_point = np.array([[0.15], [0.2]])
p_back_point = np.array([[0.25], [0.2]])


# TIME STEPPING LOOP

t = 0
for steps in range (num_step):

    t +=dt

    ####################
    ## Step 1 : Tentative velocity ##
    ####################

    temp = gf.Model("real")

    temp.add_initialized_data("rho_f", rho_fluid)
    temp.add_initialized_data("nu_f", ν_fluid)
    temp.add_initialized_data("dt", dt)

    temp.add_fem_variable('u', mfv_fluid)
    temp.add_fem_data('u_n', mfv_fluid)
    temp.add_fem_data('u_n1', mfv_fluid)
    temp.add_fem_data('p_n', mfp_fluid)

    temp.set_variable('u_n', u_n)
    temp.set_variable('u_n1', u_n1)
    temp.set_variable('p_n', p_n)

    if steps == 0:
        # First step: use backward Euler
        conv_term = 'rho_f*(u_n.Grad(u_u)).Test_u'
    else:
        # Subsequent steps: use Adams-Bashforth
        conv_term = 'rho_f*((1.5*u_n - 0.5*u_n1).Grad(u_n)).Test_u'
   
    
    temp.add_nonlinear_term(mim_fluid,
            '0.5*rho*((1.5*u_n - 0.5*u_n1).Grad_u).Test_u', FLUID)
    temp.add_linear_term(mim_fluid, 'rho_f/dt * (u-u_n).Test_u +'           #time derivative
                        f'{conv_term} +'                                    #convection  \   
                        '0.5*nu_f*(Grad(u)+Grad(u_n)):Grad_Test_u - '       #diffusion \
                        'p_n.Div_Test_u', FLUID)                            #pressure
    
    V_noslip = temp.interpolation("[0,0]", mfv_fluid)
    temp.add_initialized_fem_data('V_noslip', mfv_fluid, V_noslip)

    temp.add_Dirichlet_condition_with_simplification('u', WALLS)
    temp.add_Dirichlet_condition_with_simplification('u', OBSTACLE)


    # ramp_factor = 1.5*np.sin(np.pi * t / 8)
    if t < 4.0:
        ramp_factor = 1.5 * (1 - np.cos(np.pi * t / 4)) / 2  # Smooth ramp from 0 to 1.5
    else:
        ramp_factor = 1.5 * np.sin(np.pi * t / 8)
    V_inlet_expr = f"{ramp_factor}*[4*X(2)*(0.41-X(2))/(0.41*0.41), 0]"
    V_inlet=temp.interpolation(V_inlet_expr, mfv_fluid)

    temp.add_initialized_fem_data('V_inlet', mfv_fluid, V_inlet)

    temp.add_Dirichlet_condition_with_multipliers(mim_fluid,'u', mfv_fluid, INLET, 'V_inlet')
    # temp.add_Dirichlet_condition_with_penalization(mim_fluid, 'u', 1e100, INLET, 'V_inlet')


    temp.solve("noisy", "max_iter", 500, "max_res", 1e-8, "lsolver", "mumps")
    u_tent = temp.variable("u")

    # DEBUG: Check if inlet velocity was actually imposed
    inlet_dofs = mfv_fluid.basic_dof_on_region(INLET)
    print(f"\nAfter Step 1 solve:")
    print(f"  V_inlet at inlet (expected): min={V_inlet[inlet_dofs].min():.6f}, max={V_inlet[inlet_dofs].max():.6f}")
    print(f"  u_tent at inlet (actual):    min={u_tent[inlet_dofs].min():.6f}, max={u_tent[inlet_dofs].max():.6f}")
    print(f"  Difference: {np.linalg.norm(u_tent[inlet_dofs] - V_inlet[inlet_dofs]):.6e}")

    wall_dofs = mfv_fluid.basic_dof_on_region(WALLS)
    print(f"  V_noslip at walls (expected): min={V_noslip[wall_dofs].min():.6f}, max={V_noslip[wall_dofs].max():.6f}")
    print(f"  u_tent at walls (actual):    min={u_tent[wall_dofs].min():.6f}, max={u_tent[wall_dofs].max():.6f}")

    # # Manually override inlet DOFs
    # u_tent[inlet_dofs] = V_inlet[inlet_dofs]

    # print(f"After manual override:")
    # print(f"u_tent at inlet: min={u_tent[inlet_dofs].min():.6f}, max={u_tent[inlet_dofs].max():.6f}")

    #################################
    # STEP 2: Pressure correction
    #################################   

    md2 = gf.Model('real')

    md2.add_fem_variable('phi', mfp_fluid)
    md2.add_fem_data('u_tent', mfv_fluid)
    md2.set_variable('u_tent', u_tent)

    md2.add_initialized_data("rho_f", rho_fluid)
    md2.add_initialized_data("dt", dt)

    md2.add_linear_term(mim_fluid, 'Grad_phi.Grad_Test_phi +' \
                                    'rho_f/dt * Div_u_tent*Test_phi', FLUID)

    # BC: φ = 0 at outlet
    md2.add_Dirichlet_condition_with_simplification('phi', OUTLET)

    md2.solve("noisy", "max_iter", 500, "max_res", 1e-8, "lsolver", "mumps")   
    phi = md2.variable("phi")
    

    #################################
    # STEP 3: Velocity correction
    #################################

    md3 = gf.Model('real')

    md3.add_fem_variable('u', mfv_fluid)

    md3.add_fem_data('u_tent', mfv_fluid)
    md3.set_variable('u_tent', u_tent)
    md3.add_fem_data('phi', mfp_fluid)
    md3.set_variable('phi', phi)

    md3.add_initialized_data("rho_f", rho_fluid)
    md3.add_initialized_data("dt", dt)

    md3.add_linear_term (mim_fluid, 'rho_f * u.Test_u - ' \
                                    'rho_f * u_tent.Test_u +' \
                                    'dt*Grad_phi.Test_u ')
    
    V_inlet= md3.interpolation(V_inlet_expr, mfv_fluid)
    md3.add_initialized_fem_data('V_inlet', mfv_fluid, V_inlet)
    V_noslip = md3.interpolation("[0,0]", mfv_fluid)
    md3.add_initialized_fem_data('V_noslip', mfv_fluid, V_noslip)

    md3.add_Dirichlet_condition_with_simplification('u', WALLS)
    md3.add_Dirichlet_condition_with_simplification('u', OBSTACLE)

    md3.add_Dirichlet_condition_with_multipliers(mim_fluid,'u', mfv_fluid, INLET, 'V_inlet')

    md3.solve("noisy", "max_iter", 500, "max_res", 1e-8, "lsolver", "mumps")
    
    u_new = md3.variable("u")
    p_new = p_n + phi

    u_n1[:] = u_n.copy()   # previous previous
    u_n[:]  = u_new.copy()  # new previous
    p_n[:]  = p_new.copy()

    print(u_new)
    print (f"step {steps:05d}")

    if steps % 100 == 0:
        # Velocity vector
        vtu = os.path.join(output_dir, f"step_{steps:05d}.vtu")
        mfv_fluid.export_to_vtu(
            vtu,
            mfv_fluid,
            u_new,
            'Velocity',
            mfp_fluid,
            p_new,
            'Pressure'
        )

    #################################
    # Compute drag and lift
    #################################
    
    md_force = gf.Model("real")
    md_force.add_fem_data("p_new", mfp_fluid)
    md_force.set_variable("p_new", p_new)
    
    md_force.add_fem_data("u_new", mfv_fluid)
    md_force.set_variable("u_new", u_new)
    
    md_force.add_initialized_data("mu", ν_fluid)
    # Traction: σ·n = [μ(∇u + ∇u^T) - pI]·n
    traction = gf.asm_generic(mim_fluid, 0, "(mu*(Grad_u_new + Grad_u_new') - p_new*Id(2))*Normal",OBSTACLE, md_force)
    
    Fx = -traction[0]
    Fy = -traction[1]
    
    # Drag and lift coefficients
    D = 2 * 0.05  # Diameter
    U_mean = 1
    
    Cd = 2 * Fx / (rho_fluid * U_mean**2 * D)
    Cl = 2 * Fy / (rho_fluid * U_mean**2 * D)
    
    # Pressure difference
    try:
        p_front = gf.compute_interpolate_on(mfp_fluid, p_new, p_front_point)[0]
        p_back = gf.compute_interpolate_on(mfp_fluid, p_new, p_back_point)[0]
        p_diff = p_front - p_back
    except:
        p_diff = 0.0
    
    time_history.append(t)
    cd_history.append(Cd)
    cl_history.append(Cl)
    p_diff_history.append(p_diff)
    
    print(f"  Cd={Cd:.6f}, Cl={Cl:.6f}, ΔP={p_diff:.6f}")
    np.savetxt(f"{output_dir}/force_coefficients_channel.txt",
        np.column_stack([time_history, cd_history, cl_history, p_diff_history]),
        header="Time, Cd, Cl, Pressure_Diff",
        fmt='%.8e')

#################################
# Save last iteration
#################################
vtu = os.path.join(output_dir, f"step_final.vtu")
mfv_fluid.export_to_vtu(
    vtu,
    mfv_fluid,
    u_new,
    'Velocity',
    mfp_fluid,
    p_new,
    'Pressure'
)

#################################
# Save force coefficients
#################################



print(f"Results saved to {output_dir}/")
print(f"Final time: {t:.4f}")