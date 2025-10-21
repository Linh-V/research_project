import getfem as gf 
import numpy as np
import os
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import gmsh
from Functions import verify_regions
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
    fluid_marker = 1
    inlet_marker = 2
    outlet_marker = 3
    wall_marker = 4
    obstacle_marker = 5
    
    volumes = gmsh.model.getEntities(dim=gdim)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")
    
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
    
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")
    
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
    
    # IMPORTANT: Generate QUADRILATERAL mesh
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)        
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    
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

###########################
# DFG 2D-3 Benchmark: Cylinder in Channel
###########################


# Generate the mesh
if __name__ == "__main__":
    #create_dfg_mesh("dfg_benchmark.msh", visualize=True)

    # ############
    # ## MESH creation##
    # ############

    # # Load mesh
    #mesh_file = 'dfg_benchmark.msh'  # or 'dfg_benchmark_tri.msh' for triangular
    #Mesh = gf.Mesh('import', 'gmsh', mesh_file)
    # # Mesh.export_to_vtk('mesh_dfg.vtk')

    # # Print mesh info
    # print(f"Mesh dimension: {Mesh.dim()}")
    # print(f"Number of convexes: {Mesh.nbcvs()}")
    # print(f"Number of points: {Mesh.nbpts()}")

    ###################
    ## MESH imported ##
    ###################

    Mesh= gf.Mesh('Import', 'gmsh','fluid/Mesh/cylinder_channel_tri.msh')

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
    print("Regions in the mesh are:", Mesh.regions())

    OBSTACLE = 2001

    Mesh.region_merge(OBSTACLE, Cylinder_1)
    Mesh.region_merge(OBSTACLE, Cylinder_2)
    Mesh.region_merge(OBSTACLE, Cylinder_3)
    Mesh.region_merge(OBSTACLE, Cylinder_4)

    # physical surface: 
    fluid1 = 6 
    fluid2 = 12
    fluid3 = 13
    fluid4 = 14
    fluid5 = 15

    FLUID = 2002
    Mesh.region_merge(FLUID,fluid1)
    Mesh.region_merge(FLUID,fluid2)
    Mesh.region_merge(FLUID,fluid3)
    Mesh.region_merge(FLUID,fluid4)
    Mesh.region_merge(FLUID,fluid5)

    WALLS  = 2003
    Mesh.region_merge(WALLS,Bottom_left)
    Mesh.region_merge(WALLS,Bottom_right)
    Mesh.region_merge(WALLS,Top_left)
    Mesh.region_merge(WALLS,Top_right)

    
    

    ##################
    ## PROBLEM DATA ##
    ##################

    # Geometry parameters
    L = 2.2      # Channel length
    H = 0.41     # Channel height
    c_x = 0.2    # Cylinder center x
    c_y = 0.2    # Cylinder center y
    r = 0.05     # Cylinder radius

    # Fluid properties
    mu = 0.001   # Dynamic viscosity (Pa·s)
    rho = 1.0    # Density (kg/m³)

    # Inlet velocity parameters
    U_max = 1.5  # Maximum inlet velocity (m/s)

    # Time parameters:


    h = min(Mesh.convex_radius())
    print( f"Minimum mesh size h ={h}, and CFL = "f"{1}, thus dt should be less than {1*h/U_max}" )
    T = 1.0      # Total simulation time (s) - reduced for testing
    dt = 0.0001  # Time step
    num_steps = int(T / dt)

    print(f"Reynolds number (based on diameter): {rho * 2/3*U_max * (2*r) / mu}")
    print(f"Number of time steps: {num_steps}")

    ########################
    ## INTEGRATION METHOD ##
    ########################

    mim = gf.MeshIm(Mesh, gf.Integ('IM_TRIANGLE(5)'))

    #########################
    ## FEM ELEMENTS ##
    #########################

    # Velocity: P2 elements (quadratic)
    mf_v = gf.MeshFem(Mesh, 2)
    mf_v.set_fem(gf.Fem('FEM_PK(2,2)'))

    # Pressure: P1 elements (linear)
    mf_p = gf.MeshFem(Mesh, 1)
    mf_p.set_fem(gf.Fem('FEM_PK(2,1)'))

    print(f"Velocity DOFs: {mf_v.nbdof()}")
    print(f"Pressure DOFs: {mf_p.nbdof()}")

    ########################
    ## INITIAL CONDITIONS ##
    ########################

    u_n = np.zeros(mf_v.nbdof())      # u^n
    u_n1 = np.zeros(mf_v.nbdof())     # u^{n-1}
    p_n = np.zeros(mf_p.nbdof())      # p^n

    #verify_regions(Mesh, 'fluid/Mesh/meshfluid_cylinder_channel')

    ####################
    ## SOLVER SETUP   ##
    ####################

    output_dir = "fluid/results_cylinder_channel_getfem_tri"
    os.makedirs(output_dir, exist_ok=True)

    # Storage for results
    time_history = []
    cd_history = []
    cl_history = []
    p_diff_history = []

    # Points for pressure evaluation
    p_front_point = np.array([[0.15], [0.2]])
    p_back_point = np.array([[0.25], [0.2]])

    ####################
    ## TIME STEPPING  ##
    ####################

    print("Starting time integration...")
    print(f"dt = {dt}, T = {T}, steps = {num_steps}")

    t = 0.0

    for step in range(num_steps):
        t += dt
        
        if step % 100 == 0:
            print(f"Step {step}/{num_steps}, t = {t:.4f}")
        
        #################################
        # STEP 1: Tentative velocity
        #################################
        # Solve: rho/dt*(u* - u^n) + rho*(1.5*u^n - 0.5*u^{n-1})·∇u*
        #        + mu*∇²u* = 0
        
        md1 = gf.Model("real")
        

        md1.add_fem_variable("u", mf_v) # u i s the variable
        md1.add_fem_data("u_n", mf_v) # u is the previus step
        md1.add_fem_data("u_n1", mf_v) # u is the previus to the previus step
        md1.add_fem_data("p_n", mf_p) # p is the previus step
        
        md1.set_variable("u_n", u_n)    # set the previus step u, at the beginning is zero
        md1.set_variable("u_n1", u_n1)  # set the previus step of the previus step u, at the beginning is zero
        md1.set_variable("p_n", p_n)    # set the previus step p, at the beginning is zero
        
        md1.add_initialized_data("rho", rho)
        md1.add_initialized_data("mu", mu)
        md1.add_initialized_data("dt", dt)
        md1.add_initialized_data("H", H)

        
        # Time derivative
        md1.add_linear_term(mim, '(rho/dt)*(u - u_n).Test_u', FLUID)
        
        # Convection (Adams-Bashforth): (1.5*u_n - 0.5*u_n1)*0.5*Grad(u+un)
        md1.add_nonlinear_term(mim,
            '0.5*rho*((1.5*u_n - 0.5*u_n1).Grad_u).Test_u', FLUID)
        md1.add_linear_term(mim,
            '0.5*rho*((1.5*u_n - 0.5*u_n1).Grad_u_n).Test_u', FLUID)
        
        # Crank-Nicolson diffusion: 0.5*(mu*∇²(u+u_n))
        md1.add_linear_term(mim, ' 0.5*mu*(Grad_u):Grad_Test_u', FLUID) # segno corretto? 
        md1.add_linear_term(mim, ' 0.5*mu*(Grad_u_n):Grad_Test_u', FLUID) # segno corretto? 

        # Pressure from previous step
        md1.add_linear_term(mim, '- p_n*Trace(Grad_Test_u)', FLUID)
        
        # Boundary conditions
        # Inlet velocity profile with ramp-up

        inlet_dofs = mf_v.basic_dof_on_region(INLET)

        ramp_factor = 1.5*np.sin(np.pi * t / 8) 
        V_inlet_expr = f"{ramp_factor}*[4*1.5*X(2)*(H-X(2))/(H*H), 0]"
        V_inlet= md1.interpolation(V_inlet_expr, mf_v)
        md1.add_initialized_fem_data('V_inlet', mf_v, V_inlet)

        V_noslip = md1.interpolation( "[0,0]" , mf_v)
        md1.add_initialized_fem_data('V_noslip', mf_v, V_noslip)
        
        # Boundary conditions
        md1.add_Dirichlet_condition_with_multipliers(mim, "u", 1, INLET, "V_inlet")
        md1.add_Dirichlet_condition_with_multipliers(mim, "u", 1, WALLS, "V_noslip")
        md1.add_Dirichlet_condition_with_multipliers(mim, "u", 1, OBSTACLE, "V_noslip")
        
        # Solve
        md1.solve("noisy", "max_iter", 100, "max_res", 1e-8, "lsolver", "superlu")
        u_star = md1.variable("u")

        #################################
        # STEP 2: Pressure correction
        #################################
        # Solve: ∇²φ = (rho/dt)*∇·u*
        
        md2 = gf.Model("real")
        md2.add_fem_variable("phi", mf_p)
        md2.add_fem_data("u_star", mf_v)
        md2.set_variable("u_star", u_star)
        md2.add_initialized_data("rho", rho)
        md2.add_initialized_data("dt", dt)
        
        # Poisson equation
        md2.add_linear_term(mim, 'Grad_phi.Grad_Test_phi', FLUID)
        md2.add_linear_term(mim, '(rho/dt)*Trace(Grad_u_star)*Test_phi', FLUID)
        
        # BC: φ = 0 at outlet
        md2.add_Dirichlet_condition_with_multipliers(mim, "phi", 1, OUTLET)
        
        md2.solve("noisy", "max_iter", 100, "max_res", 1e-8, "lsolver", "superlu")   
        phi = md2.variable("phi")
        phi_norm = np.linalg.norm(phi)
        print(f"  ‖φ‖ = {phi_norm:.6e}")
        if phi_norm < 1e-12:
            print("  ⚠️  WARNING: phi is essentially zero!")
                
        #################################
        # STEP 3: Velocity correction
        #################################
        # u^{n+1} = u* - (dt/rho)*∇φ
        
        # Mass matrix M = ∫ ρ * u · v dx
        md3 = gf.Model("real")
        md3.add_fem_variable("u_new", mf_v)
        md3.add_fem_data("phi", mf_p)
        md3.set_variable("phi", phi)
        md3.add_fem_data("u_star", mf_v)
        md3.set_variable("u_star", u_star)

        md3.add_initialized_data("rho", rho)
        md3.add_initialized_data("dt", dt)
        md3.add_initialized_data("mu", mu)
        md3.add_initialized_data("H", H)
       
        md3.add_linear_term(mim, 'rho*u_new.Test_u_new', FLUID)
        md3.add_linear_term(mim, '-rho*u_star.Test_u_new + dt*Grad_phi.Test_u_new', FLUID)

        md3.solve("noisy", "max_iter", 100, "max_res", 1e-8, "lsolver", "superlu")
        u_new = md3.variable("u_new")

        # Update pressure: p^{n+1} = p^n + φ
        p_new = p_n + phi
        
        # L2 norm of the velocity divergence
        div_norm2 = gf.asm_generic(mim, 0,'pow((Trace(Grad_u_new)),2)',FLUID, md3 )
        div_norm = np.sqrt(div_norm2)
        print(f"‖div(u_new)‖ₗ₂ = {div_norm:.6e}")
        #################################
        # Compute drag and lift
        #################################
        
        
        md_force = gf.Model("real")
        md_force.add_fem_data("p_new", mf_p)
        md_force.set_variable("p_new", p_new)
        md_force.add_fem_data("u_new", mf_v)
        md_force.set_variable("u_new", u_new)
        md_force.add_initialized_data("mu", mu)
        # Traction: σ·n = [μ(∇u + ∇u^T) - pI]·n
        traction = gf.asm_generic(mim, 0, "(mu*(Grad_u_new + Grad_u_new') - p_new*Id(2))*Normal",OBSTACLE, md_force)
        
        Fx = -traction[0]
        Fy = -traction[1]
        
        # Drag and lift coefficients
        D = 2 * r  # Diameter
        U_mean = 2.0 / 3.0 * U_max  # Average velocity for parabolic profile
        
        Cd = 2 * Fx / (rho * U_mean**2 * D)
        Cl = 2 * Fy / (rho * U_mean**2 * D)
        
        # Pressure difference
        try:
            p_front = gf.compute_interpolate_on(mf_p, p_new, p_front_point)[0]
            p_back = gf.compute_interpolate_on(mf_p, p_new, p_back_point)[0]
            p_diff = p_front - p_back
        except:
            p_diff = 0.0
        
        time_history.append(t)
        cd_history.append(Cd)
        cl_history.append(Cl)
        p_diff_history.append(p_diff)
        
        print(f"  Cd={Cd:.6f}, Cl={Cl:.6f}, ΔP={p_diff:.6f}")
        

        #################################
        # Export results
        #################################
        
        #if step % 25 == 0: # export every 25 steps thus every 0.003788s, and there will be 64 files in total 
        
        mf_v.export_to_vtu(f"{output_dir}/velocity_{step:06d}.vtu",
                        mf_v, u_new, "Velocity",
                        mf_p, p_new, "Pressure")
        
        #################################
        # Update for next time step
        #################################
        
        u_n1 = u_n.copy()
        u_n = u_new.copy()
        p_n = p_new.copy()


        #################################
        # Save force coefficients
        #################################

        np.savetxt(f"{output_dir}/force_coefficients_channel_triangle.txt",
                np.column_stack([time_history, cd_history, cl_history, p_diff_history]),
                header="Time Cd Cl Pressure_Diff",
                fmt='%.8e')

        print(f"Results saved to {output_dir}/")
        print(f"Final time: {t:.4f}")

    #################################
    # Plot results (optional)
    #################################

try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].plot(time_history, cd_history, 'b-', linewidth=2)
    axes[0].set_ylabel('Drag Coefficient $C_D$')
    axes[0].grid(True)
    axes[0].set_title('DFG 2D-3 Benchmark Results')
    
    axes[1].plot(time_history, cl_history, 'r-', linewidth=2)
    axes[1].set_ylabel('Lift Coefficient $C_L$')
    axes[1].grid(True)
    
    axes[2].plot(time_history, p_diff_history, 'g-', linewidth=2)
    axes[2].set_ylabel('Pressure Difference $\Delta P$')
    axes[2].set_xlabel('Time [s]')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/coefficients.png", dpi=300)
    print(f"Plot saved to {output_dir}/coefficients.png")
    
except ImportError:
    print("Matplotlib not available for plotting")



