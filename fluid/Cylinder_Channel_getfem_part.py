import getfem as gf 
import numpy as np
import os
from Functions import verify_regions


###########################
# DFG 2D-3 Benchmark: Cylinder in Channel
###########################


# Generate the mesh
if __name__ == "__main__":
    output_dir = "fluid/results_cylinder_channel_getfem_tri_coarse"
    os.makedirs(output_dir, exist_ok=True)

    ###################
    ## MESH imported ##
    ###################

    Mesh= gf.Mesh('Import', 'gmsh','fluid/Mesh/cylinder_channel_tri_coarse.msh')

    ############
    # REGIONS ##
    ############
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

    # Mesh.region_merge(OBSTACLE, Cylinder_1)
    # Mesh.region_merge(OBSTACLE, Cylinder_2)
    # Mesh.region_merge(OBSTACLE, Cylinder_3)
    # Mesh.region_merge(OBSTACLE, Cylinder_4)

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
    nu = 0.001   # Dynamic viscosity (Pa·s)
    rho = 1.0    # Density (kg/m³)

    # Inlet velocity parameters
    U_max = 1.5  # Maximum inlet velocity (m/s)

    # Time parameters:


    h = min(Mesh.convex_radius())
    print( f"Minimum mesh size h ={h}, and CFL = "f"{1}, thus dt should be less than {1*h/U_max}" )
    T = 8.0      # Total simulation time (s) - reduced for testing
    dt = 0.0001  # Time step
    num_steps = int(T / dt)

    print(f"Reynolds number (based on diameter): {rho * 2/3*U_max * (2*r) / nu}")
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

    # Storage for results
    time_history = []
    cd_history = []
    cl_history = []
    p_diff_history = []
    div_history = []

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
        #        + nu*∇²u* = 0
        
        md1 = gf.Model("real")
        

        md1.add_fem_variable("u", mf_v) # u i s the variable
        md1.add_fem_data("u_n", mf_v) # u is the previus step
        md1.add_fem_data("u_n1", mf_v) # u is the previus to the previus step
        md1.add_fem_data("p_n", mf_p) # p is the previus step
        
        md1.set_variable("u_n", u_n)    # set the previus step u, at the beginning is zero
        md1.set_variable("u_n1", u_n1)  # set the previus step of the previus step u, at the beginning is zero
        md1.set_variable("p_n", p_n)    # set the previus step p, at the beginning is zero
        
        md1.add_initialized_data("rho", rho)
        md1.add_initialized_data("nu", nu)
        md1.add_initialized_data("dt", dt)
        md1.add_initialized_data("H", H)

        
        # Time derivative
        md1.add_linear_term(mim, '(rho/dt)*(u - u_n).Test_u', FLUID)
        
        # Convection (Adams-Bashforth): (1.5*u_n - 0.5*u_n1)*0.5*Grad(u+un)
        md1.add_nonlinear_term(mim,
            '0.5*rho*((1.5*u_n - 0.5*u_n1).Grad_u).Test_u', FLUID)
        md1.add_linear_term(mim,
            '0.5*rho*((1.5*u_n - 0.5*u_n1).Grad_u_n).Test_u', FLUID)
        
        # Crank-Nicolson diffusion: 0.5*(nu*∇²(u+u_n))
        md1.add_linear_term(mim, ' 0.5*nu*(Grad_u):Grad_Test_u', FLUID) 
        md1.add_linear_term(mim, ' 0.5*nu*(Grad_u_n):Grad_Test_u', FLUID) 

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
        md1.add_Dirichlet_condition_with_multipliers(mim, "u", mf_v, INLET, "V_inlet")
        md1.add_Dirichlet_condition_with_multipliers(mim, "u", mf_v, WALLS, "V_noslip")
        md1.add_Dirichlet_condition_with_multipliers(mim, "u", mf_v, OBSTACLE, "V_noslip")
        
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
        md3.add_initialized_data("nu", nu)
        md3.add_initialized_data("H", H)
       
        md3.add_linear_term(mim, 'rho*u_new.Test_u_new', FLUID)
        md3.add_linear_term(mim, '-rho*u_star.Test_u_new + dt*Grad_phi.Test_u_new', FLUID)
        # Boundary conditions

        V_inlet= md3.interpolation(V_inlet_expr, mf_v)
        md3.add_initialized_fem_data('V_inlet', mf_v, V_inlet)

        V_noslip = md3.interpolation( "[0,0]" , mf_v)
        md3.add_initialized_fem_data('V_noslip', mf_v, V_noslip)
        
        md3.add_Dirichlet_condition_with_multipliers(mim, "u_new", mf_v, INLET, "V_inlet")
        md3.add_Dirichlet_condition_with_multipliers(mim, "u_new", mf_v, WALLS, "V_noslip")
        md3.add_Dirichlet_condition_with_multipliers(mim, "u_new", mf_v, OBSTACLE, "V_noslip")
        
        md3.solve("noisy", "max_iter", 100, "max_res", 1e-8, "lsolver", "superlu")
        u_new = md3.variable("u_new")


         # Boundary conditions check:
        
        # Get the degrees of freedom on the inlet boundary
        inlet_dofs = mf_v.basic_dof_on_region(INLET)

        # Extract the interpolated inlet velocity values at those DOFs
        V_inlet_at_dofs = V_inlet[inlet_dofs]

        # Extract the computed solution values at those DOFs
        u_new_at_inlet = u_new[inlet_dofs]

        # Compare the values
        diff = np.linalg.norm(u_new_at_inlet - V_inlet_at_dofs)
        relative_diff = diff / np.linalg.norm(V_inlet_at_dofs) if np.linalg.norm(V_inlet_at_dofs) > 0 else diff

        print(f"Inlet BC verification:")
        print(f"  Absolute difference: {diff:.6e}")
        print(f"  Relative difference: {relative_diff:.6e}")
        print(f"  Max absolute difference: {np.max(np.abs(u_new_at_inlet - V_inlet_at_dofs)):.6e}")

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
        
        
        
        md_force.add_initialized_data("nu", nu)
        # Traction: σ·n = [μ(∇u + ∇u^T) - pI]·n
        traction = gf.asm_generic(mim, 0, "(nu*(Grad_u_new + Grad_u_new') - p_new*Id(2))*Normal",OBSTACLE, md_force)
        
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
        div_history.append(div_norm)
        
        print(f"  Cd={Cd:.6f}, Cl={Cl:.6f}, ΔP={p_diff:.6f}")
        

        #################################
        # Export results
        #################################
        
        if step % 200 == 0: # export every 200 steps thus every  0.02s hence there are going to be 500 files
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
        # Save Values if it's neccessary to reinitailize simulation
        #################################
    
        
        np.save(f"{output_dir}/model1_stat.npy", md1.from_variables())
        np.save(f"{output_dir}/model2_stat.npy", md2.from_variables())
        np.save(f"{output_dir}/model3_stat.npy", md3.from_variables())
        #################################
        # Save force coefficients
        #################################

        np.savetxt(f"{output_dir}/force_coefficients_channel_triangle.txt",
                np.column_stack([time_history, cd_history, cl_history, p_diff_history, div_history]),
                header="Time Cd Cl Pressure_Diff, div norm",
                fmt='%.8e')

        print(f"Results saved to {output_dir}/")
        print(f"Final time: {t:.4f}")

    #################################
    # Plot results (optional)
    #################################

try:
    import matplotlib.pyplot as plt

    # Create figure and axes
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot Drag Coefficient
    axes[0].plot(time_history, cd_history, 'b-', linewidth=2)
    axes[0].set_ylabel('Drag Coefficient $C_D$')
    axes[0].grid(True)
    axes[0].set_title('DFG 2D-3 Benchmark Results')

    # Plot Lift Coefficient
    axes[1].plot(time_history, cl_history, 'r-', linewidth=2)
    axes[1].set_ylabel('Lift Coefficient $C_L$')
    axes[1].grid(True)

    # Plot Pressure Difference
    axes[2].plot(time_history, p_diff_history, 'g-', linewidth=2)
    axes[2].set_ylabel('Pressure Difference $\Delta P$')
    axes[2].set_xlabel('Time [s]')
    axes[2].grid(True)

    # Improve layout
    plt.tight_layout()

    # Define the output path and save
    output_path = f"{output_dir}/coefficients.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # close figure to free memory

    print(f"✅ Plot saved successfully to: {output_path}")

except ImportError:
    print("⚠️ Matplotlib not available for plotting — skipping figure creation.")
except Exception as e:
    print(f"❌ Error during plotting: {e}")

