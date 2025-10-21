import getfem as gf 
import numpy as np
import os
from Functions import verify_regions

###########################
# DFG 2D-3 Benchmark: Cylinder in Channel
###########################


# Generate the Mesh_fluid
if __name__ == "__main__":

    output_dir = f"fluid/results_cylinder_open_domain2_TRI"
    ##################
    ## PROBLEM DATA ##
    ##################
    """
    The fluid will be the air
    """

    scale_factor = 1 #m -> m  (it can be used if we want to pass from meter to anything else cm, dm etc...) 


    # Fluid properties (air):
    #ν_fluid = 1.516e-5 * scale_factor**2                # m²/s 
    ν_fluid = 1/60 * scale_factor**2  
    rho_fluid = 1.204 /(scale_factor**3  )               # kg/m³ 
    mu_fluid = rho_fluid * ν_fluid                             # Dynamic viscosity

    # Boundaries values: 
    # Inlet velocity parameters
    U_mean = 1     # Mean inlet velocity m/s

    # Reference values for coefficients
    D = 0.1  # Cylinder diameter (you need to set this based on your Mesh_fluid)
    A_ref = D   # Reference area (diameter × unit depth for 2D)
    q_inf = 0.5 * rho_fluid * U_mean**2  # Dynamic pressure


    # Transient paramters: 
    T = 5.0           # Total simulation time
    dt = 1e-3      # Time step
    theta = 0.5      # Theta parameter (0.5 = Crank-Nicolson)
    num_steps = int(T / dt)
    Re = U_mean * D / ν_fluid
    print(f"Reynolds number is: {Re}")


    ############
    ## Mesh_fluid ##
    #############

    Mesh_fluid= gf.Mesh('Import', 'gmsh','fluid/Mesh/cylinder_tri_4p1.msh')
    #Mesh_fluid.export_to_vtk('fluid/Mesh_fluid/Cylinder_open_domain.vtk') 
    h = min(Mesh_fluid.convex_radius())
    print( f"Minimum mesh size h ={h}, and CFL = "f"{1}, thus dt = {dt} should be less than {1*h/U_mean}" )

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
    print("Regions in the Mesh_fluid are:", Mesh_fluid.regions())

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

    mim = gf.MeshIm(Mesh_fluid, gf.Integ('IM_TRIANGLE(5)'))

    #########################
    ## FEM ELEMENTS ##
    #########################

    # Velocity: P2 elements (quadratic)
    mf_v = gf.MeshFem(Mesh_fluid, 2)
    mf_v.set_fem(gf.Fem('FEM_PK(2,2)'))

    # Pressure: P1 elements (linear)
    mf_p = gf.MeshFem(Mesh_fluid, 1)
    mf_p.set_fem(gf.Fem('FEM_PK(2,1)'))

    print(f"Velocity DOFs: {mf_v.nbdof()}")
    print(f"Pressure DOFs: {mf_p.nbdof()}")

    ########################
    ## INITIAL CONDITIONS ##
    ########################

    u_n = np.zeros(mf_v.nbdof())      # u^n
    u_n1 = np.zeros(mf_v.nbdof())     # u^{n-1}
    p_n = np.zeros(mf_p.nbdof())      # p^n

    #verify_regions(Mesh_fluid, 'fluid/Mesh_fluid/meshfluid_cylinder_channel')

    ####################
    ## SOLVER SETUP   ##
    ####################

    
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
        # Solve: rho_fluid/dt*(u* - u^n) + rho_fluid*(1.5*u^n - 0.5*u^{n-1})·∇u*
        #        + mu_fluid*∇²u* = 0
        
        md1 = gf.Model("real")
        

        md1.add_fem_variable("u_star", mf_v) # u i s the variable
        md1.add_fem_data("u_n", mf_v) # u is the previus step
        md1.add_fem_data("u_n1", mf_v) # u is the previus to the previus step
        md1.add_fem_data("p_n", mf_p) # p is the previus step
        
        md1.set_variable("u_n", u_n)    # set the previus step u, at the beginning is zero
        md1.set_variable("u_n1", u_n1)  # set the previus step of the previus step u, at the beginning is zero
        md1.set_variable("p_n", p_n)    # set the previus step p, at the beginning is zero
        
        md1.add_initialized_data("rho_fluid", rho_fluid)
        md1.add_initialized_data("mu_fluid", mu_fluid)
        md1.add_initialized_data("U_mean", U_mean)
        md1.add_initialized_data("dt", dt)
        
        # Time derivative
        md1.add_linear_term(mim, '(rho_fluid/dt)*(u_star - u_n).Test_u_star', FLUID)
        
        # Convection (Adams-Bashforth): (1.5*u_n - 0.5*u_n1)*0.5*Grad(u+un)
        md1.add_nonlinear_term(mim,
            '0.5*rho_fluid*((1.5*u_n - 0.5*u_n1).Grad_u_star).Test_u_star', FLUID)
        md1.add_linear_term(mim,
            '0.5*rho_fluid*((1.5*u_n - 0.5*u_n1).Grad_u_n).Test_u_star', FLUID)
        
        # Crank-Nicolson diffusion: - 0.5*(mu_fluid*∇²(u+u_n)) => + 0.5(mu_fluid*∇(u+u_n):∇(u_test)) -int(mu_fluid*∇(u+u_n)*n)u_test on outflow
        md1.add_linear_term(mim, '0.5*mu_fluid*(Grad_u_star):Grad_Test_u_star', FLUID)  
        md1.add_linear_term(mim, '0.5*mu_fluid*(Grad_u_n):Grad_Test_u_star', FLUID) 

        # Pressure from previous step
        md1.add_linear_term(mim, ' - p_n*Trace(Grad_Test_u_star)', FLUID)
        
        # Boundary conditions
        # Inlet velocity profile with ramp-up

        inlet_dofs = mf_v.basic_dof_on_region(INLET)

        # t_ramp = 50 * dt  # = 0.001 s
        # if t < t_ramp:
        #     ramp_factor = np.sin(0.5 * np.pi * t / t_ramp)
        # else:
        #     ramp_factor = 1.0 
        ramp_factor = 1.0
        V_inlet_expr = f"{ramp_factor}*[U_mean, 0]"
        V_inlet= md1.interpolation(V_inlet_expr, mf_v) # Interpolation over the all domain. 
        md1.add_initialized_fem_data('V_inlet', mf_v, V_inlet)


        V_noslip = md1.interpolation( "[0,0]" , mf_v)
        md1.add_initialized_fem_data('V_noslip', mf_v, V_noslip)
        
        # Boundary conditions
        md1.add_Dirichlet_condition_with_multipliers(mim, "u_star", mf_v, INLET, "V_inlet") # Here is applaying only at the inlet even though the data is defined everywhere.
        md1.add_Dirichlet_condition_with_multipliers(mim, "u_star", mf_v, WALLS, "V_inlet")
        md1.add_Dirichlet_condition_with_multipliers(mim, "u_star", mf_v, CYLINDER_INTERFACE, "V_noslip")
        
        # Solve
        md1.solve("noisy", "max_iter", 100, "max_res", 1e-8, "lsolver", "superlu")
        u_star = md1.variable("u_star")

        #################################
        # STEP 2: Pressure correction
        #################################
        # Solve: ∇²φ = (rho_fluid/dt)*∇·u*
        
        md2 = gf.Model("real")
        md2.add_fem_variable("phi", mf_p)
        md2.add_fem_data("u_star", mf_v)
        md2.set_variable("u_star", u_star)
        md2.add_initialized_data("rho_fluid", rho_fluid)
        md2.add_initialized_data("dt", dt)
        
        # Poisson equation
        md2.add_linear_term(mim, 'Grad_phi.Grad_Test_phi', FLUID)
        md2.add_linear_term(mim, '(rho_fluid/dt)*Trace(Grad_u_star)*Test_phi', FLUID)
        
        # BC: φ = 0 at outlet
        md2.add_Dirichlet_condition_with_multipliers(mim, "phi", mf_p, OUTLET)
        
        md2.solve("noisy", "max_iter", 100, "max_res", 1e-8, "lsolver", "superlu")   
        phi = md2.variable("phi")
        # Update pressure: p^{n+1} = p^n + φ

        
        #################################
        # STEP 3: Velocity correction
        #################################
        # u^{n+1} = u* - (dt/rho_fluid)*∇φ
        
        # Mass matrix M = ∫ ρ * u · v dx
        md3 = gf.Model("real")
        md3.add_fem_variable("u_new", mf_v)
        md3.add_fem_data("phi", mf_p)
        md3.set_variable("phi", phi)
        md3.add_fem_data("u_star", mf_v)
        md3.set_variable("u_star", u_star)

        md3.add_initialized_data("rho_fluid", rho_fluid)
        md3.add_initialized_data("dt", dt)
        md3.add_initialized_data("mu_fluid", mu_fluid)
        md3.add_initialized_data("U_mean", U_mean)
        md3.add_linear_term(mim, 'rho_fluid*u_new.Test_u_new', FLUID)
        md3.add_linear_term(mim, '-rho_fluid*u_star.Test_u_new + dt*Grad_phi.Test_u_new', FLUID)


        # md3.add_initialized_fem_data('V_inlet', mf_v, V_inlet)
        # md3.add_initialized_fem_data('V_noslip', mf_v, V_noslip)
        
        V_inlet= md3.interpolation(V_inlet_expr, mf_v) # I'm interpolating over the all domain. 
        md3.add_initialized_fem_data('V_inlet', mf_v, V_inlet)
        V_noslip = md3.interpolation( "[0,0]" , mf_v)
        md3.add_initialized_fem_data('V_noslip', mf_v, V_noslip)
        
        md3.add_Dirichlet_condition_with_multipliers(mim, "u_new", mf_v, INLET, "V_inlet")
        md3.add_Dirichlet_condition_with_multipliers(mim, "u_new", mf_v, WALLS, "V_inlet")
        md3.add_Dirichlet_condition_with_multipliers(mim, "u_new", mf_v, CYLINDER_INTERFACE, "V_noslip")


        md3.solve("noisy", "max_iter", 100, "max_res", 1e-8, "lsolver", "superlu")
        u_new = md3.variable("u_new")
      
        
        
        
        # L2 norm of the velocity divergence
        div_norm = gf.asm_generic(mim, 0,'(Trace(Grad_u_new))',FLUID, md3 )
        #div_norm = np.sqrt(div_norm2)
        print(f"‖div(u_new)‖ₗ₂ = {div_norm:.6e}")
        #################################
        # Compute drag and lift
        #################################
        
        #if step % 50 == 0:
        md_force = gf.Model("real")
        md_force.add_fem_data("p_new", mf_p)
        md_force.set_variable("p_new", p_new)
        md_force.add_fem_data("u_new", mf_v)
        md_force.set_variable("u_new", u_new)
        md_force.add_initialized_data("mu_fluid", mu_fluid)
        
        # Traction: σ·n = [μ(∇u + ∇u^T) - pI]·n
        traction = gf.asm_generic(mim, 0, "(mu_fluid*(Grad_u_new + Grad_u_new') - p_new*Id(2))*Normal",CYLINDER_INTERFACE, md_force)
        
        Fx = -traction[0]
        Fy = -traction[1]
        
        # Drag and lift coefficients           
        Cd = 2 * Fx / (rho_fluid * U_mean**2 * D)
        Cl = 2 * Fy / (rho_fluid * U_mean**2 * D)
        
        time_history.append(t)
        cd_history.append(Cd)
        cl_history.append(Cl)
        
        print(f"  Cd={Cd:.6f}, Cl={Cl:.6f}")
            

        #################################
        # Export results
        #################################
        
        #if step % 25 == 0: # export every 25 steps thus every 0.003788s, and there will be 64 files in total 
        time = step * dt
        mf_v.export_to_vtu(f"{output_dir}/velocity_{time:.4f}.vtu",
                        mf_v, u_new, "Velocity",
                        mf_p, p_new, "Pressure")
          
        #################################
        # Update for next time step
        #################################
        
        u_n1 = u_n.copy()
        u_n = u_new.copy()
        p_new = p_n + phi
        p_n = p_new.copy()

        #################################
        # Save force coefficients
        #################################

        np.savetxt(f"fluid/force_coefficients_open_triangle.txt",
                np.column_stack([time_history, cd_history, cl_history, div_norm]),
                header="Time Cd Cl",
                fmt='%.8e')

        print(f"Results saved to {output_dir}/")
    print(f"Final time: {t:.4f}")



