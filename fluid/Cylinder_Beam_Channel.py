import getfem as gf 
import numpy as np
import os
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import gmsh

## FSI fluid benchmark from the model proposed by S.Turek and J.Hron

"""
This is done using the formulation benchmarked previously for the cylinder in a channel
"""

# Physical groups
FLUID = 1
INLET= 2
OUTLET = 3
WALLS = 4
CYLINDER = 5
BEAM = 6

def create_CylBeam_mesh(mesh_file="cylinder_beam_channel.msh", visualize=False):
    """
    Create QUADRILATERAL mesh for cylinder-beam assembly in channel
    """
    
    gmsh.initialize()
    gmsh.model.add("Cylinder_Beam_Channel")
    
    # Geometry
    L = 2.2
    H = 0.41
    c_x = 0.2
    c_y = 0.2
    r = 0.05
    L_beam = 0.4
    W_beam = 0.02
    gdim = 2
    
    # Create geometry
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    cylinder = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
    beam = gmsh.model.occ.addRectangle(c_x + r, c_y - W_beam/2, 0, L_beam, W_beam)
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, cylinder), (gdim, beam)], removeObject=True, removeTool=False)
    gmsh.model.occ.synchronize()
    
    # Physical groups
    FLUID = 1
    INLET= 2
    OUTLET = 3
    WALLS = 4
    CYLINDER = 5
    BEAM = 6
    
    volumes = gmsh.model.getEntities(dim=gdim)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], FLUID)
    gmsh.model.setPhysicalName(volumes[0][0], FLUID, "FLUID")
    
    # --------------------------
    # 1D boundaries
    # --------------------------
    # Cylinder and beam edges
    cylinder_bnd = gmsh.model.getBoundary([(gdim, cylinder)], oriented=False)
    beam_bnd     = gmsh.model.getBoundary([(gdim, beam)], oriented=False)

    cylinder_edges = [b[1] for b in cylinder_bnd]
    beam_edges     = [b[1] for b in beam_bnd]

    gmsh.model.addPhysicalGroup(1, cylinder_edges, CYLINDER)
    gmsh.model.setPhysicalName(1, CYLINDER, "CYLINDER")
    gmsh.model.addPhysicalGroup(1, beam_edges, BEAM)
    gmsh.model.setPhysicalName(1, BEAM, "BEAM")

    # Boundary classification
    inflow, outflow, walls = [], [], []
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
    
    gmsh.model.addPhysicalGroup(1, inflow,INLET)
    gmsh.model.setPhysicalName(1, INLET, "INLET")
    gmsh.model.addPhysicalGroup(1, outflow, OUTLET)
    gmsh.model.setPhysicalName(1, OUTLET, "OUTLET")
    gmsh.model.addPhysicalGroup(1, walls, WALLS)
    gmsh.model.setPhysicalName(1, WALLS, "WALLS")
    
    # --------------------------
    # Mesh fields
    # --------------------------

    fields = []

    # Cylinder refinement (Distance + Threshold)
    d_cyl = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(d_cyl, "EdgesList", cylinder_edges)

    t_cyl = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(t_cyl, "IField", d_cyl)
    gmsh.model.mesh.field.setNumber(t_cyl, "LcMin", r / 3)
    gmsh.model.mesh.field.setNumber(t_cyl, "LcMax", 0.08)
    gmsh.model.mesh.field.setNumber(t_cyl, "DistMin", r)
    gmsh.model.mesh.field.setNumber(t_cyl, "DistMax", 3*r)

    fields.append(t_cyl)

    # Beam refinement (BOX — avoids corner collapse)
    box = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(box, "VIn", W_beam / 5)
    gmsh.model.mesh.field.setNumber(box, "VOut", 0.06)
    gmsh.model.mesh.field.setNumber(box, "XMin", c_x + r - 0.02)
    gmsh.model.mesh.field.setNumber(box, "XMax", c_x + r + L_beam + 0.02)
    gmsh.model.mesh.field.setNumber(box, "YMin", c_y - W_beam/2 - 0.02)
    gmsh.model.mesh.field.setNumber(box, "YMax", c_y + W_beam/2 + 0.02)

    fields.append(box)

    # Combine fields
    bg = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(bg, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(bg)

    
    # --------------------------
    # Quadrilateral mesh
    # --------------------------
    gmsh.option.setNumber("Mesh.Algorithm", 8)            # DelQuad / Q4
    gmsh.option.setNumber("Mesh.RecombineAll", 1)         # recombine triangles → quads
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
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


def create_CylBeam_mesh_FSI(
    mesh_file="Mesh/cylinder_beam_channel_FSI.msh",
    level=0,
    visualize=False
):
    """
    Pure TRIANGULAR mesh for cylinder + rigid beam in channel.
    Five refinement levels (0–4) suitable for convergence studies.
    """

    assert 0 <= level <= 4, "level must be between 0 and 4"

    gmsh.initialize()
    gmsh.model.add("Cylinder_Beam_Channel_TRI")

    # -----------------------
    # Geometry parameters
    # -----------------------
    L = 2.5
    H = 0.41

    c_x = 0.2
    c_y = 0.2
    r = 0.05

    L_beam = 0.35
    W_beam = 0.02

    gdim = 2

    # -----------------------
    # Refinement levels
    # -----------------------
    levels = [
        dict(lc_cyl=r/2.5,  lc_beam=W_beam/1.5, lc_far=0.15),
        dict(lc_cyl=r/3.5,  lc_beam=W_beam/2.2, lc_far=0.10),
        dict(lc_cyl=r/5.0,  lc_beam=W_beam/3.2, lc_far=0.065),
        dict(lc_cyl=r/7.0,  lc_beam=W_beam/4.5, lc_far=0.045),
        dict(lc_cyl=r/10.0, lc_beam=W_beam/6.0, lc_far=0.030),
    ]

    lc_cyl  = levels[level]["lc_cyl"]
    lc_beam = levels[level]["lc_beam"]
    lc_far  = levels[level]["lc_far"]

    # -----------------------
    # Geometry
    # -----------------------
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
    cylinder  = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
    beam      = gmsh.model.occ.addRectangle(
        c_x + r, c_y - W_beam / 2, 0, L_beam, W_beam
    )

    fluid, _ = gmsh.model.occ.cut(
        [(2, rectangle)],
        [(2, cylinder), (2, beam)],
        removeObject=True,
        removeTool=False
    )

    gmsh.model.occ.synchronize()

    # -----------------------
    # Physical groups
    # -----------------------
    FLUID, INLET, OUTLET, WALLS, CYLINDER, BEAM = 1, 2, 3, 4, 5, 6

    gmsh.model.addPhysicalGroup(2, [fluid[0][1]], FLUID)
    gmsh.model.setPhysicalName(2, FLUID, "FLUID")

    cyl_bnd  = gmsh.model.getBoundary([(2, cylinder)], oriented=False)
    beam_bnd = gmsh.model.getBoundary([(2, beam)], oriented=False)

    gmsh.model.addPhysicalGroup(1, [b[1] for b in cyl_bnd], CYLINDER)
    gmsh.model.addPhysicalGroup(1, [b[1] for b in beam_bnd], BEAM)

    # Channel boundaries
    inflow, outflow, walls = [], [], []
    for dim, tag in gmsh.model.getBoundary(fluid, oriented=False):
        x, y, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        if np.isclose(x, 0.0, atol=1e-6):
            inflow.append(tag)
        elif np.isclose(x, L, atol=1e-6):
            outflow.append(tag)
        elif np.isclose(y, 0.0, atol=1e-6) or np.isclose(y, H, atol=1e-6):
            walls.append(tag)

    gmsh.model.addPhysicalGroup(1, inflow, INLET)
    gmsh.model.addPhysicalGroup(1, outflow, OUTLET)
    gmsh.model.addPhysicalGroup(1, walls, WALLS)

    # -----------------------
    # Mesh size fields
    # -----------------------
    f_cyl = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(
        f_cyl, "EdgesList", [b[1] for b in cyl_bnd]
    )

    f_beam = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(
        f_beam, "EdgesList", [b[1] for b in beam_bnd]
    )

    th_cyl = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(th_cyl, "IField", f_cyl)
    gmsh.model.mesh.field.setNumber(th_cyl, "LcMin", lc_cyl)
    gmsh.model.mesh.field.setNumber(th_cyl, "LcMax", lc_far)
    gmsh.model.mesh.field.setNumber(th_cyl, "DistMin", r)
    gmsh.model.mesh.field.setNumber(th_cyl, "DistMax", 4*r)

    th_beam = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(th_beam, "IField", f_beam)
    gmsh.model.mesh.field.setNumber(th_beam, "LcMin", lc_beam)
    gmsh.model.mesh.field.setNumber(th_beam, "LcMax", lc_far)
    gmsh.model.mesh.field.setNumber(th_beam, "DistMin", W_beam)
    gmsh.model.mesh.field.setNumber(th_beam, "DistMax", 4*W_beam)

    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(
        f_min, "FieldsList", [th_cyl, th_beam]
    )
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    # -----------------------
    # Mesh generation
    # -----------------------
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(mesh_file)

    if visualize:
        gmsh.fltk.run()

    gmsh.finalize()



# Generate the mesh
if __name__ == "__main__":

    # ##################
    # ## MESH creation##
    # ##################

    # Generate the mesh
    # create_CylBeam_mesh("Mesh/quad_cylinder_beam_channel.msh", visualize=False)
    create_CylBeam_mesh_FSI("Mesh/cylinder_beam_channel_FSI.msh", level=0, visualize=False)

    ###################
    ## MESH imported ##
    ###################
    Mesh = gf.Mesh('Import', 'gmsh','Mesh/cylinder_beam_channel_FSI.msh')

    
    ##################
    ## PROBLEM DATA ##
    ##################

    # Geometry parameters
    L = 2.5         # Channel length
    H = 0.41        # Channel height
    c_x = 0.2       # Cylinder center x
    c_y = 0.2       # Cylinder center y
    r = 0.05        # Cylinder radius
    L_beam = 0.35    # Beamn length
    W_beam = 0.02   # Beam width

    # Fluid properties
    mu = 0.001   # Dynamic viscosity (Pa·s)
    rho = 1000.0    # Density (kg/m³)

    # Inlet velocity parameters
    U_mean = 1  # Maximum inlet velocity (m/s)

    # Time parameters:
    # Time step recommendation:
    # - 0 > 0.005 s
    # - 1 > 0.0035
    # - 2 > 0.0025
    # - 3 > 0.0015
    # - 4 > 0.001

    h = min(Mesh.convex_radius())
    print( f"Minimum mesh size h ={h}, and CFL = "f"{1}, thus dt should be less than {1*h/(1.5*U_mean)}" )
    T = 10.0      # Total simulation time (s) - reduced for testing
    dt = 0.0005  # Time step
    num_steps = int(T/ dt)

    print(f"Reynolds number (based on cylinder diameter): {rho * 3/2*U_mean * (2*r) / mu}")
    print(f"Number of time steps: {num_steps}")

    ########################
    ## INTEGRATION METHOD ##
    ########################

    mim = gf.MeshIm(Mesh, gf.Integ("IM_TRIANGLE(6)"))

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

    output_dir = "results_CylBeam"
    os.makedirs(output_dir, exist_ok=True)

    # Storage for results
    time_history = []
    cd_history = []
    cl_history = []
    p_diff_history = []
    div_history = []

    # Points for pressure evaluation
    p_front_point = np.array([[0.15], [0.2]])
    p_back_point = np.array([[0.6], [0.2]])

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

        md1.add_fem_variable("u", mf_v) # u is the variable - tentative velocity
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
        md1.add_nonlinear_term(mim, '(1/dt)*u.Test_u', FLUID)
        md1.add_source_term(mim, '(1/dt)*u_n.Test_u', FLUID)
        
        # Convection (Adams-Bashforth): (1.5*u_n - 0.5*u_n1)*0.5*Grad(u+un)
        md1.add_linear_term(mim,
            '0.5*((1.5*u_n - 0.5*u_n1).(Grad_u)).Test_u', FLUID)
        md1.add_linear_term(mim,
            '-0.5*((1.5*u_n - 0.5*u_n1).(Grad_Test_u)).u', FLUID)
        # md1.add_source_term(mim,
        #      '((1.5*u_n - 0.5*u_n1).(Grad_u_n)).Test_u', FLUID)
        
        # Crank-Nicolson diffusion: 0.5*(mu*∇²(u+u_n))
        md1.add_linear_term(mim, ' 0.5*mu*(Grad_u:Grad_Test_u)', FLUID) 
        md1.add_source_term(mim, ' -0.5*mu*(Grad_u_n:Grad_Test_u)', FLUID)

        # Pressure from previous step
        md1.add_source_term(mim, 'p_n*Div_Test_u', FLUID)

        # Boundary conditions
        # Inlet velocity profile with ramp-up

        inlet_dofs = mf_v.basic_dof_on_region(INLET)
        if t < 2:
            ramp_factor = U_mean*(1 - np.cos(np.pi * t/2)/2)
        else:
            ramp_factor = U_mean

        V_inlet_expr = f"{ramp_factor}*1.5*4*[X(2)*(H-X(2))/(H/2*H/2), 0]"
        V_inlet= md1.interpolation(V_inlet_expr, mf_v)
        md1.add_initialized_fem_data('V_inlet', mf_v, V_inlet)

        V_noslip = md1.interpolation( "[0,0]" , mf_v)
        md1.add_initialized_fem_data('V_noslip', mf_v, V_noslip)
        
        # Boundary conditions
        md1.add_Dirichlet_condition_with_multipliers(mim, "u", mf_v, INLET, "V_inlet")
        # md1.add_Dirichlet_condition_with_multipliers(mim, "u", mf_v, WALLS, "V_noslip")
        # md1.add_Dirichlet_condition_with_multipliers(mim, "u", mf_v, OBSTACLE, "V_noslip")
        
        #Imposing BC by simplification yields better results (lower divergence)
        # md1.add_Dirichlet_condition_with_simplification('u', INLET, 'V_inlet')
        md1.add_Dirichlet_condition_with_simplification('u', WALLS)
        md1.add_Dirichlet_condition_with_simplification('u', CYLINDER)
        md1.add_Dirichlet_condition_with_simplification('u', BEAM)

        # Solve
        md1.solve("noisy", "max_iter", 100, "max_res", 1e-8, "lsolver", "mumps")
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
        md2.add_source_term(mim, '-(Div_u_star*Test_phi)', FLUID)
        
        # BC: φ = 0 at outlet
        # phi_outler = md1.interpolation( "0" , mf_p)
        # md1.add_initialized_fem_data('phi_outlet', mf_p, phi_outlet)
        md2.add_Dirichlet_condition_with_simplification('phi', OUTLET)
        
        md2.solve("noisy", "max_iter", 100, "max_res", 1e-8, "lsolver", "mumps")   
        phi = md2.variable("phi")

        #Boundary condition check 
        outlet_dofs = mf_p.basic_dof_on_region(OUTLET)
        phi_outlet = phi[outlet_dofs]

        # Compare the values
        diff = np.linalg.norm(phi_outlet)

        print(f"Outlet BC verification:")
        print(f"  Phi at outlet: {diff:.6e}")
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
       
        md3.add_linear_term(mim, '(u_new.Test_u_new)', FLUID)
        md3.add_source_term(mim, 'u_star.Test_u_new + (phi*Div_Test_u_new)', FLUID)

        md3.solve("noisy", "max_iter", 100, "max_res", 1e-8, "lsolver", "mumps")
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
        p_new = p_n + phi/dt
        
        # L2 norm of the velocity divergence
        div_norm2 = gf.asm_generic(mim, 0,'pow(Div_u_new,2)',FLUID, md3 )
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
        md_force.add_initialized_data("rho", rho)

        # Traction: t = σ·n = [μ(∇u + ∇u^T) - pI]·n
        traction_beam = gf.asm_generic(mim, 0, "mu*((Grad_u_new + Grad_u_new') - p_new*Id(2))*Normal", BEAM, md_force)
        traction_cyl = gf.asm_generic(mim, 0, "mu*((Grad_u_new + Grad_u_new') - p_new*Id(2))*Normal", CYLINDER, md_force)
        traction = traction_beam + traction_cyl
        Fx = -traction[0]
        Fy = traction[1]

    #     Cd = gf.asm_generic(mim, 0, "-2/0.1 * ( \
    #   mu/rho*( (Grad_u_new(1,1)*Normal(2) - Grad_u_new(2,1)*Normal(1))*Normal(1) + \
    #             (Grad_u_new(1,2)*Normal(2) - Grad_u_new(2,2)*Normal(1))*Normal(2) ) * Normal(2) \
    #   - p_new*Normal(1) \
    # )", BEAM, md_force)
    #     Cl = gf.asm_generic(mim, 0, "-2/0.1 * ( \
    #   mu/rho*( (Grad_u_new(1,1)*Normal(2) - Grad_u_new(2,1)*Normal(1))*Normal(1) + \
    #              (Grad_u_new(1,2)*Normal(2) - Grad_u_new(2,2)*Normal(1))*Normal(2) ) * Normal(1) \
    #   + p_new*Normal(2) \
    # )", BEAM, md_force)

        # Drag and lift coefficients
        S = 2*L_beam + W_beam # Diameter
        
        Cd = 2 * Fx / (rho * U_mean**2 * S)
        Cl = 2 * Fy / (rho * U_mean**2 * S)
        
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
            
            try:
                import matplotlib.pyplot as plt

                # Create figure and axes
                fig, axes = plt.subplots(3, 1, figsize=(12, 10))

                # Plot Drag Coefficient
                axes[0].plot(time_history, cd_history, 'b-', linewidth=2)
                axes[0].set_ylabel('Drag Coefficient $C_D$')
                axes[0].grid(True)
                axes[0].set_title('Reaction loading on the beam')

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
            
        #################################
        # Update for next time step
        #################################
        
        u_n1 = u_n.copy()
        u_n = u_new.copy()
        p_n = p_new.copy()
        
        #################################
        # Save Values if it's neccessary to reinitailize simulation
        #################################
    
        
        np.save("model1_stat.npy", md1.from_variables())
        np.save("model2_stat.npy", md2.from_variables())
        np.save("model3_stat.npy", md3.from_variables())
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




