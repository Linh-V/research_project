# PROJECT GRADIENT
md_grad = gf.Model("real")
md_grad.add_fem_variable("grad_phi", mf_v)
md_grad.add_fem_data("phi", mf_p)
md_grad.set_variable("phi", phi)

md_grad.add_linear_term(mim, 'grad_phi.Test_grad_phi', FLUID)
md_grad.add_linear_term(mim, '-Grad_phi.Test_grad_phi', FLUID)

md_grad.solve("noisy", "max_iter", 100, "max_res", 1e-8)
grad_phi = md_grad.variable("grad_phi")

# STEP 3: Explicit update
u_new = u_star - (dt / rho_fluid) * grad_phi
p_new = p_n + phi