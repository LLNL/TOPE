import firedrake as fd
import numpy as np
from firedrake import inner, grad, dx, dS, jump, sqrt
from firedrake.petsc import PETSc
import firedrake_adjoint as fda
import os
from pyMMAopt import MMASolver
from pyMMAopt import ReducedInequality
import argparse
from parameters import DIRICHLET_1, DIRICHLET_2
from pyadjoint.placeholder import Placeholder
import itertools
import glob
import re
from solver_parameters import solver_options


def print(x):
    return PETSc.Sys.Print(x)


def cp_cost(gamma, p):
    return gamma ** p


def forward(gamma, args, phi_pvd=None, final_stored_cp=None):
    tau = args.tau
    xi = args.xi
    porosity = args.porosity
    filter_radius = args.filter_radius
    effective_porosity = args.effective_porosity
    # 1.5355 time needed for the block porous with effective porosity
    # 0.722 for the simple porosity
    tlimit = 1.0 / xi
    n_steps = args.n_steps
    dt_value = tlimit / n_steps
    porosity = args.porosity
    filter_solver_params, supercap_solver_params = solver_options(dim=args.dim)
    dt = fd.Constant(dt_value)
    print(f"dt_value: {dt_value}")

    V = fd.FunctionSpace(gamma.ufl_domain(), "CG", 1)
    W = V * V
    u, v = fd.TrialFunction(W), fd.TestFunction(W)
    print(f"DOFs: {W.dim()}")

    # Solve the PDE filter
    gamma_f = fd.Function(gamma.function_space())
    flt_problem = filter_problem(
        gamma, gamma_f, filter_radius=filter_radius, dim=args.dim
    )
    flt_solver = fd.LinearVariationalSolver(
        flt_problem, solver_parameters=filter_solver_params
    )
    flt_solver.solve()

    factor_porosity = (
        1.0 if effective_porosity == "simple" else 0.0736806299
    )  # 0.02^(2/3)
    porosity_new = porosity * factor_porosity

    # gamma = 0 ----> epsilon = 1
    # gamma = 1 ----> epsilon = porosity
    epsilon = fd.Constant(1.0) - fd.Constant(1.0 - porosity) * gamma_f
    epsilon_ionic = fd.Constant(1.0) - fd.Constant(1.0 - porosity_new) * gamma_f
    p = 3.0 / 2.0
    kappa_hat = (epsilon_ionic) ** (fd.Constant(p))
    sigma_hat = (fd.Constant(1.0) - epsilon) ** (fd.Constant(p))

    inv_ks_rho = fd.Constant((porosity_new) ** p - 1.0) * gamma_f + fd.Constant(1.0)
    inv_sigma_rho = fd.Constant((1.0 - porosity) ** p) * gamma_f
    cp_hat = gamma_f + 1e-5

    # Potentials function.
    # First component is electronic
    # Second component is ionic
    u_n = fd.Function(W, name="potential")
    a = (
        cp_hat * (u[0] - u[1]) / dt * v[0] * dx
        + fd.Constant(1.0 / tau) * inner(sigma_hat * grad(u[0]), grad(v[0])) * dx
        - cp_hat * (u[0] - u[1]) / dt * v[1] * dx
        + inner(kappa_hat * grad(u[1]), grad(v[1])) * dx
    )
    L = (
        cp_hat * (u_n[0] - u_n[1]) / dt * v[0] * dx
        - cp_hat * (u_n[0] - u_n[1]) / dt * v[1] * dx
    )


    # Boundary conditions
    ud = fd.Constant(0.0)
    bc1 = fd.DirichletBC(W.sub(1), ud, (DIRICHLET_2,))
    bc2 = fd.DirichletBC(W.sub(0), fd.Constant(0.0), (DIRICHLET_1,))
    bc1.apply(u_n)
    bcs = [bc1, bc2]

    u_sol = fd.Function(W)
    problem = fd.LinearVariationalProblem(a, L, u_sol, bcs=bcs)
    solver = fd.LinearVariationalSolver(
        problem, solver_parameters=supercap_solver_params
    )

    energy_resistive_total = 0.0
    energy_resistive_n = 0.0

    t = 0.0
    for _ in range(n_steps):
        solver.solve()

        # Energy resistive (Ohmic losses) domain integration
        energy_resistive_current = fd.assemble(
            energy_resistive_step(u_sol, inv_ks_rho, inv_sigma_rho, tau) * dx
        )
        # Trapezoid time step integration
        energy_resistive_total += (
            dt_value / 2.0 * (energy_resistive_n + energy_resistive_current)
        )
        # Update variables
        energy_resistive_n = energy_resistive_current
        t += dt_value
        u_n.assign(u_sol)
        vd = t * xi
        ud.assign(fd.Constant(vd))

        # Plotting
        if phi_pvd:
            with fda.stop_annotating():
                phi1, phi2 = u_n.split()
                phi1.rename("Phi1")
                phi2.rename("Phi2")
                phi_pvd.write(phi1, phi2)  # , time=t)

    # Plot some quantities for the last time step
    if final_stored_cp:
        plot_final_state(u_n, gamma_f, inv_ks_rho, inv_sigma_rho, tau, final_stored_cp)
    return gamma_f, u_n, energy_resistive_total


def plot_final_state(u_n, gamma_f, inv_ks_rho, inv_sigma_rho, tau, final_stored_cp):
    V = u_n.function_space().sub(0)
    mesh = V.ufl_domain()
    with fda.stop_annotating():
        phi1, phi2 = u_n.split()
        phi1.rename("Phi1")
        phi2.rename("Phi2")
        V_gradient = fd.VectorFunctionSpace(mesh, "CG", 1)
        # Energy stored distribution
        energy_cap = fd.interpolate(
            cp_cost(gamma_f, fd.Constant(3.0)) * (phi1 - phi2) * (phi1 - phi2), V
        )
        energy_cap.rename("Energy stored")
        # Ionic current distribution
        current_2 = fd.project(inv_ks_rho * grad(phi2), V_gradient)
        current_2.rename("Current Phi 2")
        # Electronic current distribution
        current_1 = fd.project(inv_sigma_rho * grad(phi1), V_gradient)
        current_1.rename("Current Phi 1")
        # Energy Loss distribution
        final_energy_resist = fd.interpolate(
            inner(inv_ks_rho * grad(phi2), grad(phi2))
            + inner(fd.Constant(1.0 / tau) * inv_sigma_rho * grad(phi1), grad(phi1)),
            V,
        )
        final_energy_resist.rename("Energy resistance")
        final_stored_cp.write(energy_cap, final_energy_resist, current_1, current_2)


def energy_resistive_step(u_sol, inv_kappa_hat, inv_sigma_hat, tau):
    return inner(inv_kappa_hat * grad(u_sol[1]), grad(u_sol[1])) + inner(
        fd.Constant(1.0) / tau * inv_sigma_hat * grad(u_sol[0]),
        grad(u_sol[0]),
    )


def filter_problem(gamma, gamma_f, filter_radius=1e-4, dim=2):
    GAMMA = gamma.function_space()
    if dim == 2:
        x, y = fd.SpatialCoordinate(GAMMA.ufl_domain())
    elif dim == 3:
        x, y, z = fd.SpatialCoordinate(GAMMA.ufl_domain())
    with fda.stop_annotating():
        x_ = fd.interpolate(x, GAMMA)
        y_ = fd.interpolate(y, GAMMA)
        if dim == 3:
            z_ = fd.interpolate(z, GAMMA)
    if dim == 2:
        Delta_h = sqrt(jump(x_) ** 2 + jump(y_) ** 2)
    elif dim == 3:
        Delta_h = sqrt(jump(x_) ** 2 + jump(y_) ** 2 + jump(z_) ** 2)
    af = fd.TrialFunction(GAMMA)
    b = fd.TestFunction(GAMMA)
    a_filter = filter_radius * jump(af) / Delta_h * jump(b) * dS + af * b * dx
    L_filter = gamma * b * dx
    return fd.LinearVariationalProblem(a_filter, L_filter, gamma_f)


def two_electrode():
    parser = argparse.ArgumentParser(description="Supercapacitor parameters")
    parser.add_argument(
        "--tau",
        action="store",
        dest="tau",
        type=float,
        help="tau: k_eL / Sigma_eD ratio",
        default=0.05,
    )
    parser.add_argument(
        "--xi",
        action="store",
        dest="xi",
        type=float,
        help="Applied non-dimensional scanning rate",
        default=0.01,
    )
    parser.add_argument(
        "--n_steps",
        action="store",
        dest="n_steps",
        type=int,
        help="Number of steps for the transient simulation",
        default=20,
    )
    parser.add_argument(
        "--dim",
        action="store",
        dest="dim",
        type=int,
        help="Problem dimension",
        default=2,
    )
    parser.add_argument(
        "--effective_porosity",
        action="store",
        dest="effective_porosity",
        type=str,
        help="Electrode effective porosity\
              simple or reduced",
        default="reduced",
    )
    parser.add_argument(
        "--porosity",
        action="store",
        dest="porosity",
        type=float,
        help="Electrode porosity",
        default=0.5,
    )
    parser.add_argument(
        "--filter_radius",
        action="store",
        dest="filter_radius",
        type=float,
        help="Filter design radius",
        default=1e-4,
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        dest="output_dir",
        type=str,
        help="Output directory",
        default="./",
    )
    parser.add_argument(
        "--initial_design",
        action="store",
        dest="initial_design",
        type=str,
        help="Choose an option for initial design: uniform, fins or restart (read from file final_design_uniform in the folder)",
        default="uniform",
    )
    parser.add_argument(
        "--initial_gamma_value",
        action="store",
        dest="initial_gamma_value",
        type=float,
        help="Initial value for hte uniform gamma field",
        default=0.5,
    )
    parser.add_argument(
        "--movlim",
        action="store",
        dest="movlim",
        type=float,
        help="Value for MMA movlim",
        default=0.1,
    )
    parser.add_argument(
        "--constraint_value",
        action="store",
        dest="constraint_value",
        type=float,
        help="Constraint value: Min energy stored",
        default=0.5,
    )
    parser.add_argument(
        "--forward",
        action="store_true",
        dest="forward",
        help="Perform only forward simulation",
        default=False,
    )
    parser.add_argument(
        "--continuation",
        action="store",
        type=int,
        dest="continuation",
        help="Perform continuation for constraint penalization",
        default=0,
    )

    args, unknown = parser.parse_known_args()
    print(args)
    output_dir = args.output_dir
    constraint_value = args.constraint_value
    continuation = args.continuation == 1

    if args.dim == 2:
        if args.initial_design == "fins":
            mesh = fd.Mesh("fins_design.msh")
        else:
            mesh = fd.Mesh("electrode_mesh.msh")
        movlim = 0.2
    elif args.dim == 3:
        mesh = fd.Mesh("electrode_3D.msh")
        movlim = 0.05

    GAMMA = fd.FunctionSpace(mesh, "DG", 0)
    gamma = fd.Function(GAMMA)

    # Initialize/Read density field
    with fda.stop_annotating():
        if args.initial_design == "uniform":
            gamma.interpolate(fd.Constant(args.initial_gamma_value))
        elif args.initial_design == "fins":
            ELECTRODE = 1
            fd.par_loop(
                ("{[i] : 0 <= i < f.dofs}", "f[i, 0] = 1.0"),
                dx(ELECTRODE),
                {"f": (gamma, fd.WRITE)},
                is_loopy_kernel=True,
            )
        else:
            final_design_file = f"{output_dir}/final_design_uniform"
            if os.path.isfile(final_design_file):
                with fd.HDF5File(final_design_file, "r") as checkpoint:
                    checkpoint.read(gamma, "/final_design")
            else:
                raise ValueError("Invalid initial_design")


    if args.dim == 2:
        initial_stored_energy = fd.File(
            f"{output_dir}/initial_stored_energy_{args.initial_design}.pvd"
        )
        initial_potential = fd.File(
            f"{output_dir}/initial_potential_{args.initial_design}.pvd"
        )
    else:
        initial_stored_energy = None
        initial_potential = None

    gamma_f, u_n, energy_resistive_total = forward(
        gamma, args, phi_pvd=initial_potential, final_stored_cp=initial_stored_energy
    )

    # Stored energy
    p = fd.Constant(1.0)
    Placeholder(p)
    energy_cap = fd.assemble(
        fd.Constant(1.0 / 2.0)
        * cp_cost(gamma_f, p)
        * (u_n[0] - u_n[1])
        * (u_n[0] - u_n[1])
        * dx
    )
    # Calculate the maximum possible stored energy to use it as a constraint
    with fda.stop_annotating():
        max_energy = fd.assemble(
            fd.Constant(1.0 / 2.0)
            * fd.Constant(1.0)
            * fd.Constant(1.0)
            * dx(domain=mesh),
            annotate=False,
        )
    print(
        f"Energy resistive: {energy_resistive_total}, Energy capacitor: {energy_cap},\
             fraction of Max energy: {energy_cap / max_energy}"
    )
    if args.forward:
        np.savez(
            f"{output_dir}/xi_{args.xi}",
            energy_cap=energy_cap,
            energy_resistive_total=energy_resistive_total,
        )
        fd.File(f"{output_dir}/geometry_{args.initial_design}.pvd").write(gamma_f)
        exit()

    global_counter1 = itertools.count()
    if args.dim == 2:
        tc = None
    elif args.dim == 3:
        tc = fd.H1
    controls_f = fd.File(
        f"{output_dir}/geometry_{args.initial_design}.pvd",
        target_continuity=tc,
        mode="a",
    )
    rhof_control = fda.Control(gamma_f)
    hs_rho_viz = fd.Function(GAMMA, name="gamma")

    def deriv_cb(j, dj, gamma):
        iter = next(global_counter1)
        if iter % 2 == 0:
            with fda.stop_annotating():
                hs_rho_viz.assign(rhof_control.tape_value())
                controls_f.write(hs_rho_viz)

    # Save initial energy loss to use it as scale
    with fda.stop_annotating():
        initial_energy_loss = float(energy_resistive_total)
    m = fda.Control(gamma)
    Jhat = fda.ReducedFunctional(
        1.0 / fda.AdjFloat(initial_energy_loss) * energy_resistive_total,
        m,
        derivative_cb_post=deriv_cb,
    )

    energy_balance = fda.AdjFloat(1.0) / energy_cap
    Plimit = 1.0 / (constraint_value * max_energy)

    Phat = fda.ReducedFunctional(energy_balance, m)
    Pcontrol = fda.Control(energy_balance)

    parameters_mma = {
        "move": movlim,
        "maximum_iterations": 300,
        "m": 1,
        "IP": 0,
        "tol": 1e-7,
        "accepted_tol": 1e-5,
        # "gcmma": True,
        "norm": "L2",
        "output_dir": output_dir,
    }
    loop = 0

    checkpoints = glob.glob(f"{output_dir}/checkpoint*")
    checkpoints_sorted = sorted(
        checkpoints,
        key=lambda L: list(map(int, re.findall(r"iter_(\d+)\.h5", L)))[0],
    )

    if checkpoints_sorted:
        last_file = checkpoints_sorted[-1]
        loop = int(re.findall(r"iter_(\d+)\.h5", last_file)[0])
        parameters_mma["restart_file"] = last_file
    if continuation:
        p_arr = [1.0, 3.0]
        max_iter_arr = [100, 200]
    else:
        p_arr = [3.0]
        max_iter_arr = [300]
    for p_val, max_iter in zip(p_arr, max_iter_arr):
        with fda.stop_annotating():
            p.assign(fd.Constant(p_val))
        parameters_mma["maximum_iterations"] = max_iter

        problem = fda.MinimizationProblem(
            Jhat,
            bounds=(0.0, 1.0),
            constraints=[
                ReducedInequality(Phat, Plimit, Pcontrol),
            ],
        )

        solver = MMASolver(problem, parameters=parameters_mma)

        results = solver.solve(loop=loop)
        rho_opt = results["control"]
        with fda.stop_annotating():
            gamma.assign(rho_opt)

    hs_rho_viz.assign(rhof_control.tape_value())
    with fd.HDF5File(
        f"{output_dir}/final_design_{args.initial_design}", "w"
    ) as checkpoint:
        checkpoint.write(hs_rho_viz, "/final_design")

    final_potential = fd.File(f"{output_dir}/final_potential_{args.initial_design}.pvd")
    if args.dim == 2:
        final_stored_energy = fd.File(
            f"{output_dir}/final_stored_energy_{args.initial_design}.pvd"
        )
    else:
        final_stored_energy = None
    gamma_f, u_n, energy_resistive_total = forward(
        rho_opt, args, phi_pvd=final_potential, final_stored_cp=final_stored_energy
    )


if __name__ == "__main__":
    two_electrode()
