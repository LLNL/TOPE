from firedrake import *
from firedrake_adjoint import *
from pyMMAopt import MMASolver
from pyMMAopt import ReducedInequality
import argparse
from petsc4py import PETSc


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--Ny",  type=int, default=100)
parser.add_argument("--filter_radius", type=float, default=0.01)
parser.add_argument("--porosity", type=float, default=0.5)
parser.add_argument("--tau", type=float, default=0.5)
parser.add_argument("--delta", type=float, default=15.0)
parser.add_argument("--mu", type=float, default=5.0)
parser.add_argument("--movlim", type=float, default=0.1)
parser.add_argument("--maxiters", type=int, default=200)
parser.add_argument("--output_dir", type=str, default="./results")
parser.add_argument("--effective_porosity", type=str, default="simple")
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--forward", action="store_true", default=False)
parser.add_argument("--initial_design", type=str, default="uniform")
parser.add_argument("--initial_gamma_value", type=float, default=0.5)


args, _ = parser.parse_known_args()

PETSc.Sys.Print(args)
Ny = args.Ny
RH = args.filter_radius
porosity = args.porosity
effective_porosity = args.effective_porosity
tau = args.tau
delta = args.delta
mu = args.mu
movlim = args.movlim
max_iters = args.maxiters
output_dir = args.output_dir
isforward = args.forward

if args.dim == 2:
    target_continuity = None
elif args.dim ==3:
    target_continuity = H1 #to allow isocontours

# Load mesh
if args.dim == 2:
    Nx = 2*Ny
    Lm = 1.5
    Lw = 0.25
    Lx = 2.
    Ly = 1.
    if Ny>0:
        from utils.mesh2 import RectangleMesh2
        mesh = RectangleMesh2(Nx, Ny, Lm, Lw, Ly)
    elif args.initial_design == "fins":
        mesh = Mesh("fins_design.msh")
    else:
        mesh = Mesh('electrode_mesh.msh')
    x,y = SpatialCoordinate(mesh)
elif args.dim == 3:
    Lm = 0.75
    Lw = 0.25
    Ly = 1.
    if Ny>0:
        Ny = 100
        Nx = Ny
        from utils.mesh2 import BoxMesh3
        mesh = BoxMesh3(Nx, Ny, Nx, Lm, Lw, Ly)
    else:
        mesh = Mesh('electrode_3D.msh')
    x, y, z = SpatialCoordinate(mesh)

# initial design
GAMMA = FunctionSpace(mesh, "DG", 0)
gamma = Function(GAMMA)
with stop_annotating():
    if args.initial_design == "uniform":
        gamma.interpolate(Constant(args.initial_gamma_value))
    elif args.initial_design == "fins":
        ELECTRODE = 1
        par_loop(
            ("{[i] : 0 <= i < f.dofs}", "f[i, 0] = 1.0"),
            dx(ELECTRODE),
            {"f": (gamma, WRITE)},
            is_loopy_kernel=True,
        )
    else:
        final_design_file = f"{output_dir}/final_design_uniform"
        if os.path.isfile(final_design_file):
            with HDF5File(final_design_file, "r") as checkpoint:
                checkpoint.read(gamma, "/final_design")
        else:
            raise ValueError("Invalid initial_design")

# filtering
af, b = TrialFunction(GAMMA), TestFunction(GAMMA)
x_ = interpolate(x, GAMMA)
y_ = interpolate(y, GAMMA)
if args.dim == 3:
    z_ = interpolate(z, GAMMA)

if args.dim == 2:
    Delta_h = sqrt(jump(x_) ** 2 + jump(y_) ** 2)
elif args.dim == 3:
    Delta_h = sqrt(jump(x_) ** 2 + jump(y_) ** 2 + jump(z_) ** 2)
aH = RH**2 * jump(af) / Delta_h * jump(b) * dS + af * b * dx
LH = gamma * b * dx
lu = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    }
cg_hypre = {"mat_type": "aij",
         "ksp_rtol": 1e-4,
         "ksp_type": "cg",
         "pc_type": "hypre",
         }
gammaf = Function(GAMMA, name="gamma")
if args.dim == 2:
    helmholtz_parameters = lu
else:
    helmholtz_parameters = cg_hypre
solve(aH == LH, gammaf, solver_parameters=helmholtz_parameters)
gammafControl = Control(gammaf)

# forward problem
V = FunctionSpace(mesh, "CG", 1)
W = V*V

PETSc.Sys.Print(W.dim())
u = Function(W)
(Phi1, Phi2) = split(u)
(p1,p2) = TestFunctions(W)

factor_porosity = 1.0 if effective_porosity == "simple" else 0.0736806299  # 0.02^(2/3)
porosity_new = porosity * factor_porosity
epsilon_new = (1-gammaf) + (gammaf)*porosity_new
epsilon = (1-gammaf) + (gammaf)*porosity
a = gammaf 
DPhi = Phi1 - Phi2
sigma = (1-epsilon)**1.5 + 1e-22
kappa = epsilon_new**1.5 + 1e-22

bc = DirichletBC(W.sub(0), Constant(0), [1])
bcs =[bc]

i_n = exp(mu * DPhi) - exp(-mu * DPhi)
g2 = Constant(1)

a1 = inner(sigma*grad(Phi1), grad(p1))*dx + delta / mu * tau/(1+tau) * a * i_n * p1 * dx
a2 = tau * inner(kappa*grad(Phi2), grad(p2))*dx - delta / mu * tau / (1+tau) * a * i_n * p2 * dx - tau * g2 * p2 * ds(2)

F = a1 + a2

# solver options
snes_newtonls = {"snes_type": "newtonls",
                     "snes_linesearch_type": "l2",
                     #"snes_converged_reason": None,
                     "snes_rtol": 1e-4,
                     "snes_max_it": 50,
                }
cg_pc_triang_hypre = {"mat_type": "aij",
                     "ksp_rtol": 1e-4,
                     "ksp_type": "cg",
                     #"ksp_converged_reason": None,
                     "pc_type": "fieldsplit",
                     "pc_fieldsplit_type": "symmetric_multiplicative",
                     "fieldsplit_1_ksp_type": "preonly",
                     "fieldsplit_1_pc_type": "hypre",
                     "fieldsplit_0_ksp_type": "preonly",
                     "fieldsplit_0_pc_type": "hypre",
                     }
solver_parameters = snes_newtonls
if args.dim == 2:
    solver_parameters.update(lu)
else:
    solver_parameters.update(cg_pc_triang_hypre)
solve(F == 0, u, bcs = bcs, solver_parameters = solver_parameters)

# Optimization problem
if isforward == False:
    (Phi1sol, Phi2sol) = split(u)

    c = Control(gamma)

    gamma_viz_f = Function(GAMMA, name="gamma")
    controls_f = File(f"{output_dir}/control_iterations_f.pvd", target_continuity=target_continuity)
    def deriv_cb(j, dj, gamma):
        with stop_annotating():
            gamma_viz_f.assign(gammafControl.tape_value())
        controls_f.write(gamma_viz_f)

    J = assemble(2 *Phi2sol * ds(2))
    Jhat = ReducedFunctional(J, c, derivative_cb_post=deriv_cb)


    Vol = assemble(gammaf * dx(domain=mesh))
    VolControl = Control(Vol)
    Volhat = ReducedFunctional(Vol, c)
    Vollimit = assemble(Constant(1.0) * dx(domain=mesh)) * 1.0

    problem = MinimizationProblem(Jhat, bounds=(0.0, 1.0), constraints=[ReducedInequality(Volhat, Vollimit, VolControl)])

    parameters_mma = {
        "move": movlim,
        "maximum_iterations": max_iters,
        "m": 1,
        "IP": 0,
        "tol": 1e-7,
        "accepted_tol": 1e-5,
        # "gcmma": True,
        "norm": "L2",
    }
    solver = MMASolver(problem, parameters=parameters_mma)

    results = solver.solve()
    gamma_opt = results["control"]

    gamma.assign(gamma_opt)
    gammaf.assign(gammafControl.tape_value())
    solve(F == 0, u, bcs = bcs, solver_parameters = solver_parameters)

(Phi1vec, Phi2vec) = u.split()
(Phi1sol, Phi2sol) = split(u)

File(f"{output_dir}/gamma.pvd", target_continuity=target_continuity).write(gamma)
File(f"{output_dir}/gammaf.pvd", target_continuity=target_continuity).write(gammaf)
File(f"{output_dir}/solid_potential.pvd", target_continuity=target_continuity).write(Function(V, name="solid potential").assign(Phi1vec))
File(f"{output_dir}/liquid_potential.pvd", target_continuity=target_continuity).write(Function(V, name="Ionic potential").assign(Phi2vec))
File(f"{output_dir}/current_density.pvd", target_continuity=target_continuity).write(Function(V, name="Current density").interpolate(i_n))
File(f"{output_dir}/current_density_indomain.pvd", target_continuity=target_continuity).write(Function(V, name="Current density").interpolate(i_n*gammaf))
