from firedrake.petsc import PETSc


def print(x):
    return PETSc.Sys.Print(x)


def solver_options(dim=2):
    direct_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    if dim == 2:
        supercap_solver_params = direct_parameters
        filter_solver_params = direct_parameters
    elif dim == 3:
        amg_parameters = {
            "mat_type": "aij",
            "ksp_type": "cg",
            "ksp_atol": 1e-3,
            "ksp_rtol": 1e-3,
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "pc_hypre_boomeramg_max_iter": 200,
            "pc_hypre_boomeramg_coarsen_type": "HMIS",
            "pc_hypre_boomeramg_agg_nl": 1,
            "pc_hypre_boomeramg_strong_threshold": 0.7,
            "pc_hypre_boomeramg_interp_type": "ext+i",
            "pc_hypre_boomeramg_P_max": 4,
            "pc_hypre_boomeramg_relax_type_all": "sequential-Gauss-Seidel",
            "pc_hypre_boomeramg_grid_sweeps_all": 1,
            "pc_hypre_boomeramg_max_levels": 15,
            #"ksp_converged_reason": None,
        }
        iterative_parameters = {
            "mat_type": "aij",
            "ksp_rtol": 1e-4,
            "ksp_type": "cg",
            "pc_type": "fieldsplit",
            "ksp_max_it": 200,
            #"ksp_converged_reason": None,
            "pc_fieldsplit_type": "symmetric_multiplicative",
            "fieldsplit_1_ksp_type": "preonly",
            "fieldsplit_1_pc_type": "hypre",
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "hypre",
        }

        filter_solver_params = amg_parameters
        supercap_solver_params = iterative_parameters

    return filter_solver_params, supercap_solver_params
