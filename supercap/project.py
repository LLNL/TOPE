import flow
from flow import FlowProject
from init import parameters
import platform
import re
from matplotlib import pyplot as plt
from signac import get_project

project = get_project()
xi_range = [0.01, 0.05, 0.1, 0.5, 1.0]


@FlowProject.label
def check_iterations(job):
    filename = f"geometry_{job.sp.initial_design}_1.vtu"
    return job.isfile(filename)


@FlowProject.label
def check_h5_design(job):
    return job.isfile("final_design_uniform")


@FlowProject.label
def check_output(job):
    return job.isfile("output.txt")


@FlowProject.operation
@flow.cmd
@FlowProject.post(check_output)
@FlowProject.post(check_iterations)
@FlowProject.post(check_h5_design)
def launch_opti(job):
    program = "supercap.py"
    param_flags = [
        "--{0} {1} ".format(param, job.sp.get(param)) for param in parameters
    ]
    output = job.ws + "/output.txt"
    plat = platform.system()
    proc = 60 if job.sp.get("dim", None) == 3 else 1
    if plat == "Linux":
        simulation = "srun -n {4} --output={0} python3 {3} {1} --output_dir {2}".format(
            output, "".join(param_flags), job.ws, program, proc
        )
    else:
        simulation = "python3 {3} {0} --output_dir {1}/ > {2}".format(
            "".join(param_flags), job.ws, output, program
        )
    return simulation


@FlowProject.label
def check_figures(job):
    return all(
        job.isfile(file)
        for file in ["convergence_plots_0.svg", "convergence_plots_1.svg"]
    )


@FlowProject.operation
@FlowProject.pre(check_iterations)
@FlowProject.post(check_figures)
def plot_history(job):
    from read_history import parse_history
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    (
        iter_history_current,
        obj_iteration_history,
        constr_iteration_history,
    ) = parse_history(f"{job.ws}/output.txt")
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    ax1.plot(iter_history_current[:300], obj_iteration_history[:300])
    ax1.set_xlabel("Number of iterations", fontsize=18)
    ax1.set_ylabel("Cost function", fontsize=18)
    ax1.set_xscale('log')

    ax2.plot(iter_history_current[:300], constr_iteration_history[:300])
    ax2.set_xlabel("Number of iterations", fontsize=18)
    ax2.set_ylabel("Constraint function", fontsize=18)

    for i, figura in enumerate([fig1, fig2]):
        plt.figure(figura.number)
        plt.savefig(
            f"{job.ws}/convergence_plots_{i}.svg",
            dpi=1600,
            orientation="portrait",
            papertype=None,
            format=None,
            transparent=True,
            bbox_inches="tight",
        )


@FlowProject.label
def check_final_cost(job):
    return job.isfile("results.txt")


@FlowProject.operation
@FlowProject.pre(check_iterations)
@FlowProject.post(check_final_cost)
def write_optimal_information(job):

    # Read value of the constraint
    eng_number = "-?\d+\.?\d*(?:e-?\d+)?"
    with open("{0}/output.txt".format(job.ws), "r") as sim_output:
        results = re.findall(
            "Value: ({0}), Constraint ({0})".format(eng_number), sim_output.read()
        )
    energy_stored = 1.0 / float(results[-1][0])
    energy_required = 1.0 / float(results[-1][1])

    # Read value of the cost function
    with open("{0}/output.txt".format(job.ws), "r") as sim_output:
        results = re.findall(
            "obj: ({0}) g\[0\]: ({0})".format(eng_number), sim_output.read()
        )
    final_result = results[-1]

    # Read value of initial resistive energy so we can get the real energy loss
    with open("{0}/output.txt".format(job.ws), "r") as sim_output:
        results = re.findall(
            "Energy resistive: ({0})".format(eng_number), sim_output.read()
        )
    initial_energy_loss = results[-1]
    final_energy_loss = float(final_result[0]) * float(initial_energy_loss)

    job.doc["final_energy_loss"] = final_energy_loss
    job.doc["energy_stored"] = energy_stored

    # Write both
    with open("{0}/results.txt".format(job.ws), "w") as results:
        results.write(
            f"Cost function: {final_energy_loss}\nEnergy stored: {energy_stored}\nEnergy required: {energy_required}"
        )


@FlowProject.label
def check_design(job):
    return job.isfile(f"{screenshot_name(job)}.png")


def screenshot_name(job):
    return f"design_eff_po_{job.sp['effective_porosity']}_tau_{job.sp['tau']}_xi_{job.sp['xi']}_cv_{job.sp['constraint_value']}_gamma_{job.sp['initial_gamma_value']}"


@FlowProject.operation
@flow.cmd
@FlowProject.pre(check_iterations)
@FlowProject.post(check_design)
def post_process_design(job):
    parameters = "".join([key + " " + f"{job.sp[key]}" + "\n" for key in job.sp.keys()])

    plat = platform.system()
    return (
        "srun pvpython screenshot_design.py \
                --parameters '{0}' \
                --filename {1} \
                --initial_design {3} \
                --results_dir {2} && \
                convert {2}/{1}.png -trim {2}/{1}.png".format(
            parameters, screenshot_name(job), job.ws, job.sp.initial_design
        )
        if plat == "Linux"
        else "/Applications/ParaView-5.8.0.app/Contents/bin/pvpython screenshot_design.py \
                --parameters '{0}' \
                --filename {1} \
                --initial_design {3} \
                --results_dir {2} && \
                convert {2}/{1}.png -trim {2}/{1}.png".format(
            parameters, screenshot_name(job), job.ws, job.sp.initial_design
        )
    )


@FlowProject.label
def files_scan(job):
    files_scan = [f"xi_{xi}.npz" for xi in xi_range]
    return all(job.isfile(file) for file in files_scan)


@FlowProject.operation
@flow.cmd
@FlowProject.pre(check_h5_design)
@FlowProject.post(files_scan)
def run_simulation_design(job):
    program = "supercap.py"
    simulation = ""
    parameters_sim = dict(job.sp.items())
    parameters_sim["initial_design"] = "not_uniform"

    for xi in xi_range:
        parameters_sim["xi"] = xi
        param_flags = [
            "--{0} {1} ".format(key, param) for key, param in parameters_sim.items()
        ]
        print(param_flags)
        output = job.ws + "/output_simulation.txt"
        plat = platform.system()
        proc = 60 if job.sp.get("dim", None) == 3 else 1
        if plat == "Linux":
            simulation += "srun -n {4} --output={0} python3 {3} {1} --output_dir {2} --forward\n".format(
                output, "".join(param_flags), job.ws, program, proc
            )
        else:
            simulation += "python3 {3} {0} --output_dir {1} --forward / > {2}\n".format(
                "".join(param_flags), job.ws, output, program
            )
    return simulation


if __name__ == "__main__":
    FlowProject().main()
