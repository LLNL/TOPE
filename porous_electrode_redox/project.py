import flow
from flow import FlowProject
from init import parameters
import platform
import re
import numpy as np
from matplotlib import pyplot as plt
from signac import get_project

project = get_project()
tau_range = [0.005, 0.05, 0.1, 0.5]

@FlowProject.label
def check_iterations(job):
    filename = f"control_iterations_f_1.vtu"
    return job.isfile(filename)

@FlowProject.label
def check_output(job):
    return job.isfile("output.txt")


@FlowProject.operation
@flow.cmd
@FlowProject.post(check_output)
@FlowProject.post(check_iterations)
def launch_opti(job):
    program = "porous_electrode_redox_opt.py"
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
        for file in ["convergence_plots.svg"]
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
        obj_iteration_history 
    ) = parse_history(f"{job.ws}/output.txt")
    fig1, ax1 = plt.subplots()

    ax1.plot(iter_history_current[:300], obj_iteration_history[:300])
    ax1.set_xlabel("Number of iterations", fontsize=18)
    ax1.set_ylabel("Cost function", fontsize=18)
    #ax1.set_xscale('log')

    plt.figure(fig1.number)
    plt.savefig(
        f"{job.ws}/convergence_plots.svg",
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

    eng_number = "-?\d+\.?\d*(?:e-?\d+)?"
    # Read value of the cost function
    with open("{0}/output.txt".format(job.ws), "r") as sim_output:
        results = re.findall(
            "obj: ({0}) g\[0\]: ({0})".format(eng_number), sim_output.read()
        )
    final_result = results[-1]

    job.doc["final_result"] = final_result

    # Write both
    with open("{0}/results.txt".format(job.ws), "w") as results:
        results.write(
            f"Cost function: {final_result}"
        )

@FlowProject.label
def check_design(job):
    return job.isfile(f"{screenshot_name(job)}.png")

def screenshot_name(job):
    return f"design_po_{job.sp['effective_porosity']}_tau_{job.sp['tau']}_delta_{job.sp['delta']}_mu_{job.sp['mu']}"

@FlowProject.operation
@flow.cmd
@FlowProject.post(check_iterations)
@FlowProject.post(check_design)
def post_process_design(job):
    parameters = "".join([key + " " + f"{job.sp[key]}" + "\n" for key in job.sp.keys()])

    plat = platform.system()
    if plat == "Linux":
        post_process = "srun pvpython screenshot_design.py \
                --parameters '{0}' \
                --filename {1} \
                --results_dir {2} && \
                convert {2}/{1}.png -trim {2}/{1}.png".format(
            parameters, screenshot_name(job), job.ws        
            )
    else:
        post_process = "/Applications/ParaView-5.8.0.app/Contents/bin/pvpython screenshot_design.py \
                --parameters '{0}' \
                --filename {1} \
                --results_dir {2} && \
                convert {2}/{1}.png -trim {2}/{1}.png".format(
            parameters, screenshot_name(job), job.ws        
            )
    return post_process


@FlowProject.label
def files_scan(job):
    files_scan = [f"tau_{tau}.npz" for tau in tau_range]
    return all([job.isfile(file) for file in files_scan])

if __name__ == "__main__":
    FlowProject().main()
