import signac
import numpy as np
import itertools

import platform


def grid(gridspec):
    for values in itertools.product(*gridspec.values()):
        yield dict(zip(gridspec.keys(), values))


plt = platform.system()

workspace = "./"
if plt == "Linux":
    import os
    USER = os.environ['USER']
    workspace = "/p/lscratchh/"+USER+"/supercapacitor/non-dim-two-potential/"
elif plt == "Darwin":
    workspace = "./workspace/"

project = signac.init_project("electrode-sweep", workspace=workspace)

parameters = {
    "initial_design": ["uniform"],
    "tau": np.array([0.005, 0.05]),
    "xi": np.array([0.1, 0.01]),
    "constraint_value": np.array([0.2]),
    "effective_porosity": ['simple', 'effective'],
    "n_steps" : np.array([200]),
    "filter_radius": np.array([1e-4]),
    "initial_gamma_value": np.array([0.5, 0.6]),
    "dim": np.array([2]),
    "continuation": np.array([0]),
}

for sp in grid(parameters):
    job = project.open_job(sp)
    job.init()


parameters = {
    "initial_design": ["uniform"],
    "tau": np.array([0.005, 0.05]),
    "xi": np.array([0.1, 0.01]),
    "constraint_value": np.array([0.5]),
    "effective_porosity": ['simple', 'effective'],
    "n_steps" : np.array([200]),
    "filter_radius": np.array([1e-4]),
    "initial_gamma_value": np.array([0.5, 0.6]),
    "dim": np.array([2]),
    "continuation": np.array([1]),
}

for sp in grid(parameters):
    job = project.open_job(sp)
    job.init()

parameters = {
    "initial_design": ["uniform"],
    "tau": np.array([0.05]),
    "xi": np.array([0.01]),
    "constraint_value": np.array([0.5]),
    "effective_porosity": ['simple', 'effective'],
    "n_steps" : np.array([20]),
    "filter_radius": np.array([1e-4]),
    "initial_gamma_value": np.array([0.6]),
    "dim": np.array([3])
}

for sp in grid(parameters):
    job = project.open_job(sp)
    job.init()

if __name__ == "__main__":
    pass
