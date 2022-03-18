# init.py
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
    workspace = "/p/lscratchh/"+USER+"/porous_electrode_redox/"
elif plt == "Darwin":
    workspace = "./workspace/"

project = signac.init_project("electrode-sweep", workspace=workspace)

# New batch
parameters = {
    "Ny": [0], # 0 to use .msh file
    "filter_radius": np.array([0.01]),
    "porosity": np.array([0.5]),
    "effective_porosity": ['simple'],
    "tau": np.array([0.5, 0.1]),
    "delta": np.array([1., 25.]),
    "mu": np.array([0.1, 5.]),
    "maxiters": [300],
    "dim": [2],
}

for sp in grid(parameters):
    job = project.open_job(sp)
    job.init()

# New batch
parameters = {
    "Ny": [0], # 0 to use .msh file
    "filter_radius": np.array([0.01]),
    "porosity": np.array([0.5]),
    "effective_porosity": ['effective'],
    "tau": np.array([0.5, 0.005]),
    "delta": np.array([1., 25.]),
    "mu": np.array([0.1, 5.]),
    "maxiters": [300],
    "dim": [2],
}

for sp in grid(parameters):
    job = project.open_job(sp)
    job.init()

# New batch
parameters = {
    "Ny": [0], # 0 to use .msh file
    "filter_radius": np.array([0.01]),
    "porosity": np.array([0.5]),
    "effective_porosity": ['effective'],
    "tau": np.array([0.005]),
    "delta": np.array([25.]),
    "mu": np.array([5.]),
    "maxiters": [300],
    "dim": [3],
}

for sp in grid(parameters):
    job = project.open_job(sp)
    job.init()

