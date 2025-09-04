import os
import sys
import inspect
import tqdm
import warnings
import argparse
import datetime

currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir) 
from utils import NpEncoder, format_games
from modeling import Model, MultiStart, filter_depth, filter_rank, filter_value, value_EV, value_path, value_max, value_sum
import json

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="R")
args = parser.parse_args()
type_ = args.type

# detect if running in SLURM environment
try:
    jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    print("Job ID\t", jobid)
except TypeError:
    warnings.warn("Not running in SLURM environment. Setting job ID to 1.")
    jobid = 1


# load data and get player
datafile = "../data/raw/data_%s.json"%(type_)
with open(datafile, 'r') as f:
    data = json.load(f)
player = list(data.keys())[jobid]

param_specs = [
    {
        # a single filter_param with conditional inv_temp
        "name": "conditional_inv_temp",
        "params": {"condition_inv_temp_0": 5, "condition_inv_temp_1": 5, "condition_inv_temp_2": 5, "condition_inv_temp_3": 5, "condition_inv_temp_4": 5},
        "conditional_filter_params": False,
        # here we want to use a larger number of multistart runs ...
        "multistart_n": 500,
    }, 
    {
        # a single filter_param with global inv_temp
        "name": "original_fit", 
        "params": { "inv_temp": 5 },
        "conditional_filter_params": True,
        "multistart_n": 100,
    },
    {
        # a single filter_param with conditional lapse
        "name": "conditional_lapse",
        "params": {"inv_temp": 5, "condition_lapse_0": 5, "condition_lapse_1": 5, "condition_lapse_2": 5, "condition_lapse_3": 5, "condition_lapse_4": 5},
        "conditional_filter_params": True,
        # here we want to use a larger number of multistart runs ...
        "multistart_n": 500,
    },
    {
        # a single filter_param with global lapse
        "name": "global_lapse", 
        "params": {"inv_temp": 5, "lapse": 5},
        "conditional_filter_params": True,
        "multistart_n": 100,
    },
]

# params.update({p:5 for p in get_conditions(type_)})

filter_fns = [filter_depth, filter_rank, filter_value]
value_fns = [value_EV, value_path, value_max, value_sum]


# fit models with multi-start
with warnings.catch_warnings():
    # filter warnings
    warnings.simplefilter("ignore")
    
    games = format_games(data[player]["data"])
    for param_spec in param_specs:
        for filter_fn in filter_fns:
            for value_fn in value_fns:

                name = param_spec["name"]
                params = param_spec["params"]
                conditional_filter_params = param_spec["conditional_filter_params"]
                multistart_n = param_spec["multistart_n"]

                if not (type_ == "V" and value_fn == value_EV):
                    model = Model(filter_fn, value_fn, variant = type_, conditional_filter_params = conditional_filter_params)
                    filedir = f"../data/{name}/{type_}_{model.name}"
                    print(f"Saving to \t{filedir}/{player}.json")
                    
                    # we are *not* using the grid search here, but we can if we want ...
                    multistart = MultiStart(model, games, params, use_grid = False, n = multistart_n)
                    multistart.sweep()

                    os.makedirs(filedir, exist_ok=True)
                    with open(f"{filedir}/{player}.json", "w") as f:
                        json.dump(multistart.best, f, cls=NpEncoder)

                    # save parameters
                    with open(f"{filedir}/params.json", "w") as f: 
                        json.dump(params, f)
