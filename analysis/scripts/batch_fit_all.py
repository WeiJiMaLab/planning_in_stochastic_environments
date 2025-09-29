import os
import sys
import warnings
import argparse

currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir) 
from utils import NpEncoder, format_games
from modeling import Model, MultiStart, filter_depth, filter_rank, filter_value, value_EV, value_path, value_max, value_sum, value_levelmean
import json

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="R")
parser.add_argument("--data_folder", default="raw")
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
data_folder = args.data_folder
datafile = f"../data/{data_folder}/data_{type_}.json"
with open(datafile, 'r') as f:
    data = json.load(f)



if jobid >= len(list(data.keys())): 
    print("Job ID is greater than the number of players. Exiting.")
    sys.exit(1)
player = list(data.keys())[jobid]
print("Player\t", player)

param_specs = [
    {
        "name": "main", 
        "params": {"inv_temp": 5, "lapse": 5},
        "conditional_filter_params": True,
        "multistart_n": 100,
    },
    {
        "name": "fixed_depth_variable_beta",
        "params": {"lapse": 5, **{f"condition_inv_temp_{i}": 5 for i in range(5)}},
        "conditional_filter_params": False,
        "multistart_n": 100,
    }, 
]

filter_fns = [filter_depth, filter_rank, filter_value]
value_fns = [value_EV, value_path, value_max, value_sum, value_levelmean]

# fit models with multi-start
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    games = format_games(data[player]["data"])
    for param_spec in param_specs:
        for filter_fn in filter_fns:
            for value_fn in value_fns:
                if type_ == "V" and value_fn == value_EV:
                    continue

                model = Model(filter_fn, value_fn, variant=type_, 
                            conditional_filter_params=param_spec["conditional_filter_params"])
                filedir = f"../fitted/{data_folder}/{param_spec['name']}/{type_}_{model.name}"
                print(f"Saving to \t{filedir}/{player}.json")
                
                multistart = MultiStart(model, games, param_spec["params"], 
                                      use_grid=False, n=param_spec["multistart_n"])
                multistart.sweep()

                os.makedirs(filedir, exist_ok=True)
                with open(f"{filedir}/{player}.json", "w") as f:
                    json.dump(multistart.best, f, cls=NpEncoder)

                with open(f"{filedir}/params.json", "w") as f: 
                    json.dump(param_spec["params"], f)
