import os
import sys
import warnings
import argparse
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir) 
from utils import NpEncoder,  get_data, get_jobid
from modeling import Model, MultiStart, get_effort_filter_value_options
import json

# process a single player's data for all model combinations
def process_player(player, data, type_): 
    effort_versions, filter_fns, value_fns = get_effort_filter_value_options(type_)
    games = data[player]
    assert len(games) == 150, "Number of games is not 150."

    for effort_version in effort_versions: 
        if effort_version == "filter_adapt":
            param_spec = {"params": {"lapse": True, "inv_temp": True}}
        elif effort_version == "policy_compress":
            param_spec = {"params": {"lapse": True, **{f"condition_inv_temp_{i}": True for i in range(5)}}}
        
        for filter_fn in filter_fns:
            for value_fn in value_fns: 
                model = Model(effort_version, filter_fn, value_fn, type_)

                print(f"Fitting {model.name} for player {player}")

                multistart = MultiStart(model, games, param_spec["params"], use_grid=False, n=1)
                multistart.sweep()

                filedir = f"../fitted/{args.data_folder}/{type_}_{model.name}"
                os.makedirs(filedir, exist_ok=True)
                with open(f"{filedir}/{player}.json", "w") as f:
                    json.dump(multistart.best, f, cls=NpEncoder)

                with open(f"{filedir}/params.json", "w") as f: 
                    json.dump(param_spec["params"], f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="R")
    parser.add_argument("--data_folder", default="raw")
    args = parser.parse_args()
    type_ = args.type
    jobid = get_jobid()

    data = get_data(type_, data_folder=args.data_folder)

    if jobid > len(list(data.keys())):
        print(f"Job ID {jobid} is out of range. There are {len(list(data.keys()))} players.")
        sys.exit(1)
    else: 
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            player = list(data.keys())[jobid]
            print(f"Processing job ID {jobid} for player {player} for type {type_}")
            process_player(player, data, type_)