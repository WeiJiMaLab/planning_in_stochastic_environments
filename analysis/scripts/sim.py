import sys, os
import json

#this is a comment
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir) 
from modeling import Model, MultiStart, filter_depth, value_path
from modelchecking import *
from utils import NpEncoder, format_games, get_conditions
import numpy as np
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default="R", help="Type of data to use")
args = parser.parse_args()

def check_jobid():
    if os.getenv('SLURM_ARRAY_TASK_ID') is not None:
        jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        print("Job ID\t", jobid)
        return jobid
    else:
        print("No SLURM job ID found")
        return None

jobid = check_jobid()
type_ = args.type
n_repeats = 20
n_participants = 100

datafile = f"../data/raw/data_{type_}.json"
print("data file\t", datafile)
with open(datafile) as f:
    data = json.load(f)

users = [f"user{i}" for i in range(n_participants)]
if jobid is not None:
    users = [users[jobid]]

for user in users:
    sim_user = f"sim_{user}"
    model = Model(filter_depth, value_path, variant=type_)

    default_params = {"model": model.name, "filter_params": {p: np.random.choice(8) for p in get_conditions(type_)}}
    flexible_params = {"lapse": np.random.uniform(0, 1), "inv_temp": np.random.uniform(-4, 4)}
    sim_params = {**default_params, **flexible_params}

    games = format_games(data[user]["data"])
    simulated_data = []

    for _ in tqdm.tqdm(range(n_repeats), total=n_repeats, desc=f"simulating {user}"): 
        prediction = model.predict(sim_params, games)
        simulated_data.extend([
                {
                    "name": f"game_{user}_c{condition}_g{game}",
                    "p": get_conditions(type_)[condition],
                    "boards": prediction.boards.isel(conditions=condition, games=game).values,
                    "oracle": prediction.oracles.isel(conditions=condition, games=game, trials=0).values,
                    "tuplepath": prediction.paths.isel(conditions=condition, games=game).values,
                    "path": [f'{a},{b}' for a, b in prediction.paths.isel(conditions=condition, games=game).values],
                    "actions": prediction.choose_left.isel(conditions=condition, games=game).astype(bool).values,
                    "is_transition": prediction.is_transition.isel(conditions=condition, games=game).astype(bool).values,
                    "trials": [{"rt": 0} for _ in range(7)],
                }
                # do this for the first 5 games
                for condition in range(5) for game in range(5)
        ])

    out = {"data": simulated_data}

    filedir = f"../data/simulated_filteradapt/{type_}_{model.name}"
    os.makedirs(filedir, exist_ok=True)

    with open(f"{filedir}/{sim_user}_data.json", "w") as f:
        json.dump(out, f, cls=NpEncoder)

# this time, do conditional - inverse temp
for user in users:
    sim_user = f"sim_{user}"
    model = Model(filter_depth, value_path, variant=type_)

    default_params = {"model": model.name, "filter_params": {"global_depth": np.random.choice(8)}}
    flexible_params = {"lapse": np.random.uniform(0, 1)}

    for i in range(5):
        flexible_params[f"condition_inv_temp_{i}"] = np.random.uniform(-4, 4)

    sim_params = {**default_params, **flexible_params}

    games = format_games(data[user]["data"])
    simulated_data = []

    for _ in tqdm.tqdm(range(n_repeats), total=n_repeats, desc=f"simulating {user}"): 
        prediction = model.predict(sim_params, games)
        simulated_data.extend([
                {
                    "name": f"game_{user}_c{condition}_g{game}",
                    "p": get_conditions(type_)[condition],
                    "boards": prediction.boards.isel(conditions=condition, games=game).values,
                    "oracle": prediction.oracles.isel(conditions=condition, games=game, trials=0).values,
                    "tuplepath": prediction.paths.isel(conditions=condition, games=game).values,
                    "path": [f'{a},{b}' for a, b in prediction.paths.isel(conditions=condition, games=game).values],
                    "actions": prediction.choose_left.isel(conditions=condition, games=game).astype(bool).values,
                    "is_transition": prediction.is_transition.isel(conditions=condition, games=game).astype(bool).values,
                    "trials": [{"rt": 0} for _ in range(7)],
                }
                # do this for the first 5 games
                for condition in range(5) for game in range(5)
        ])
    
    out = {"data": simulated_data}

    filedir = f"../data/simulated_policycomp/{type_}_{model.name}"
    os.makedirs(filedir, exist_ok=True)

    with open(f"{filedir}/{sim_user}_data.json", "w") as f:
        json.dump(out, f, cls=NpEncoder)



    
    
    





