import sys, os
import json
import numpy as np
import tqdm
import argparse
# Set up path and imports
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir)
from modeling import Model, filter_depth, value_path, value_levelmean, value_sum, value_max
from utils import (
    NpEncoder, 
    format_games, 
    get_stochasticity_levels, 
    get_jobid, 
    get_data
)


def simulate_model(user, data, effort_version, filter_fn, value_fn, type_, n_repeats, n_games):
    """Run simulation for a single user with specified model type"""
    sim_user = f"sim_{user}"
    model = Model(effort_version, filter_fn, value_fn, type_)
    if effort_version == "filter_adapt":
        params = {
            "model": model.name,
            "filter_params": {p: np.random.choice(8) for p in get_stochasticity_levels(type_)},
            "lapse": np.random.uniform(0, 1),
            "inv_temp": np.random.uniform(-5, 5)
        }
    elif effort_version == "policy_compress":
        params = {
            "model": model.name,
            "filter_params": {"global": np.random.choice(8)},
            "lapse": np.random.uniform(0, 1),
            "condition_inv_temp_0": np.random.uniform(-5, 5),
            "condition_inv_temp_1": np.random.uniform(-5, 5),
            "condition_inv_temp_2": np.random.uniform(-5, 5),
            "condition_inv_temp_3": np.random.uniform(-5, 5),
            "condition_inv_temp_4": np.random.uniform(-5, 5)
        }
    
    games = data[user]
    simulated_data = []

    # Run simulations
    for _ in tqdm.tqdm(range(n_repeats), desc=f"simulating {user}"):
        prediction = model.predict(params, games)
        simulated_data.extend([
            {
                "name": f"game_{user}_c{condition}_g{game}",
                "p": get_stochasticity_levels(type_)[condition],
                "boards": prediction.boards.isel(conditions=condition, games=game).values,
                "oracle": prediction.oracles.isel(conditions=condition, games=game, trials=0).values,
                "tuplepath": prediction.paths.isel(conditions=condition, games=game).values,
                "path": [f'{a},{b}' for a, b in prediction.paths.isel(conditions=condition, games=game).values],
                "actions": prediction.choose_left.isel(conditions=condition, games=game).astype(bool).values,
                "is_transition": prediction.is_transition.isel(conditions=condition, games=game).astype(bool).values,
                "trials": [{"rt": 0} for _ in range(7)],
            }
            for condition in range(5) for game in range(n_games)
        ])

    # Save results
    filedir = f"../data/simulated_{effort_version}.{filter_fn.__name__}.{value_fn.__name__}/{type_}"
    os.makedirs(filedir, exist_ok=True)
    with open(f"{filedir}/{sim_user}_data.json", "w") as f:
        json.dump({"data": simulated_data}, f, cls=NpEncoder)

    # Save the original parameters
    with open(f"{filedir}/{sim_user}_params.json", "w") as f:
        json.dump(params, f)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="R", help="Data Variant: R, T, or V")
    args = parser.parse_args()

    # Get job ID if running on SLURM
    jobid = get_jobid()

    # Simulation parameters
    type_ = args.type
    n_repeats, n_participants, n_games = 20, 100, 5

    data = get_data(type_, data_folder="raw")

    # Get users to simulate
    users = [f"user{i}" for i in range(n_participants)]
    users = [users[jobid]]

    # Run simulations for both model families
    for user in users:
        # simulate_model(user, data, "filter_adapt", filter_depth, value_path, type_, n_repeats, n_games)
        simulate_model(user, data, "policy_compress", filter_depth, value_path, type_, n_repeats, n_games)
        simulate_model(user, data, "policy_compress", filter_depth, value_levelmean, type_, n_repeats, n_games)
        simulate_model(user, data, "policy_compress", filter_depth, value_sum, type_, n_repeats, n_games)
        simulate_model(user, data, "policy_compress", filter_depth, value_max, type_, n_repeats, n_games)



if __name__ == "__main__":
    main()
