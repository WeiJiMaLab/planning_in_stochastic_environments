import sys, os
import json
import numpy as np
import tqdm
import argparse
# Set up path and imports
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir)
from modeling import Model, filter_depth, value_path
from utils import NpEncoder, format_games, get_conditions

def simulate_model(user, model_family="filteradapt"):
    """Run simulation for a single user with specified model type"""
    sim_user = f"sim_{user}"
    model = Model(filter_depth, value_path, variant=type_)
    
    # Set parameters based on model type
    if model_family == "filteradapt":
        default_params = {"model": model.name, "filter_params": {p: np.random.choice(8) for p in get_conditions(type_)}}
        flexible_params = {"lapse": np.random.uniform(0, 1), "inv_temp": np.random.uniform(-4, 4)}
    else:  # policycomp
        default_params = {"model": model.name, "filter_params": {"global_depth": np.random.choice(8)}}
        flexible_params = {"lapse": np.random.uniform(0, 1)}
        flexible_params.update({f"condition_inv_temp_{i}": np.random.uniform(-4, 4) for i in range(5)})
    
    sim_params = {**default_params, **flexible_params}
    games = format_games(data[user]["data"])
    simulated_data = []

    # Run simulations
    for _ in tqdm.tqdm(range(n_repeats), desc=f"simulating {user}"):
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
            for condition in range(5) for game in range(n_games)
        ])

    # Save results
    filedir = f"../data/simulated_{model_family}/{type_}/"
    os.makedirs(filedir, exist_ok=True)
    with open(f"{filedir}/{sim_user}_data.json", "w") as f:
        json.dump({"data": simulated_data}, f, cls=NpEncoder)

    # Save the original parameters
    with open(f"{filedir}/{sim_user}_params.json", "w") as f:
        json.dump(sim_params, f)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="R", help="Data Variant: R, T, or V")
    args = parser.parse_args()

    # Get job ID if running on SLURM
    jobid = int(os.getenv('SLURM_ARRAY_TASK_ID')) if os.getenv('SLURM_ARRAY_TASK_ID') else None

    # Simulation parameters
    type_ = args.type
    n_repeats, n_participants, n_games = 20, 100, 5

    # Load data
    with open(f"../data/raw/data_{type_}.json") as f:
        data = json.load(f)

    # Get users to simulate
    users = [f"user{i}" for i in range(n_participants)]
    if jobid is not None:
        users = [users[jobid]]

    # Run simulations for both model families
    for user in users:
        simulate_model(user, "filteradapt")
        simulate_model(user, "policycomp")
