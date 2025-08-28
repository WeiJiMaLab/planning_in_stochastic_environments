#!/usr/bin/env python3
"""
Simulation Generation and Model Fitting Script

This script combines simulation data generation with immediate model fitting,
eliminating the need for intermediate JSON files.
"""
import sys
import os
import json
import tqdm
import warnings
import argparse
import numpy as np
import pandas as pd

# Add src directory to path
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir)

from modeling import *
from modelchecking import *
from utils import NpEncoder


def setup_directories():
    """Create necessary directories for output files."""
    dirs = ["../data/sim/", "../data/sim/fit/"]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def get_slurm_info():
    """Check if running on SLURM and return job information."""
    jobid = os.getenv('SLURM_ARRAY_TASK_ID')
    if jobid is not None:
        jobid = int(jobid)
        print(f"Running on SLURM, Job ID: {jobid}")
        return jobid
    else:
        print("Running sequentially")
        return None


def create_simulation_dataframe(type_, n_users=20):
    """Create a DataFrame with random parameters for simulation."""
    return pd.DataFrame({
        "username": [f"sim_user{i}" for i in range(n_users)],
        "inv_temp": np.random.uniform(-4, 4, size=n_users),
        **{condition: np.random.choice(8, size=n_users)
           for condition in get_conditions(type_)}
    })


def generate_simulation_data(model, params, games, n_repeats=100):
    """Generate simulation data for a given model and parameters."""
    gamedata = []

    for _ in range(n_repeats):
        prediction = model.predict(params, games)
        gamedata.extend([
            {
                "name": f"game_{params['username']}_c{condition}_g{game}",
                "p": get_conditions(params['type'])[condition],
                "boards": prediction.boards.isel(conditions=condition, games=game).values,
                "oracle": prediction.oracles.isel(conditions=condition, games=game, trials=0).values,
                "tuplepath": prediction.paths.isel(conditions=condition, games=game).values,
                "path": [f'{a},{b}' for a, b in prediction.paths.isel(conditions=condition, games=game).values],
                "actions": prediction.choose_left.isel(conditions=condition, games=game).astype(bool).values,
                "is_transition": prediction.is_transition.isel(conditions=condition, games=game).astype(bool).values,
                "trials": [{"rt": 0} for _ in range(7)],
            }
            for condition in range(5) for game in range(5)
        ])

    return gamedata


def fit_models_to_data(sim_games, type_, player, fit_params, filter_fns, value_fns):
    """Fit models to simulation data using multi-start optimization."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for filter_fn in filter_fns:
            for value_fn in value_fns:
                # Skip EV value function for V type
                if type_ == "V" and value_fn == value_EV:
                    continue

                fit_model = Model(filter_fn, value_fn, variant=type_)
                filedir = f"../data/sim/fit/{type_}_{fit_model.name}"
                filename = f"{filedir}/{player}.json"

                print(f"Saving to: {filename}")

                # Create directory if it doesn't exist
                try:
                    os.makedirs(filedir)
                except FileExistsError:
                    pass

                # Fit model using multi-start
                multistart = MultiStart(fit_model, sim_games, fit_params, use_grid=False, n=100)
                multistart.sweep()

                # Save best fit results
                with open(filename, "w") as f:
                    json.dump(multistart.best, f, cls=NpEncoder)

                # Save parameters
                with open(f"{filedir}/params.json", "w") as f:
                    json.dump(fit_params, f)


def process_single_type(type_, jobid=None):
    """Process simulation generation and fitting for a single type."""
    print(f"Processing type: {type_}")

    # Create model and simulation dataframe
    model = Model(filter_depth, value_path, variant=type_)
    df = create_simulation_dataframe(type_)

    # Load raw data
    datafile = f"../data/raw/data_{type_}.json"
    print(f"Data file: {datafile}")

    with open(datafile) as f:
        data = json.load(f)

    # Save dataframe for reference
    df.to_pickle(f"../data/sim/simdf_{type_}.pkl")

    # Determine which players to process
    if jobid is not None:
        players = [f"sim_user{jobid}"]
    else:
        players = [f"sim_user{i}" for i in range(20)]

    # Process each player
    for player in tqdm.tqdm(players, desc=f"Processing {type_} players"):
        user_idx = int(player.split('_')[2])
        row = df.iloc[user_idx]

        # Set up parameters for this player
        params = {
            "model": model.name,
            "inv_temp": row["inv_temp"],
            "filter_params": {p: row[p] for p in get_conditions(type_)},
            "username": player,
            "type": type_
        }

        # Get games for this user
        games = format_games(data[f"user{user_idx}"]["data"])

        # Generate simulation data
        gamedata = generate_simulation_data(model, params, games)

        # Format games for fitting
        sim_games = format_games(gamedata)

        # Fit models to the generated data
        fit_models_to_data(sim_games, type_, player, FIT_PARAMS, FILTER_FNS, VALUE_FNS)

    print(f"Completed processing type: {type_}")


def main():
    """Main function to run simulation generation and fitting."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate simulation data and fit models")
    parser.add_argument("--type", default="R", choices=["R", "T", "V"],
                       help="Type of data to process (R, T, or V)")
    args = parser.parse_args()

    # Setup
    setup_directories()
    jobid = get_slurm_info()

    # Process the specified type
    process_single_type(args.type, jobid)

    print("Simulation generation and fitting completed!")


# Global constants
FIT_PARAMS = {"inv_temp": 5}
FILTER_FNS = [filter_depth]
VALUE_FNS = [value_path]

if __name__ == "__main__":
    main()
