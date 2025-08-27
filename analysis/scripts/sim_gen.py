import sys, os
import json

#this is a comment
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir) 

from modeling import *
from modelchecking import *
from analysis import *
from utils import NpEncoder


import os
if not os.path.exists(f"../data/sim/"):
    os.makedirs(f"../data/sim/")

n_repeats = 500

for type_ in ["R", "T", "V"]:
    model = Model(filter_depth, value_path, variant=type_)
    df = pd.DataFrame({
        "username": [f"sim_user{i}" for i in range(20)],
        "inv_temp": np.random.uniform(-4, 4, size=20),
        # randomly simulate a depth for each condition
        **{condition: np.random.choice(8, size=20) for condition in get_conditions(type_)}
    })

    datafile = f"../data/raw/data_{type_}.json"
    print("data file\t", datafile)
    with open(datafile) as f:
        data = json.load(f)

    sim_games_all = {}

    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        user = row.username
        params = {
            "model": model.name,
            "inv_temp": row["inv_temp"],
            "filter_params": {p: row[p] for p in get_conditions(type_)}
        }

        games = format_games(data[f"user{i}"]["data"])

        gamedata = []
        for _ in range(n_repeats): 
            prediction = model.predict(params, games)
            gamedata.extend([
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
                    for condition in range(5) for game in range(5)
            ])

        sim_games_all[user] = {
            "data": gamedata
        }

    df.to_pickle(f"../data/sim/simdf_{type_}.pkl")

    with open(f"../data/sim/simdata_{type_}.json", "w") as f:
        json.dump(sim_games_all, f, cls=NpEncoder)
