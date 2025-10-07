import numpy as np
import pandas as pd
import requests
import json
import tqdm
import copy
import os

type, prob = ("R", "p_unreliable")
# type, prob = ("T", "p_transition")
# type, prob = ("V", "p_volatile")

with open(f"raw_{type}.json", "r") as f:
    rawdata = json.load(f)

# Filter out:
# 1. Lab test / pilot participants
# 2. Participants who didn't complete (no earnings recorded) 
# 3. Participant with rendering errors (ID: 5ebb96b6cd673a0008ace9ce)
# 4. Participants who were part of prolific pilot studies (ID: 5ae3223938df950001455e66)
valid_users = [
    user_id for user_id in rawdata.keys()
    if "pilot" not in user_id 
    and "earnings" in rawdata[user_id]
    and user_id != "5ebb96b6cd673a0008ace9ce"
    and user_id != "5ae3223938df950001455e66"
]

print(f"Total participants in dataset: {len(rawdata.keys())}")
print(f"Total valid participants: {len(valid_users)}")

anon_data = {}
survey_data = []

for i, user in tqdm.tqdm(enumerate(valid_users), total = len(valid_users)): 
    anon_username = "user" + str(i)
    gamedata = []
    survey = {}
    for frame in rawdata[user]["data"]:  
        if "trial_type" in frame.keys() and frame["trial_type"] == "survey":
            survey.update(frame["response"])

        if "type" in frame.keys() and frame["type"] == "game_end":
            game = copy.deepcopy(frame["game"])
            game["trials"] = frame["trials"]
            game["p"] = game[prob]
            gamedata.append(game)
            
    del survey["signature"]
    survey_data.append(survey)
    anon_data["user" + str(i)] = gamedata
    assert len([g for g in gamedata if 'practice' not in g["name"]]) == 150

with open(f'data_{type}.json', "r") as f:
    compare_data = json.load(f)

for key in compare_data.keys(): 
    compare_games = compare_data[key]["data"]
    games = anon_data[key]

    for game, compare_game in zip(games, compare_games): 
        for k in game.keys():
            assert game[k] == compare_game[k]