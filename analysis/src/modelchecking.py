import xarray as xr
import numpy as np
import pandas as pd
from utils import format_games
from helper import make_pov_array, preprocess_data
from modeling import value_path, Model
from collections import defaultdict
import tqdm as tqdm
import os
from prodict import Prodict
from utils import get_conditions
import json
import copy

def load_fit(variant, filter_fn, value_fn, folder = "fit", verbose = False): 
    '''
    Loads and returns the fit data for a given model variant.
    Parameters:
    variant (str): The variant of the model to load.
    filter_fn (function): The filter function used by the model.
    value_fn (function): The value function used by the model.
    Returns:
    dict: A dictionary where keys are player identifiers and values are Prodict objects containing fit data.
    Raises:
    AssertionError: If the number of loaded fits is not equal to 100.
    '''
    model = Model(filter_fn, value_fn, variant = variant)
    filedir = f"../data/{folder}/{variant}_{model.name}"
    
    fit = defaultdict(lambda: {})
    files = [f for f in sorted(os.listdir(filedir)) if "user" in f]
    for file in tqdm.tqdm(files, disable = not verbose): 
        player = file.split(".")[0]
        with open(f"{filedir}/{file}", "r") as f: 
            player_fit = Prodict(json.load(f))

            if type(player_fit.filter_params) == dict:
                player_fit["filter_params"] = {float(i): player_fit.filter_params[i] for i in player_fit.filter_params.keys()}
            else: 
                player_fit["filter_params"] = {"global_depth": player_fit.filter_params}
    
        fit[player] = player_fit.copy()
    
    assert len(fit) == 100, f"Expected 100 participants, got {len(fit)}"
    return fit

def fit_to_dataframe(fit):
    fit = copy.deepcopy(fit)
    df = []
    
    for player in fit.keys():
        
        fit[player].update(fit[player]["filter_params"])
        del fit[player]["filter_params"]
        del fit[player]["model"]
        fit[player]["player"] = player
        df.append(pd.Series(fit[player]))
        
    df = pd.DataFrame(df)
    df = df.sort_values(by = "player")
    return df

#functions for model checking
def trialwise_rewards(game_data, baseline = "none"): 
    oracles_pov = make_pov_array(game_data.oracles, game_data.paths)
    boards_pov = make_pov_array(game_data.boards, game_data.paths)

    # if there was a transition, return 1 - choose_left, else return choose_left
    move_left = xr.where(game_data.is_transition, 1 - game_data.choose_left, game_data.choose_left)
    rewards = xr.where(move_left, oracles_pov.isel(rows = 1, cols = 0), oracles_pov.isel(rows = 1, cols = 1))
    
    if baseline == "none":
        return rewards
    
    elif baseline == "random":
        random_baseline = 0.5 * (oracles_pov.isel(rows = 1, cols = 0) + oracles_pov.isel(rows = 1, cols = 1))
        return rewards - random_baseline
    
    elif baseline == "greedy":
        # note that this greedy choice preferentially chooses the left when the values are tied -- not to be used
        # for determining whether a given choice is greedy
        greedy_choose_left = (boards_pov.isel(rows = 1, cols = 0) >= oracles_pov.isel(rows = 1, cols = 1)).astype(int)
        greedy_move_left = xr.where(game_data.is_transition, 1 - greedy_choose_left, greedy_choose_left)
        greedy_baseline = xr.where(greedy_move_left, oracles_pov.isel(rows = 1, cols = 0), oracles_pov.isel(rows = 1, cols = 1))
        return rewards - greedy_baseline
    
    else: 
        assert False, "InputError: Baseline must be one of: none, random, greedy"

def trialwise_vdiff(game_data):
    boards_pov = make_pov_array(game_data.boards, game_data.paths)
    return value_path(boards_pov)

def trialwise_greedydiff(game_data): 
    boards_pov = make_pov_array(game_data.boards, game_data.paths)
    
    # left minus right choice
    greedy_diff = boards_pov.isel(rows = 1, cols = 0) - boards_pov.isel(rows = 1, cols = 1)
    return greedy_diff

def trialwise_choosegreedy(game_data): 
    boards_pov = make_pov_array(game_data.boards, game_data.paths)

    return (game_data.choose_left & (boards_pov.isel(rows = 1, cols = 0) >= boards_pov.isel(rows = 1, cols = 1))| \
            ~game_data.choose_left & (boards_pov.isel(rows = 1, cols = 0) <= boards_pov.isel(rows = 1, cols = 1)))

def trialwise_chooseleft(game_data): 
    return game_data.choose_left

def trialwise_trialnumber(game_data): 
    return xr.ones_like(game_data.choose_left) * [1, 2, 3, 4, 5, 6, 7]

def trialwise_ignore(game_data): 
    return xr.zeros_like(game_data.choose_left)

def trialwise_reactiontime(game_data): 
    return np.log(game_data.reaction_time)

def empirical(data, x_fun, y_fun): 
    x_ = []
    y_ = []

    for player in tqdm.tqdm(data.keys(), position=0, leave=True):
        games = format_games(data[player]["data"])
        game_data = preprocess_data(games)
        x_.append(x_fun(game_data))
        y_.append(y_fun(game_data))

    x = xr.concat(x_, dim = "participants")
    y = xr.concat(y_, dim = "participants")

    return x, y
     

def simulate_model(data, model, fitted_params, x_fun, y_fun, iters = 1):
    x_ = []
    y_ = []
    
    for player in tqdm.tqdm(data.keys(), position=0, leave=True): 
        x__ = []
        y__ = []

        for _ in range(iters):
            games = format_games(data[player]["data"])
            model_data = model.predict(fitted_params[player], games)
            x__.append(x_fun(model_data))
            y__.append(y_fun(model_data))
        
        x_.append(xr.concat(x__, dim = "trials"))
        y_.append(xr.concat(y__, dim = "trials"))

    x = xr.concat(x_, dim = "participants")
    y = xr.concat(y_, dim = "participants")
    return x, y

## helpers for model checking
def segment_by_condition(y, n_bins = None): 
    condition_y = {}
    
    for condition in y.conditions: 
        y_ = y.sel(conditions = condition).values.flatten()
        condition_y[condition.item()] = y_
        
    return condition_y


    
def summary_statistics(x, y, n_bins=None):
    """
    Calculate summary statistics for given data.

    Parameters:
    x (xarray.DataArray): The x data array with conditions and participants dimensions.
    y (xarray.DataArray): The y data array with conditions and participants dimensions.
    n_bins (int, optional): Number of bins to use for quantile binning. If None, no binning is applied.

    Returns:
    tuple: A tuple containing three dictionaries:
        - condition_x (dict): Dictionary with conditions as keys and aggregated x values as values.
        - condition_y (dict): Dictionary with conditions as keys and aggregated y values as values.
        - condition_sem (dict): Dictionary with conditions as keys and standard error of the mean (SEM) values as values.

    Notes:
    - The function aggregates data for each condition and participant.
    - If n_bins is specified, the x data is binned into quantiles.
    - The aggregate_data function is used to compute the aggregated values and SEM.
    """

    condition_x, condition_y, condition_sem = defaultdict(lambda: []), defaultdict(lambda: []), defaultdict(lambda: [])

    for condition in y.conditions: 
        y_ = y.sel(conditions = condition).values.flatten()
        x_ = x.sel(conditions = condition).values.flatten()
        participant_ = (xr.ones_like(y.sel(conditions = condition)) * y.participants).values.flatten()
        x_quantiles = None

        if n_bins is not None: 
            x_ = []
            x_quantiles = []

            # for each participant, bin the x values into quantiles
            for participant in x.participants: 
                x_quantile_, bins = pd.qcut(x.sel(conditions = condition, participants = participant).values.flatten(), n_bins, labels = np.arange(n_bins), retbins = True)
                bin_midpoints = (bins[1:] + bins[:-1])/2
                x_ += list(bin_midpoints[x_quantile_])
                x_quantiles += list(x_quantile_)

        condition_x[condition.item()], condition_y[condition.item()], condition_sem[condition.item()] = aggregate_data(participant_, x_, y_, quantile = x_quantiles)

    return condition_x, condition_y, condition_sem


def aggregate_data(index, x, y, quantile = None):
    '''
    Takes the data and returns summary statistics
    for independent variable x and dependent variable y. 
    There is an option to use discrete quantile bins 
    along with x.
    
    example: 
    index:      ["user1", "user1", "user2", "user2"]
    x:          [1      , 2      , 2      , 1.     ]
    y:          [3.5.   , 4.5.   , 2.4    , 3.0.   ]
    quantile:   [0.     , 1.     , 0.     , 1.     ]
    
    returns the mean of the y, given each bin of x
    
    '''
    if quantile is None: 
        quantile = x
        
    df = pd.DataFrame([])
    df["u"] = index
    df["x"] = x
    df["q"] = quantile
    df["y"] = y

    group = df.groupby(["u", "q"])
    
    return heirarchical_means(group)


def heirarchical_means(group, x = "x", y = "y"):
    mean_x = group[x].apply(np.mean).unstack().values
    mean_y = group[y].apply(np.mean).unstack().values
    var_y = group[y].apply(lambda x: np.var(x, ddof = 1)).unstack().values
    count = group[y].apply(len).unstack().values

    #calculate aggregate statistics
    out_x = np.mean(mean_x, axis = 0)
    out_y = np.mean(mean_y, axis = 0)

    #variance of hierarchical means

    #this is the typical SEM^2
    var_mean = np.var(mean_y, axis = 0, ddof = 1) / len(mean_y)

    #we have to adjust for the additional sample effects
    #to be conservative, use np.min(count) instead of count
    var_mean_adj = 1/(len(count)**2) * np.sum(var_y / np.min(count), axis = 0)
    var_mean_adj = 0
    sem_y = np.sqrt(var_mean + var_mean_adj)

    return out_x, out_y, sem_y
