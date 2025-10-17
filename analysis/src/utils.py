import numpy as np
import json
import copy
import xarray as xr
from collections import defaultdict
from prodict import Prodict
import matplotlib.pyplot as plt
import os
import warnings

########################################################
# Helper functions
########################################################
def alphabet(n):
    """Return the nth letter of the alphabet (0-indexed)"""
    return chr(ord('A') + n)

def copy_and_update(dict_orig, dict_update): 
    dict_copy = copy.deepcopy(dict_orig)
    dict_copy.update(dict_update)
    return dict_copy

def softmax(d, b_0 = 0, b_1 = 1):
    e_x = np.exp(-(b_0 + b_1 * d))
    return e_x / e_x.sum()

def sigmoid(d, b_0 = 0, b_1 = 1): 
    return 1/(1 + np.exp(-(b_0 + b_1 * d)))

def boardify(array, remove_top = True): 
    '''
    sets the upper right triangle of a 
    board to zeros. also sets the value of the
    first node at (0, 0) to zero.
    '''
    x, y = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))
    array[x > y] = 0
    if remove_top:
        array[0, 0] = 0
    return array

def argmax_random_tiebreaker(a):
    return np.random.choice(np.flatnonzero(np.isclose(a, a.max())))


########################################################
# Data loading and preprocessing
########################################################
def get_stochasticity_levels(variant):
    '''
    Returns stochasticity levels for given variant
    '''  
    if variant == "T": 
        return [0, 0.125, 0.25, 0.375, 0.5]
    else: 
        return [0, 0.25, 0.5, 0.75, 1]

def get_data(variant, data_folder = "raw"):
    datafile = f"../data/{data_folder}/data_{variant}.json"
    with open(datafile, 'r') as f:
        data = json.load(f)

    for key in data.keys(): 
        # filter out practice games
        games = format_games(data[key]["data"])
        
        if data_folder == "raw":
            # make sure there are 150 games
            assert len(games) == 150, f"Number of games is not 150 for {key}"

        # update the data
        data[key] = games

    return data

def format_games(games): 
    '''
    removes the practice games from the list of games
    '''
    return [Prodict(g) for g in games if not "practice" in g["name"]]

def preprocess_data(games: list):
    '''
    Creates xarrays for boards, paths, and actions taken (where true is left and false is right)
    for each game - used later in model fitting.
    '''
    boards, paths, actions, oracles, is_transition, rt = defaultdict(lambda: []), defaultdict(lambda: []), defaultdict(lambda: []), defaultdict(lambda: []), defaultdict(lambda: []), defaultdict(lambda: [])
    conditions = []

    for game in games:
        boards[game.p].append(game.boards)
        paths[game.p].append(game.tuplepath)
        actions[game.p].append(game.actions)
        is_transition[game.p].append(game.is_transition)
        oracles[game.p].append([game.oracle]*len(game.boards))
        rt[game.p].append(np.array([trial["rt"] for trial in game.trials])/ 1000)
        conditions.append(game.p)

    ps = sorted(list(set(conditions)))
    game_data = {
        "boards": xr.DataArray([boards[p] for p in ps], dims = ["conditions", "games", "trials", "rows", "cols"]),
        "paths": xr.DataArray([paths[p] for p in ps], dims = ["conditions", "games", "trials", "coords"]),
        "choose_left": xr.DataArray([actions[p] for p in ps], dims = ["conditions", "games", "trials"]).astype(int),
        "oracles": xr.DataArray([oracles[p] for p in ps], dims = ["conditions", "games", "trials", "rows", "cols"]), 
        "is_transition": xr.DataArray([is_transition[p] for p in ps], dims = ["conditions", "games", "trials"]).astype(int),
        "reaction_time": xr.DataArray([rt[p] for p in ps], dims = ["conditions", "games", "trials"]).astype(float) + 1e-10
    }
    game_data["pov_array"] = make_pov_array(game_data["boards"], game_data["paths"])
    
    for key in game_data.keys(): 
        game_data[key]["conditions"] = ps

    player_data = Prodict.from_dict(game_data)
    
    return player_data

def make_pov_array(boards: xr.DataArray, paths: xr.DataArray): 
    '''
    Given a set of boards: dims = (conditions, games, trials, rows, columns)
    and a set of paths: dims = (conditions, games, trials, coord)
    Returns a board from the point-of-view of the player
    such that the position of the board at (game, trial) is matched
    to (0, 0), the upper-left corner of the board
    '''

    boards_pov = np.zeros_like(boards).flatten()

    #get the row and column of the paths taken
    path_row, path_col = paths.sel(coords = 0), paths.sel(coords = 1)
    cols_, rows_ = np.meshgrid(boards.rows, boards.cols)
    rows = xr.ones_like(boards) * rows_
    cols = xr.ones_like(boards) * cols_
    
    #get the indices of the pov_array and match them with the indices
    #of the original array
    pov_indices = np.where(np.logical_and(rows <= np.max(rows) - path_row, cols <= np.max(cols) - path_col), True, False).flatten()
    orig_indices = np.where(np.logical_and(rows >= path_row, cols >= path_col), True, False).flatten()
    
    boards_pov[pov_indices] = boards.values.flatten()[orig_indices]
    boards_pov = xr.DataArray(np.reshape(boards_pov, boards.shape), dims = boards.dims)
    boards_pov["conditions"] = boards.conditions

    #zero the above-diagonal parts of the board
    boards_pov = xr.where(rows >= cols, boards_pov, 0)
    boards_pov = boards_pov.sel(trials = boards_pov.trials[:-1])
    
    return boards_pov


########################################################
# Plotting
########################################################
def strsimplify(num):
    """
    Converts a number to a string with the minimum number of significant figures needed.
    """
    # Convert to string with many decimal places
    s = f"{num:.10f}"
    
    # Remove trailing zeros after decimal
    s = s.rstrip('0')
    
    # Remove decimal point if no decimals remain
    if s.endswith('.'):
        s = s[:-1]
        
    return s

def report_p_value(p, threshold_power = -10): 
    '''
    Basically just prettifies the p-values; 
    Reports exact if p > threshold, otherwise reports p<threshold
    '''
    threshold = 10**threshold_power
    if p < threshold: 
        return f"p < 10^{{{threshold_power}}}"
    else:
        # Convert to scientific notation with 2 sigfigs
        p_str = f"{p:.1e}"
        # Split into mantissa and exponent
        mantissa, exponent = p_str.split('e')
        # Remove leading zero from exponent
        exponent = str(int(exponent))
        return f"p = {mantissa} \\times 10^{{{exponent}}}"

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def get_jobid():
    # detect if running in SLURM environment
    try:
        jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        print("Job ID\t", jobid)
    except TypeError:
        warnings.warn("Not running in SLURM environment. Setting job ID to 1.")
        jobid = 1
    return jobid
