import numpy as np
import json
import copy
import xarray as xr
from collections import defaultdict
from prodict import Prodict
import matplotlib.pyplot as plt

########################################################
# Helper functions
########################################################

class mapdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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

def get_data(variant):
    datafile = "../data/raw/data_%s.json"%(variant)
    with open(datafile, 'r') as f:
        data = json.load(f)
    return data

def format_games(games): 
    '''
    removes the practice games from the list of games
    '''
    return [mapdict(g) for g in games if not "practice" in g["name"]]

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

colormaps_ = {
    "blurple": ['#0c3679', '#807ad1', '#d067b9'],
    "political": ['#244b89', '#c6c9cd', '#b33756'],
    "popsicle": ['#ff5e57', '#ffbd69', '#2ec4b6'],
    "lavender": ['#240372', '#ecd5db', '#005ca3'],
    "sunset": ['#240372', '#d3033b', '#db7b51'],
    "arctic": ['#080745', '#246c99', '#92d9d4'],
    "easter": ['#eea79b', '#eca7dd', '#80b3ea', '#3cc3b3'],
    'countyfair': ['#f59e9e', '#a894d6', '#164374', '#2c9da5'],
    "playdough":['#d52320', '#2a78c0', '#06a288', '#fbae41'],
    "foliage": ['#d53f3f', '#ffc894', '#076e62'],
    "rouge": ['#6b0037', '#b40439', '#f2a673'],
    "berry": ['#64006b', '#b40462', '#f27373'],
    "sage": ['#043d48', '#196c2b', '#92a592'],
    "grass": ['#00663f', '#43896b', '#b9c997'],
}

colormaps = {name: plt.cm.colors.LinearSegmentedColormap.from_list(name, colors) for name, colors in colormaps_.items()}

def report_p_value(p, threshold = 1e-10): 
    '''
    Basically just prettifies the p-values; 
    Reports exact if p > threshold, otherwise reports p<threshold
    '''
    if p < threshold: 
        return f"p < {threshold:.1e}"
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
