import numpy as np
import xarray as xr
from collections import defaultdict
from prodict import Prodict
from numba import jit

def softmax(d, b_0 = 0, b_1 = 1):
    e_x = np.exp(-(b_0 + b_1 * d))
    return e_x / e_x.sum()

@jit(nopython=True)
def sigmoid(d, b_0 = 0, b_1 = 1, norm = None): 
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

def subtree(board, r, c, remove_top = False): 
    '''
    Given a board and a root cell (r, c), returns the subtree rooted at
    (r, c)
    inputs: 
        board: a board or subtree
    returns: 
        subtree
    '''
    subboard = board[r:, c:].copy()
    n = min(subboard.shape[0], subboard.shape[1])
    subboard = subboard[:n, :n]
    x, y = np.meshgrid(np.arange(subboard.shape[1]), np.arange(subboard.shape[0]))
    subboard[x > y] = 0
    
    if remove_top: 
        subboard[0, 0] = 0
        
    return subboard

def argmax_random_tiebreaker(a):
    return np.random.choice(np.flatnonzero(np.isclose(a, a.max())))

def preprocess_games(games: list):
    '''
    Creates xarrays for boards, paths, and actions taken (where true is left and false is right)
    for each game - used later in model fitting.
    '''
    boards, paths, actions = (defaultdict(lambda: []), defaultdict(lambda: []), defaultdict(lambda: []))
    conditions = []

    for game in games:
        boards[game.p].append(game.boards)
        paths[game.p].append(game.tuplepath)
        actions[game.p].append(game.actions)
        conditions.append(game.p)

    ps = sorted(list(set(conditions)))

    board_array = xr.DataArray([boards[p] for p in ps], dims = {"conditions": ps, "games": None, "trials": None, "rows": None, "cols":None})
    path_array = xr.DataArray([paths[p] for p in ps], dims = {"conditions": ps, "games": None, "trials": None, "coords":None})
    choose_left = xr.DataArray([actions[p] for p in ps], dims = {"conditions": ps, "games": None, "trials": None}).astype(int)

    board_array["conditions"], path_array["conditions"], choose_left["conditions"] =  ps, ps, ps
    
    return board_array, path_array, choose_left


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
