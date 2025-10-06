from collections import defaultdict
import numpy as np 
from scipy.optimize import minimize
import tqdm
from prodict import Prodict
import xarray as xr
from utils import *
import itertools

class MultiStart(): 
    """
    A class to perform multi-start optimization for a given model over a set of games.
    Attributes:
    -----------
    model : object
        The model to be optimized.
    games : list
        A list of games to be used for fitting the model.
    sweeps : list
        A list to store the results of each parameter sweep.
    best : dict or None
        The best result obtained from the sweeps.
    best_nll : float
        The best negative log-likelihood obtained from the sweeps.
    param_list : list
        A list of dictionaries containing parameter combinations to be tested.
    Methods:
    --------
    __init__(model, games, grid_params={}):
        Initializes the MultiStart object with the given model, games, and grid parameters.
    sweep():
        Performs a parameter sweep over the grid parameters and updates the best result.
    """
    def __init__(self, model, games, params = {}, use_grid = True, n = 100):
        self.model = model
        self.games = games
        self.sweeps = []
        self.best = None
        self.best_nll = np.inf
        
        param_ranges = {}
        if use_grid:
            # Generate a grid of parameter sets
            for k in params.keys(): 
                lo, hi = self.model.default_bounds[k]
                param_ranges[k] = np.linspace(lo, hi, params[k])

            self.param_list = [dict(zip(param_ranges.keys(), i)) for i in itertools.product(*param_ranges.values())]
        else:
            # Randomly sample n parameter sets
            assert n > 0, "n must be a positive integer"
            for k in params.keys(): 
                lo, hi = self.model.default_bounds[k]
                param_ranges[k] = np.random.uniform(lo, hi, n)

            self.param_list = [dict(zip(param_ranges.keys(), i)) for i in zip(*param_ranges.values())]        
            
    def sweep(self): 
        for params in tqdm.tqdm(self.param_list):
            res = self.model.fit(params, self.games)
            self.sweeps.append(res)
            if res["nll"] < self.best_nll: 
                self.best = res
                self.best_nll = res["nll"]

class Model(): 
    """
    Model class for fitting and predicting game behavior based on specified filter and value functions.
    Attributes:
    filter_fn : function
        The filter function to be applied to the point of view array.
    value_fn : function
        The value function to be used in the model.
    name : str
        The name of the model, derived from the value and filter function names.
    variant : str
        The variant of the model.
    default_params : dict
        Default parameters for the model, including "inv_temp" and "lapse".
    default_bounds : dict
        Default bounds for the parameters, including "inv_temp", "lapse", and various conditions.
    Methods:
    __init__(filter_fn, value_fn, variant)
        Initializes the Model with the given filter and value functions, and variant.
    fit(fittable_params: dict, games: list) -> dict
    negative_log_likelihood(params_values, params_names, filter_pov_array, choose_left) -> float
        Computes the negative log likelihood for the given parameters and data.
    max_likelihood_estimation(params: dict, filter_pov_array, choose_left) -> tuple
        Performs maximum likelihood estimation for the given parameters and data.
    predict(fitted_dict: dict, games: list) -> Prodict
    """
    def __init__(self, filter_fn, value_fn, variant, conditional_filter_params = True): 
        self.filter_fn = filter_fn
        self.value_fn = value_fn
        self.name = self.value_fn.__name__  + "." + self.filter_fn.__name__ 
        self.variant = variant
        self.default_params = {p:p for p in get_stochasticity_levels(self.variant)}
        self.default_bounds = {
                "inv_temp": (-5, 5),
                "condition_inv_temp_0": (-5, 5),
                "condition_inv_temp_1": (-5, 5),
                "condition_inv_temp_2": (-5, 5),
                "condition_inv_temp_3": (-5, 5),
                "condition_inv_temp_4": (-5, 5),
                "lapse": (0, 1),
                "condition_lapse_0": (0, 1),
                "condition_lapse_1": (0, 1),
                "condition_lapse_2": (0, 1),
                "condition_lapse_3": (0, 1),
                "condition_lapse_4": (0, 1),
        }
        self.default_bounds.update({p:(0, 1) for p in get_stochasticity_levels(self.variant)})

        # if set to false, we will only fit a single filter parameter for each condition, 
        # default is TRUE
        self.conditional_filter_params = conditional_filter_params

    def fit(self, fittable_params:dict, games):
        '''
            Fits the model to the given set of games by finding the parameter values that minimize the negative log likelihood.
            Parameters:
            -----------
            fittable_params : dict
                A dictionary where keys are the names of the parameters to be fitted and values are their initial guesses.
            games : list
                A list of game data to be used for fitting the model.
            Returns:
            --------
            fitted_params : dict
                A dictionary containing the fitted parameters, including:
                - The optimized parameter values.
                - The negative log likelihood (nll) of the fit.
                - The model name.
                - The filter parameters used in the model.
        '''

        #split the parameter values and the names
        params_init, params_names = list(fittable_params.values()), list(fittable_params.keys())

        #we process the game by taking the "point of view" of the player
        game_data = preprocess_data(games)
        pov_array = make_pov_array(game_data.boards, game_data.paths)

        #apply the filter function to the boards
        filter_pov_array = self.filter_fn(pov_array)

        #minimize the negative log likelihood
        res = minimize(self.negative_log_likelihood, 
                       params_init, 
                       args = (params_names, filter_pov_array, game_data.choose_left), 
                       bounds = [self.default_bounds[k] for k in fittable_params.keys()])
        
        #now we have selected the optimal set of parameters - we now pass these back into 
        #our maximum likelihood estimator to get the remaining best parameters
        params = copy_and_update(self.default_params, dict(zip(params_names, res.x)))
        ml, ml_params = self.max_likelihood_estimation(params, filter_pov_array, game_data.choose_left)

        #get the parameters of interest and return them
        fitted_params = {k: ml_params[k] for k in params_names + ["filter_params"]}
        fitted_params["nll"] = res.fun
        fitted_params["model"] = self.name
        assert np.isclose(-ml, res.fun)
        
        return fitted_params

    def negative_log_likelihood(self, params_values, params_names, filter_pov_array, choose_left): 
        params = copy_and_update(self.default_params, dict(zip(params_names, params_values)))
        ml, _ = self.max_likelihood_estimation(params, filter_pov_array, choose_left)
        nll = -ml
        return nll.item()

    def max_likelihood_estimation(self, params, filter_pov_array, choose_left): 
        """
            Perform maximum likelihood estimation for the given parameters and data.
            Parameters:
            -----------
            params : dict
                Dictionary containing model parameters, including:
                - "inv_temp": Inverse temperature parameter for the sigmoid function.
                - "lapse": Lapse rate parameter.
            filter_pov_array : xarray.DataArray
                Array containing the filtered point of view data.
            choose_left : xarray.DataArray
                Array indicating whether the left option was chosen (1) or not (0).
            Returns:
            --------
            ml : xarray.DataArray
                Maximum log-likelihood value across all filter parameters.
            updated_params : dict
                Updated parameters dictionary with the best filter parameters based on maximum likelihood estimation.
            Notes:
            ------
            - The function uses a sigmoid function to calculate the probability of choosing the left option.
            - The log-likelihood is computed for each trial and summed across games and trials.
            - The maximum log-likelihood is determined, and the best filter parameters are selected using a random tie-breaking strategy.
        """
        p_left = self.get_p_left(params, filter_pov_array)

        log_likelihood = choose_left * np.log(p_left) + (1 - choose_left) * np.log(1 - p_left)
        
        if self.conditional_filter_params:
            log_likelihood_player = log_likelihood.sum(["games", "trials"], keep_attrs = True)
            ml = log_likelihood_player.max("filter_params")
            mle = [argmax_random_tiebreaker(log_likelihood_player.sel(conditions = condition)) for condition in log_likelihood_player.conditions]
            mle = xr.DataArray(mle, dims = "conditions")
            mle["conditions"] = log_likelihood_player.conditions
            return ml.sum(), copy_and_update(params, {"filter_params": dict(zip(mle.conditions.values, mle.values))})
        
        else: 
            log_likelihood_player = log_likelihood.sum(["games", "trials", "conditions"], keep_attrs = True)
            ml = log_likelihood_player.max("filter_params")
            mle = argmax_random_tiebreaker(log_likelihood_player)
            return ml.sum(), copy_and_update(params, {"filter_params": mle})

    def get_p_left(self, params, filter_pov_array):
        """
        Computes the probability of choosing the left option based on the given parameters and filtered point of view array.
        Parameters:
        -----------
        params : dict
            Dictionary containing model parameters, including:
            - "inv_temp": Inverse temperature parameter for the sigmoid function.
            - "condition_lapse_*": Lapse rate parameters for each condition.
        filter_pov_array : xarray.DataArray
            Array containing the filtered point of view data.
        Returns:
        --------
        p_left : xarray.DataArray
            Array containing the probability of choosing the left option for each condition.
        """
        if "inv_temp" in params.keys(): 
            inv_temp = params["inv_temp"]
        elif "condition_inv_temp_0" in params.keys():
            # using conditional inverse temperatures
            inv_temp = xr.DataArray([params[f"condition_inv_temp_{i}"] for i in range(len(get_stochasticity_levels(self.variant)))], dims="conditions")
        else:
            inv_temp = 0
        
        if "lapse" in params.keys():
            lapse = params["lapse"]
        elif "condition_lapse_0" in params.keys():
            # using conditional lapse rates
            lapse = xr.DataArray([params[f"condition_lapse_{i}"] for i in range(len(get_stochasticity_levels(self.variant)))], dims="conditions")
        else:
            lapse = 0
        
        # Compute the probability of choosing the left option using the sigmoid function
        p_left = sigmoid(self.value_fn(filter_pov_array, variant=self.variant, value_params=params), b_1=np.exp(inv_temp))
        
        # Adjust the probability with the lapse rate
        p_left = (1 - lapse) * p_left + lapse * 0.5

        return p_left

    
    def predict(self, fitted_dict:dict, games:list):
        """
        Predicts the model's behavior based on the fitted parameters and game data.
        Parameters:
        -----------
        fitted_dict : dict
            A dictionary containing the fitted parameters for the model. Must include the key "model" which should match the model's name.
        games : list
            A list of game data to be processed and used for prediction.
        Returns:
        --------
        model_data : Prodict
            A Prodict object containing the following keys:
            - boards: The boards from the game data.
            - oracles: The oracles from the game data.
            - choose_left: An array indicating the model's choice (left or not) for each game condition.
            - paths: The paths from the game data.
            - is_transition: A boolean array indicating transitions in the game data.
        Raises:
        -------
        AssertionError
            If the model name in fitted_dict does not match the model's name.
        """
        assert fitted_dict["model"] == self.name
        params = copy_and_update(self.default_params, fitted_dict)

        #we process the game by taking the "point of view" of the player
        game_data = preprocess_data(games)
        pov_array = make_pov_array(game_data.boards, game_data.paths)

        #apply the filter function to the boards
        filter_pov_array = self.filter_fn(pov_array)

        p_left_ = self.get_p_left(params, filter_pov_array)
        if "global_depth" in params["filter_params"].keys():
            p_left = xr.concat([p_left_.sel(conditions = condition, filter_params = params["filter_params"]["global_depth"]) for condition in p_left_.conditions.values], dim = "conditions")
        else:
            p_left = xr.concat([p_left_.sel(conditions = condition, filter_params = params["filter_params"][condition]) for condition in p_left_.conditions.values], dim = "conditions")
        p_left['conditions'] = p_left_.conditions
    
        choose_left = (np.random.rand(*p_left.shape) < p_left).astype(int)

        model_data = {
                        "boards": game_data.boards,
                        "oracles": game_data.oracles,
                        "choose_left": choose_left,
                        "paths": game_data.paths,
                        "is_transition": game_data.is_transition,
                    }

        model_data = Prodict(model_data)
        return model_data


# Filter functions
def filter_value(pov_array: xr.DataArray, filter_params: dict= {"value": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}):
    assert "value" in filter_params.keys(), "Missing value in filter_params"
    filtered_array = xr.concat([xr.where(pov_array >= 10 - i, pov_array.copy(), 0) for i in filter_params["value"]], dim = "filter_params")
    return filtered_array

def filter_rank(pov_array: xr.DataArray, filter_params: dict= {"rank": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}):
    assert "rank" in filter_params.keys(), "Missing rank in filter_params"

    filtered_array = []
    thresh_pov_array = pov_array.copy()

    for _ in filter_params["rank"]:
        thresh_pov_array = xr.where(thresh_pov_array > thresh_pov_array.min(["rows", "cols"]), thresh_pov_array, np.nan)
        filtered_array.append(thresh_pov_array.copy().fillna(0))

    filtered_array = xr.concat(filtered_array[::-1], dim = "filter_params")
    return filtered_array

def filter_depth(pov_array: xr.DataArray, filter_params: dict = {"depth": [0, 1, 2, 3, 4, 5, 6, 7]}): 
    assert "depth" in filter_params.keys(), "Missing depth in filter_params"
    
    filtered_array = xr.concat([xr.where(pov_array.rows <= i, pov_array.copy(), 0) for i in filter_params["depth"]], dim = "filter_params")
    filtered_array["filter_params"] = filter_params["depth"]
    return filtered_array

# Value functions
def value_EV(pov_array: xr.DataArray, variant, value_params): 
    value_params_ = {"exp_value": 5, 0:0, 0.125:0.125, 0.25:0.25, 0.375:0.375, 0.5:0.5, 0.75:0.75, 1:1}
    value_params_.update(value_params)
    
    if variant == "R":
        v_diff = []

        for condition in pov_array.conditions:
            q = 1 - value_params_[condition.item()]
            E = value_params_["exp_value"]
            
            pov_array_condition = pov_array.sel(conditions = condition)
            values = pov_array_condition.copy()
            
            table = xr.zeros_like(values).astype(float)
            
            #base case: T[-1, col] = qV[-1, col] + (1 - q)E
            table[dict(rows = max(table.rows))] = q * values.isel(rows = max(table.rows)) + (1 - q) * E

            for row in pov_array.rows.values[:-1][::-1]:
                for col in range(row + 1): 
                    # T[row, col] = qV[row, col] + (1 - q)E + max(T[row + 1, col], T[row + 1, col + 1])
                    table[dict(rows = row, cols = col)] = q * values.isel(rows = row, cols = col) + (1 - q) * E + table.isel(rows = row + 1, cols = [col, col + 1]).max("cols")

            #decision variable for each game, trial is T[1, 0] - T[1, 1]
            v_left = table.isel(rows = 1, cols = 0)
            v_right = table.isel(rows = 1, cols = 1)
            v_diff_ = v_left - v_right
            v_diff.append(v_diff_)
            
        v_diff = xr.concat(v_diff, dim = "conditions")
        v_diff["conditions"] = pov_array.conditions
        
        return v_diff
    
    if variant == "T":
        v_diff = []
        for condition in pov_array.conditions:
            q = 1 - value_params_[condition.item()]

            pov_array_condition = pov_array.sel(conditions = condition)
            values = pov_array_condition.copy()

            table = xr.zeros_like(values).astype(float)

            #base case: T[-1, col] = qV[-1, col] + (1 - q)E
            table[dict(rows = max(table.rows))] = values.isel(rows = max(table.rows))

            for row in pov_array.rows.values[:-1][::-1]:
                for col in range(row + 1): 
                    # T[row, col] = qV[row, col] + (1 - q)E + max(T[row + 1, col], T[row + 1, col + 1])
                    v_left = (table.isel(rows = row + 1, cols = [col, col + 1]) * [q, (1 - q)]).sum("cols")
                    v_right = (table.isel(rows = row + 1, cols = [col, col + 1]) * [(1 - q), q]).sum("cols")
                    v_max = xr.concat([v_left, v_right], dim = "cols").max("cols")
                    
                    table[dict(rows = row, cols = col)] = values.isel(rows = row, cols = col) + v_max
                    
            #decision variable for each game, trial is T[1, 0] - T[1, 1]
            v_left = (table.isel(rows = 1, cols = [0, 1]) * [q, (1 - q)]).sum("cols")
            v_right = (table.isel(rows = 1, cols = [0, 1]) * [(1 - q), q]).sum("cols")
            v_diff_ = v_left - v_right
            v_diff.append(v_diff_)
            
        v_diff = xr.concat(v_diff, dim = "conditions")
        v_diff["conditions"] = pov_array.conditions
        return v_diff

def value_path(pov_array: xr.DataArray, variant = None, value_params = None):
    values = pov_array.copy()
    
    table = xr.zeros_like(values).astype(float)
    table[dict(rows = max(table.rows))] = values.isel(rows = max(table.rows))

    for row in pov_array.rows.values[:-1][::-1]:
        for col in range(row + 1): 
            # T[row, col] = V[row, col] + max(T[row + 1, col], T[row + 1, col + 1])
            table[dict(rows = row, cols = col)] = values.isel(rows = row, cols = col) + table.isel(rows = row + 1, cols = [col, col + 1]).max("cols")

    #decision variable for each game, trial is T[1, 0] - T[1, 1]
    v_left = table.isel(rows = 1, cols = 0)
    v_right = table.isel(rows = 1, cols = 1)
    v_diff = v_left - v_right
    
    return v_diff

def value_levelmean(pov_array: xr.DataArray, variant = None, value_params = None):
    values = pov_array.copy()
    left = values.isel(rows = values.rows[1:], cols = values.cols[:-1])
    right = values.isel(rows = values.rows[1:], cols = values.cols[1:])
    
    #set the above-diagonal values to np.nan
    left_subtree = left.where(left.rows >= left.cols)
    right_subtree = right.where(right.rows >= right.cols)

    v_left = left_subtree.mean("cols").sum("rows")
    v_right = right_subtree.mean("cols").sum("rows")
    
    return v_left - v_right

def value_sum(pov_array: xr.DataArray, variant = None, value_params = None):
    values = pov_array.copy()
    left = values.isel(rows = values.rows[1:], cols = values.cols[:-1])
    right = values.isel(rows = values.rows[1:], cols = values.cols[1:])

    #set the above-diagonal values to np.nan
    left_subtree = left.where(left.rows >= left.cols)
    right_subtree = right.where(right.rows >= right.cols)
    
    v_left = left_subtree.sum(["rows", "cols"])
    v_right = right_subtree.sum(["rows", "cols"])
    
    return v_left - v_right


def value_max(pov_array: xr.DataArray, variant = None, value_params = None):
    values = pov_array.copy()
    left = values.isel(rows = values.rows[1:], cols = values.cols[:-1])
    right = values.isel(rows = values.rows[1:], cols = values.cols[1:])

    #set the above-diagonal values to np.nan
    left_subtree = left.where(left.rows >= left.cols)
    right_subtree = right.where(right.rows >= right.cols)
    
    v_left = left_subtree.max(["rows", "cols"])
    v_right = right_subtree.max(["rows", "cols"])
    
    return v_left - v_right

    




    


