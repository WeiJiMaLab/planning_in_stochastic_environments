from collections import defaultdict
import numpy as np 
from scipy.optimize import minimize
import tqdm
from prodict import Prodict
import xarray as xr
from utils import *
import itertools

DEFAULT_BOUNDS = {
    "inv_temp": (-5, 5),
    "condition_inv_temp_0": (-5, 5),
    "condition_inv_temp_1": (-5, 5),
    "condition_inv_temp_2": (-5, 5),
    "condition_inv_temp_3": (-5, 5),
    "condition_inv_temp_4": (-5, 5),
    "lapse": (0, 1),
}

class Model(): 
    def __init__(self, effort_version: str, filter_fn: callable, value_fn: callable, variant: str):
        """
            Initializes the Model object.
            Parameters:
            -----------
            effort_version : str
                The version of the effort model to use.
            filter_fn : callable
                The filter function to use.
            value_fn : callable
                The value function to use.
            variant : str
                The stochasticity variant. Options are "R", "V", "T".
            Returns:
            --------
            None
            Raises:
            -------
            AssertionError
                If the effort version is not valid.
                If the variant is not valid.
        """
        assert effort_version in ["filter_adapt", "policy_compress", "hybrid"], "Invalid effort version"
        assert variant in ["R", "V", "T"], "Invalid variant"
        self.effort_version, self.filter_fn, self.value_fn, self.variant = effort_version, filter_fn, value_fn, variant
        # the name of the model is something like "filter_adapt.filter_depth.value_path"
        self.name = self.effort_version + "." + self.filter_fn.__name__ + "." + self.value_fn.__name__

    def fit(self, initial_params : dict, games : list):
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
            Raises:
            -------
            AssertionError
                If the model name in fitted_dict does not match the model's name.
        '''

        #split the initial parameter values and the names
        params_init, params_names = list(initial_params.values()), list(initial_params.keys())

        #we process the game by taking the "point of view" of the player
        game_data = preprocess_data(games)
        
        #minimize the negative log likelihood
        res = minimize(self.negative_log_likelihood, 
                       params_init, 
                       args = (params_names, game_data.pov_array, game_data.choose_left), 
                       bounds = [DEFAULT_BOUNDS[k] for k in params_names])
        
        #now we have selected the optimal set of parameters - we now pass these back into 
        #our maximum likelihood estimator to get the remaining best parameters
        params = dict(zip(params_names, res.x))
        ml, best_filter_params = self.select_best_filter_params(params, game_data.pov_array, game_data.choose_left)
        fitted_params = copy_and_update(params, {"filter_params": best_filter_params, "nll": res.fun, "model": self.name})

        # double check that the negative log likelihood is the same as that from the MLE
        assert np.isclose(-ml, fitted_params["nll"], rtol=1e-6, atol=1e-8)
        
        return fitted_params

    def negative_log_likelihood(self, params_values:list, params_names:list, pov_array:xr.DataArray, choose_left:xr.DataArray): 
        best_log_likelihood, _ = self.select_best_filter_params(
            params = dict(zip(params_names, params_values)), 
            pov_array = pov_array, 
            choose_left = choose_left)
        nll = -best_log_likelihood
        return nll

    def select_best_filter_params(self, params:dict, pov_array:xr.DataArray, choose_left:xr.DataArray): 
        # finds the best set of filter parameters and returns the maximum likelihood and the parameters that achieved it
        p_left = self.get_prob_left(params, pov_array)

        # we have a log likelihood for each condition, game, trial, and filter parameter
        # we want to eventually find the filter parameter that maximizes the log likelihood
        log_likelihood = choose_left * np.log(p_left) + (1 - choose_left) * np.log(1 - p_left)
        
        # for cases where we want ONE filter parameter FOR EACH CONDITION
        if self.effort_version == "filter_adapt" or self.effort_version == "hybrid":

            # we marginalize over games and trials to get the marginal log likelihood for conditions x filter_params
            marginal_log_likelihood = log_likelihood.sum(["games", "trials"], keep_attrs = True)
            best_log_likelihood = marginal_log_likelihood.max("filter_params")

            # get the best filter parameter for each condition
            best_filter_params = xr.DataArray(
                [argmax_random_tiebreaker(marginal_log_likelihood.sel(conditions=c)) for c in marginal_log_likelihood.conditions],
                coords={"conditions": marginal_log_likelihood.conditions},
                dims="conditions"
            )

            return best_log_likelihood.sum(), dict(zip(best_filter_params.conditions.values, best_filter_params.values))
        
        # for cases where we want a GLOBAL filter parameter for ALL CONDITIONS
        elif self.effort_version == "policy_compress": 

            # we marginalize over games, trials, AND conditions to get the marginal log likelihood for filter_params
            marginal_log_likelihood = log_likelihood.sum(["games", "trials", "conditions"], keep_attrs = True)

            # get the filter parameter whose log likelihood is highest
            best_log_likelihood = marginal_log_likelihood.max("filter_params")
            best_filter_param = argmax_random_tiebreaker(marginal_log_likelihood)

            # return the best log likelihood and the best filter parameter
            return best_log_likelihood.item(), {"global": best_filter_param}
        
        else: raise ValueError(f"Invalid effort version: {self.effort_version}")

    def get_prob_left(self, params:dict, pov_array:xr.DataArray):
        """
            Computes the probability of choosing the left option based on the given parameters and filtered point of view array.
            Parameters:
            -----------
            params : dict
                Dictionary containing model parameters, including:
                - "inv_temp": Inverse temperature parameter for the sigmoid function.
                - "condition_lapse_*": Lapse rate parameters for each condition.
            pov_array : xarray.DataArray
                Array containing the filtered point of view data.
            Returns:
            --------
            p_left : xarray.DataArray
                Array containing the probability of choosing the left option for each condition.
        """
        lapse = params["lapse"]

        if self.effort_version == "filter_adapt":
            inv_temp = params["inv_temp"]

        # "hybrid" also includes conditional inverse temperatures
        elif self.effort_version == "policy_compress" or self.effort_version == "hybrid":
            inv_temp = xr.DataArray([params[f"condition_inv_temp_{i}"] for i in range(len(get_stochasticity_levels(self.variant)))], dims="conditions")        
        else: raise ValueError(f"Invalid effort version: {self.effort_version}")
        
        # Compute the probability of choosing the left option using the sigmoid function and apply lapse rate
        p_left = sigmoid(self.value_fn(self.filter_fn(pov_array), variant=self.variant), b_1=np.exp(inv_temp))
        p_left = (1 - lapse) * p_left + lapse * 0.5
        return p_left

    
    def sample(self, params:dict, games:list):
        """
            Samples model choices based on the given parameters and game data.
            Parameters:
            -----------
            params : dict
                A dictionary containing the model parameters, including:
                - "inv_temp" or "condition_inv_temp_*": Inverse temperature parameter(s)
                - "lapse": Lapse rate parameter
                - "filter_params": Filter parameters for each condition or global
            games : list
                A list of game data to be processed and used for sampling.
            Returns:
            --------
            model_data : Prodict
                A Prodict object containing the following keys:
                - boards: The boards from the game data.
                - oracles: The oracles from the game data.
                - choose_left: An array indicating the model's sampled choices (left=1, right=0).
                - paths: The paths from the game data.
                - is_transition: A boolean array indicating transitions in the game data.
        """
        params = copy.deepcopy(params)

        #we process the game by taking the "point of view" of the player
        game_data = preprocess_data(games)

        #apply the filter function to the boards
        p_left_ = self.get_prob_left(params, game_data.pov_array)

        if self.effort_version == "filter_adapt":
            p_left = xr.concat([p_left_.sel(conditions = condition, filter_params = params["filter_params"][condition]) for condition in p_left_.conditions.values], dim = "conditions")
        elif self.effort_version == "policy_compress" or self.effort_version == "hybrid":
            p_left = xr.concat([p_left_.sel(conditions = condition, filter_params = params["filter_params"]["global"]) for condition in p_left_.conditions.values], dim = "conditions")
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
    def __init__(self, model : Model, games : list, params : dict = {}, use_grid : bool = False, n : int = 100):
        self.model = model
        self.games = games
        self.sweeps = []
        self.best = None
        self.best_nll = np.inf

        param_values = {}
        if use_grid:
            # Generate a grid of parameter sets
            for k in params.keys(): 
                lo, hi = DEFAULT_BOUNDS[k]
                param_values[k] = np.linspace(lo, hi, params[k])

            self.param_list = [dict(zip(param_values.keys(), i)) for i in itertools.product(*param_values.values())]
        else:
            # Randomly sample n parameter sets
            assert n > 0, "n must be a positive integer"
            for k in params.keys(): 
                lo, hi = DEFAULT_BOUNDS[k]
                param_values[k] = np.random.uniform(lo, hi, n)

            self.param_list = [dict(zip(param_values.keys(), i)) for i in zip(*param_values.values())]        
            
    def sweep(self): 
        for params in tqdm.tqdm(self.param_list):
            res = self.model.fit(params, self.games)
            self.sweeps.append(res)
            if res["nll"] < self.best_nll: 
                self.best = res
                self.best_nll = res["nll"]

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
def value_EV(pov_array: xr.DataArray, variant = None, value_params = {}): 
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

def get_effort_filter_value_options(type_):
    effort_versions = ["filter_adapt", "policy_compress"]

    # which filter functions to compare to
    filter_fns = [filter_depth, filter_rank, filter_value]

    value_fns = [value_path, value_max, value_sum, value_levelmean]

    if type_ == "R" or type_ == "T": 
        value_fns.append(value_EV)

    return effort_versions, filter_fns, value_fns
    




    


