from collections import defaultdict
import numpy as np 
from scipy.optimize import minimize
import tqdm
from prodict import Prodict
import xarray as xr
from utils import *
from modelfilters import filter_depth, filter_rank, filter_value
from modelvalues import value_path, value_max, value_sum, value_levelmean, value_EV
import itertools
import copy

# default bounds for the parameters to be fitted
DEFAULT_BOUNDS = {
    "inv_temp": (-5, 5),
    "condition_inv_temp_0": (-5, 5),
    "condition_inv_temp_1": (-5, 5),
    "condition_inv_temp_2": (-5, 5),
    "condition_inv_temp_3": (-5, 5),
    "condition_inv_temp_4": (-5, 5),
    "lapse": (0, 1),
}

def get_effort_filter_value_options(type_):
    # return the possible effort versions, filter functions, and value functions for a given variant
    effort_versions = ["filter_adapt", "policy_compress"]
    filter_fns = [filter_depth, filter_rank, filter_value]
    value_fns = [value_path, value_max, value_sum, value_levelmean]
    if type_ == "R" or type_ == "T": value_fns.append(value_EV)
    return effort_versions, filter_fns, value_fns
    
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
        assert variant in ["R", "V", "T"], "Invalid variant"
        self.effort_version, self.filter_fn, self.value_fn, self.variant = effort_version, filter_fn, value_fn, variant
        self.name = self.effort_version + "." + self.filter_fn.__name__ + "." + self.value_fn.__name__
        
        if self.effort_version == "filter_adapt": 
            self.conditional_filter = True
            self.conditional_inv_temp = False
        
        elif self.effort_version == "policy_compress": 
            self.conditional_filter = False
            self.conditional_inv_temp = True
        
        elif self.effort_version == "hybrid": 
            self.conditional_filter = True
            self.conditional_inv_temp = True
        
        else: raise ValueError(f"Invalid effort version: {self.effort_version}")

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
        game_data = preprocess_data(games)
        
        #minimize the negative log likelihood
        res = minimize(self.negative_log_likelihood, params_init, 
                       args = (params_names, game_data.pov_array, game_data.choose_left), 
                       bounds = [DEFAULT_BOUNDS[k] for k in params_names])
        
        #now we have selected the optimal set of parameters - we now pass these back into 
        #our maximum likelihood estimator to get the remaining best parameters
        params = dict(zip(params_names, res.x))
        best_ll, best_filter_params = self.select_best_filter_params(params, game_data.pov_array, game_data.choose_left)
        fitted_params = copy_and_update(params, {"filter_params": best_filter_params, "nll": res.fun, "model": self.name})

        # double check that the negative log likelihood is the same as that from the MLE
        assert np.isclose(-best_ll, fitted_params["nll"], rtol=1e-5)
        return fitted_params

    def negative_log_likelihood(self, params_values, params_names, pov_array, choose_left):
        return -self.select_best_filter_params(dict(zip(params_names, params_values)), pov_array, choose_left)[0]


    def select_best_filter_params(self, params:dict, pov_array:xr.DataArray, choose_left:xr.DataArray): 
        # finds the best set of filter parameters and returns the maximum likelihood and the parameters that achieved it
        p_left = self.get_prob_left(params, pov_array)
        log_likelihood = choose_left * np.log(p_left) + (1 - choose_left) * np.log(1 - p_left)
        
        # for cases where we want ONE filter parameter FOR EACH CONDITION
        if self.conditional_filter:
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
        else:
            # we marginalize over games, trials, AND conditions to get the marginal log likelihood for filter_params
            # and get the filter parameter that maximizes the log likelihood
            marginal_log_likelihood = log_likelihood.sum(["games", "trials", "conditions"], keep_attrs = True)
            best_log_likelihood = marginal_log_likelihood.max("filter_params")
            best_filter_param = argmax_random_tiebreaker(marginal_log_likelihood)
            return best_log_likelihood.item(), {"global": best_filter_param}
        

    def get_prob_left(self, params:dict, pov_array:xr.DataArray):
        """
            Computes the probability of choosing the left option based on the given parameters and filtered point of view array.
            Parameters:
            -----------
            params : dict
                Dictionary containing model parameters, including:
                - "inv_temp": Inverse temperature parameter for the sigmoid function. Note that this is in log space, so can take negative or positive values.
                - "condition_inv_temp_*": Inverse temperature parameters for each condition.
            pov_array : xarray.DataArray
                Array containing the filtered point of view data.
            Returns:
            --------
            p_left : xarray.DataArray
                Array containing the probability of choosing the left option for each condition.
        """
        lapse = params["lapse"]

        if self.conditional_inv_temp:
            inv_temp = xr.DataArray([params[f"condition_inv_temp_{i}"] for i in range(len(get_stochasticity_levels(self.variant)))], dims="conditions")
        else:
            inv_temp = params["inv_temp"]

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

        if self.conditional_filter:
            p_left = xr.concat([p_left_.sel(
                conditions = condition, 
                filter_params = params["filter_params"][condition]) 
                for condition in p_left_.conditions.values], 
                dim = "conditions"
            )
        else:
            p_left = xr.concat([p_left_.sel(conditions = condition, 
                filter_params = params["filter_params"]["global"]) 
                for condition in p_left_.conditions.values], 
                dim = "conditions"
            )
        
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