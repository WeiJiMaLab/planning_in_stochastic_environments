import numpy as np
import xarray as xr
import pytest 
import os
import sys
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from modeling import (
    Model, 
    filter_depth,
    filter_rank,
    filter_value,
    value_EV,
    value_path, 
    value_max,
    value_sum,
    value_levelmean,
)
from utils import make_pov_array, get_stochasticity_levels

def get_symmetric_board_pov(type_, paths = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]):
    # this board should always have a 50% chance of moving left no matter what the model is 
    board = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0], 
            [1, 1, 0, 0, 0, 0, 0, 0], 
            [1, 1, 1, 0, 0, 0, 0, 0], 
            [1, 1, 1, 1, 0, 0, 0, 0], 
            [1, 1, 1, 1, 1, 0, 0, 0], 
            [1, 1, 1, 1, 1, 1, 0, 0], 
            [1, 1, 1, 1, 1, 1, 1, 0], 
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]
    )
    n_games = 5
    n_conditions = 5
    boards = [board]*len(board - 1)
    boards = [boards]*n_games
    boards = [boards]*n_conditions
    boards = xr.DataArray(boards, dims = ["conditions","games", "trials", "rows", "cols"])
    paths = paths
    paths = [paths]*n_games
    paths = [paths]*n_conditions
    paths = xr.DataArray(paths, dims = ["conditions","games", "trials", "coords"])
    boards["conditions"] = get_stochasticity_levels(type_)
    paths["conditions"] = get_stochasticity_levels(type_)
    pov_array = make_pov_array(boards, paths)
    return pov_array

def get_asymmetric_board_pov(type_, paths = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]):
    # this board should always have a 100% chance of moving left
    # for all nonrandom filter functions and value functions 
    # exclusions: value_EV when stoch = 100%; filter functions when threshold = 0; when lapse = 1; inv_temp = -100
    board = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0], 
            [9, 1, 0, 0, 0, 0, 0, 0], 
            [9, 1, 1, 0, 0, 0, 0, 0], 
            [9, 1, 1, 1, 0, 0, 0, 0], 
            [9, 1, 1, 1, 1, 0, 0, 0], 
            [9, 1, 1, 1, 1, 1, 0, 0], 
            [9, 1, 1, 1, 1, 1, 1, 0], 
            [9, 1, 1, 1, 1, 1, 1, 1]
        ]
    )

    n_games = 5
    n_conditions = 5
    boards = [board]*len(board - 1)
    boards = [boards]*n_games
    boards = [boards]*n_conditions
    boards = xr.DataArray(boards, dims = ["conditions","games", "trials", "rows", "cols"])
    paths = paths
    paths = [paths]*n_games
    paths = [paths]*n_conditions
    paths = xr.DataArray(paths, dims = ["conditions","games", "trials", "coords"])
    boards["conditions"] = get_stochasticity_levels(type_)
    paths["conditions"] = get_stochasticity_levels(type_)
    pov_array = make_pov_array(boards, paths)
    return pov_array


@pytest.mark.parametrize("type_", ["R", "V", "T"])
@pytest.mark.parametrize("effort_version", ["filter_adapt", "policy_compress"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_path, value_max, value_sum, value_levelmean])
def test_symmetric_equalprob(type_,effort_version, filter_fn, value_fn):
    pov_array = get_symmetric_board_pov(type_)
    if effort_version == "filter_adapt":
        params = {
            "inv_temp": 10,
            "lapse": 0
        }
    elif effort_version == "policy_compress":
        params = {
            "condition_inv_temp_0": 10,
            "condition_inv_temp_1": 10,
            "condition_inv_temp_2": 10,
            "condition_inv_temp_3": 10,
            "condition_inv_temp_4": 10,
            "lapse": 0
        }
    model = Model(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_
    )
    filter_pov_array = model.filter_fn(pov_array)
    p_left = model.get_p_left(params, filter_pov_array)
    assert np.allclose(p_left, 0.5)


@pytest.mark.parametrize("type_", ["R", "T"])
@pytest.mark.parametrize("effort_version", ["filter_adapt", "policy_compress"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_EV])
def test_symmetric_equalprob_EV(type_,effort_version, filter_fn, value_fn):
    pov_array = get_symmetric_board_pov(type_)
    if effort_version == "filter_adapt":
        params = {
            "inv_temp": 10,
            "lapse": 0
        }
    elif effort_version == "policy_compress":
        params = {
            "condition_inv_temp_0": 10,
            "condition_inv_temp_1": 10,
            "condition_inv_temp_2": 10,
            "condition_inv_temp_3": 10,
            "condition_inv_temp_4": 10,
            "lapse": 0
        }
    model = Model(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_
    )
    filter_pov_array = model.filter_fn(pov_array)
    p_left = model.get_p_left(params, filter_pov_array)
    assert np.allclose(p_left, 0.5)

@pytest.mark.parametrize("type_", ["R", "V", "T"])
@pytest.mark.parametrize("effort_version", ["filter_adapt", "policy_compress"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_path, value_max, value_sum, value_levelmean])
def test_asymmetric_always_left(type_,effort_version, filter_fn, value_fn):
    pov_array = get_asymmetric_board_pov(type_)

    if effort_version == "filter_adapt":
        params = {
            "inv_temp": 10,
            "lapse": 0
        }

    elif effort_version == "policy_compress":
        params = {
            "condition_inv_temp_0": 10,
            "condition_inv_temp_1": 10,
            "condition_inv_temp_2": 10,
            "condition_inv_temp_3": 10,
            "condition_inv_temp_4": 10,
            "lapse": 0
        }

    model = Model(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_
    )

    filter_pov_array = model.filter_fn(pov_array, {"depth": [7], "rank": [9], "value": [9]})
    p_left = model.get_p_left(params, filter_pov_array)
    assert np.allclose(p_left, 1)

@pytest.mark.parametrize("type_", ["R", "T"])
@pytest.mark.parametrize("effort_version", ["filter_adapt", "policy_compress"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_EV])
def test_asymmetric_always_left_EV(type_,effort_version, filter_fn, value_fn):
    pov_array = get_asymmetric_board_pov(type_)

    if effort_version == "filter_adapt":
        params = {
            "inv_temp": 10,
            "lapse": 0
        }

    elif effort_version == "policy_compress":
        params = {
            "condition_inv_temp_0": 10,
            "condition_inv_temp_1": 10,
            "condition_inv_temp_2": 10,
            "condition_inv_temp_3": 10,
            "condition_inv_temp_4": 10,
            "lapse": 0
        }

    model = Model(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_
    )

    filter_pov_array = model.filter_fn(pov_array, {"depth": [7], "rank": [9], "value": [9]})
    p_left = model.get_p_left(params, filter_pov_array)
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[0]), 1)
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[1]), 1)
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[2]), 1)
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[3]), 1)
    # in EV, we should get a 50% chance of moving left for maximum stochasticity
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[4]), 0.5)




@pytest.mark.parametrize("type_", ["R", "V", "T"])
@pytest.mark.parametrize("effort_version", ["filter_adapt", "policy_compress"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_path, value_max, value_sum, value_levelmean])
def test_lapse(type_,effort_version, filter_fn, value_fn):
    pov_array = get_asymmetric_board_pov(type_)

    # if we set the lapse to 1, we should *undo* the asymmetric effect
    # and we should get a 50% chance of moving left again
    if effort_version == "filter_adapt":
        params = {
            "inv_temp": 10,
            "lapse": 1
        }

    elif effort_version == "policy_compress":
        params = {
            "condition_inv_temp_0": 10,
            "condition_inv_temp_1": 10,
            "condition_inv_temp_2": 10,
            "condition_inv_temp_3": 10,
            "condition_inv_temp_4": 10,
            "lapse": 1
        }

    model = Model(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_
    )

    filter_pov_array = model.filter_fn(pov_array, {"depth": [7], "rank": [9], "value": [9]})
    p_left = model.get_p_left(params, filter_pov_array)
    assert np.allclose(p_left, 0.5)

@pytest.mark.parametrize("type_", ["R", "V", "T"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_path, value_max, value_sum, value_levelmean])
def test_inv_temp(type_, filter_fn, value_fn):
    pov_array = get_asymmetric_board_pov(type_)

    effort_version = "filter_adapt"
    # if we set the inv_temp to -100, we should get a 50% chance of moving left
    # where we would otherwise have a 100% chance of moving left
    params = {
        "inv_temp": -100,
        "lapse": 0
    }
    model = Model(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_
    )

    # we choose the highest filters so we don't actually filter anything out
    filter_pov_array = model.filter_fn(pov_array, {"depth": [7], "rank": [9], "value": [9]})
    p_left = model.get_p_left(params, filter_pov_array)
    assert np.allclose(p_left, 0.5)

@pytest.mark.parametrize("type_", ["R", "V", "T"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_path, value_max, value_sum, value_levelmean])
def test_conditional_inv_temp(type_,filter_fn, value_fn): 
    pov_array = get_asymmetric_board_pov(type_)

    effort_version = "policy_compress"
    # if we set the inv_temp to -100 for the middle condition, we should get a 50% chance of moving left
    # for the middle condition, where we would otherwise have a 100% chance of moving left
    # we should still have a 100% chance of moving left for the other conditions
    params = {
        "condition_inv_temp_0": 10,
        "condition_inv_temp_1": 10,
        "condition_inv_temp_2": -100,
        "condition_inv_temp_3": 10,
        "condition_inv_temp_4": 10,
        "lapse": 0
    }


    model = Model(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_
    )

    # we choose the highest filters so we don't actually filter anything out
    filter_pov_array = model.filter_fn(pov_array, {"depth": [7], "rank": [9], "value": [9]})
    p_left = model.get_p_left(params, filter_pov_array)
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[0]), 1)
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[1]), 1)
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[2]), 0.5)
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[3]), 1)
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[4]), 1)


@pytest.mark.parametrize("type_", ["R", "V", "T"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_path, value_max, value_sum, value_levelmean])
def test_conditional_inv_temp_does_not_affect_filter_adapt(type_,filter_fn, value_fn):
    pov_array = get_asymmetric_board_pov(type_)
    
    effort_version = "filter_adapt"
    params = {
        "inv_temp": 10,
        "lapse": 0,
        "condition_inv_temp_0": -100,
        "condition_inv_temp_1": -100,
        "condition_inv_temp_2": -100,
        "condition_inv_temp_3": -100,
        "condition_inv_temp_4": -100,
    }
    model = Model(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_
    )
    filter_pov_array = model.filter_fn(pov_array, {"depth": [7], "rank": [9], "value": [9]})
    p_left = model.get_p_left(params, filter_pov_array)
    assert np.allclose(p_left, 1)
    
    params = {
        "inv_temp": -100,
        "lapse": 0,
        "condition_inv_temp_0": 10,
        "condition_inv_temp_1": 10,
        "condition_inv_temp_2": 10,
        "condition_inv_temp_3": 10,
        "condition_inv_temp_4": 10,
    }
    p_left = model.get_p_left(params, filter_pov_array)
    assert np.allclose(p_left, 0.5)


@pytest.mark.parametrize("type_", ["R", "V", "T"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_path, value_max, value_sum, value_levelmean])
def test_inv_temp_does_not_affect_policy_compress(type_,filter_fn, value_fn):
    pov_array = get_asymmetric_board_pov(type_)
    
    effort_version = "policy_compress"
    # if we set the inv_temp to -100, we should get a 50% chance of moving left
    # where we would otherwise have a 100% chance of moving left
    params = {
        "inv_temp": -100,
        "lapse": 0,
        "condition_inv_temp_0": 10,
        "condition_inv_temp_1": 10,
        "condition_inv_temp_2": 10,
        "condition_inv_temp_3": 10,
        "condition_inv_temp_4": 10,
    }
    
    model = Model(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_
    )

    filter_pov_array = model.filter_fn(pov_array, {"depth": [7], "rank": [9], "value": [9]})
    p_left = model.get_p_left(params, filter_pov_array)
    assert np.allclose(p_left, 1)

    params = {
        "inv_temp": 10,
        "lapse": 0,
        "condition_inv_temp_0": -100,
        "condition_inv_temp_1": -100,
        "condition_inv_temp_2": -100,
        "condition_inv_temp_3": -100,
        "condition_inv_temp_4": -100,
    }
    p_left = model.get_p_left(params, filter_pov_array)
    assert np.allclose(p_left, 0.5)


@pytest.mark.parametrize("type_", ["R", "V", "T"])
@pytest.mark.parametrize("effort_version", ["filter_adapt", "policy_compress"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_path, value_max, value_sum, value_levelmean])
def test_asymmetric_NLL(type_,effort_version,filter_fn,value_fn): 
    pov_array = get_asymmetric_board_pov(type_)
    if effort_version == "filter_adapt":
        params_low_inv_temp = {
            "inv_temp": -100,
            "lapse": 0
        }
        params_high_inv_temp = {
            "inv_temp": 10,
            "lapse": 0
        }
    elif effort_version == "policy_compress":
        params_low_inv_temp = {
            "condition_inv_temp_0": -100,
            "condition_inv_temp_1": -100,
            "condition_inv_temp_2": -100,
            "condition_inv_temp_3": -100,
            "condition_inv_temp_4": -100,
            "lapse": 0
        }
        params_high_inv_temp = {
            "condition_inv_temp_0": 10,
            "condition_inv_temp_1": 10,
            "condition_inv_temp_2": 10,
            "condition_inv_temp_3": 10,
            "condition_inv_temp_4": 10,
            "lapse": 0
        }

    model = Model(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_
    )

    filter_pov_array = model.filter_fn(pov_array, {"depth": [7], "rank": [9], "value": [9]})
    choose_left = xr.DataArray(np.ones((5, 5)), dims = ["conditions", "games"])
    choose_left["conditions"] = get_stochasticity_levels(type_)
    nll_low = model.negative_log_likelihood(params_low_inv_temp.values(), params_low_inv_temp.keys(), filter_pov_array, choose_left)
    nll_high = model.negative_log_likelihood(params_high_inv_temp.values(), params_high_inv_temp.keys(), filter_pov_array, choose_left)
    assert nll_low > nll_high