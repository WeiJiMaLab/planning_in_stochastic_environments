import xarray as xr

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
