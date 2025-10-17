import xarray as xr

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

