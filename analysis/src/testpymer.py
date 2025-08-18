from pymer4.models import lmer, glmer
from pymer4 import load_dataset
sleep = load_dataset('sleep')

def get_results(model, var = "Days"):
    print(model.convergence_status)
    if not "[1] TRUE" in model.convergence_status: 
        return None, "Model failed to converge"
    elif "singular" in " ".join(model.r_console).lower(): 
        return None, "Singular fit detected"
    elif "warning" in " ".join(model.r_console).lower(): 
        return None, "Convergence warnings"
    elif not model.fitted: 
        return None, "Model not properly fitted"

    result = model.result_fit.to_pandas().set_index("term").transpose()
    loglik = model.result_fit_stats.to_pandas()["logLik"]
    result_dict = {
        "Estimate": float(result[var]["estimate"]),
        "T-stat": float(result[var]["t_stat"]), 
        "DF": float(result[var]["df"]),
        "p_val": float(result[var]["p_value"]),
        "loglik": float(loglik.iloc[0])
    }
    return result_dict, "Converged Properly"

model = glmer('Reaction ~ Days + (1 | Subject)', data=sleep)
model.fit()
print(get_results(model, "Days"))


# mtcars = load_dataset('mtcars')
# model = lmer('mpg ~ 1 + (hp | cyl)', data=mtcars)
# model.set_factors('cyl')
# model.fit()
# print(get_results(model, "cyl"))