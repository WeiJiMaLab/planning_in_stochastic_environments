from modeling import *
from modelchecking import *
from scipy import stats
import statsmodels.api as sm 
import statsmodels.formula.api as smf 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from pymer4.models import lmer, glmer
from scipy.stats import chi2
from utils import report_p_value
import polars as pl 


def get_results(model):
    model.fit()
    if not "[1] TRUE" in model.convergence_status: 
        return None, None, "Model failed to converge"
    elif "singular" in " ".join(model.r_console).lower(): 
        return None, None, "Singular fit detected"
    elif "warning" in " ".join(model.r_console).lower(): 
        return None, None, "Convergence warnings"
    elif not model.fitted: 
        return None, None, "Model not properly fitted"
    result = model.result_fit.to_pandas().set_index("term").transpose()
    loglik = model.result_fit_stats.to_pandas()["logLik"].iloc[0]
    return result, loglik, "Converged Properly"


def lmm(df):
    df = pl.from_pandas(df)
    error_log = ""
    for formula, label in [ 
        ("y ~ conditions + (1 + conditions|participants)", "FULL"), 
        ("y ~ conditions + (1 + conditions||participants)", "Uncorrelated Slopes"), 
        ("y ~ conditions + (1|participants)", "Intercept-only")]:

        model = lmer(formula, data = df)
        result, _, message = get_results(model)
        error_log += f"{formula}: {message}\n"
        if result is not None: 
            estimate = result["conditions"]["estimate"]
            tstat = result["conditions"]["t_stat"]
            dof = result["conditions"]["df"]
            pval = result["conditions"]["p_value"]
            return {
                "modeltype": label, 
                "beta": estimate, 
                "tstat": tstat, 
                "dof": dof, 
                "pval": pval
            }, error_log

    raise RuntimeError(message)

def glmm(df): 
    df = pl.from_pandas(df)
    for full, no_interaction, no_main, label in [ 
        (
            "y ~ x * conditions + (1 + x + conditions|participants)", 
            "y ~ x + conditions + (1 + x + conditions|participants)", 
            "y ~ conditions + (1 + conditions|participants)", 
            "FULL"
        ), 
        (
            "y ~ x * conditions + (1 + x + conditions||participants)", 
            "y ~ x + conditions + (1 + x + conditions||participants)", 
            "y ~ conditions + (1 + conditions||participants)", 
            "Uncorrelated Slopes"
        ), 
        (
            "y ~ x * conditions + (1|participants)", 
            "y ~ x + conditions + (1|participants)", 
            "y ~ conditions + (1|participants)", 
            "Intercept-only") 
        ]: 
        error_log = ""

        full_model = glmer(full, data = df, family = "binomial")
        no_interaction_model = glmer(no_interaction, data = df, family = "binomial")
        no_main_model = glmer(no_main, data = df, family = "binomial")

        full_result, full_loglik, full_message = get_results(full_model)
        no_interaction_result, no_interaction_loglik, no_interaction_message = get_results(no_interaction_model)
        no_main_result, no_main_loglik, no_main_message = get_results(no_main_model)

        if full_result is not None and no_interaction_result is not None and no_main_result is not None:
            error_log += f"{full}: {full_message}\n{no_interaction}: {no_interaction_message}\n{no_main}: {no_main_message}\n" 
            chi2_inter = 2 * (full_loglik - no_interaction_loglik)
            pval_inter = chi2.sf(chi2_inter, 1)

            chi2_main = 2 * (no_interaction_loglik - no_main_loglik)
            pval_main = chi2.sf(chi2_main, 1)

            beta_inter = full_result["x:conditions"]["estimate"]
            beta_main = no_interaction_result["x"]["estimate"]

            result = {
                "modeltype": label, 
                "beta_main": beta_main, 
                "chi2_main": chi2_main, 
                "pval_main": pval_main, 
                "beta_inter": beta_inter, 
                "chi2_inter": chi2_inter, 
                "pval_inter": pval_inter
            }
            
            return result, error_log
    raise RuntimeError(full_message, no_interaction_message, no_main_message)



def bootstrap(x, n = 1e4): 
    n_samps = len(x)
    samp_indices = np.random.choice(np.arange(n_samps), (int(n), n_samps), replace = True)
    return x[samp_indices]

class Analyzer(): 
    def __init__(self, baseline_name, filter_fns, value_fns, variant, colors, folders = ["fit"], verbose = False, supplementary_models = None):
        """
        Initialize the analysis object.

        Parameters:
        baseline_name (str): The name of the baseline model.
        filter_fns (list of tuples): A list of tuples where each tuple contains a filter function.
        value_fns (list of tuples): A list of tuples where each tuple contains a value function.
        variant (str): The variant of the data to be analyzed.
        verbose (bool, optional): If True, prints additional information during initialization. Default is False.

        Attributes:
        variant (str): The variant of the data being analyzed.
        data (dict): The data associated with the given variant.
        baseline_name (str): The name of the baseline model.
        conditions (list): The conditions associated with the given variant.
        model_data (defaultdict): A dictionary storing dataframes of model fits.
        model_fits (defaultdict): A dictionary storing tuples of model objects and their fits.
        baseline (DataFrame): The dataframe corresponding to the baseline model.
        reaction_time_data (DataFrame): A dataframe containing reaction time data for each player and game.
        """
        self.variant = variant
        self.data = get_data(variant)
        self.baseline_name = baseline_name
        self.conditions = get_conditions(self.variant)
        self.folders = folders

        self.model_data = defaultdict()
        self.model_fits = defaultdict()

        for folder in folders:
            for _, filter_fn in filter_fns: 
                for _, value_fn in value_fns: 
                    if verbose: print(variant, filter_fn.__name__, value_fn.__name__)
                    try:
                        fit = load_fit(variant, filter_fn, value_fn, folder = folder)
                    except FileNotFoundError:
                        print(f"Warning: file not found for variant {variant}, filter {filter_fn.__name__}, value {value_fn.__name__}, folder {folder}")
                        continue

                    df = fit_to_dataframe(fit)

                    # if the lapse rate is included, we need to account for it
                    if "condition_lapse_0" in df.columns:
                        # treat the lapse rate as if it were depth 0
                        for i, c in enumerate(get_conditions(self.variant)): 
                            lapse = df[f"condition_lapse_{i}"]
                            df[c] = df[c] * (1 - lapse)
                        df = df.drop(columns = [f"condition_lapse_{i}" for i in range(len(get_conditions(self.variant)))])

                    self.model_data[f"{folder}.{filter_fn.__name__}.{value_fn.__name__}"] = df
                    self.model_fits[f"{folder}.{filter_fn.__name__}.{value_fn.__name__}"] = (Model(filter_fn, value_fn, variant), fit)

        if supplementary_models is not None: 
            for folder, filter_fn, value_fn in supplementary_models:
                if verbose: print(folder, filter_fn.__name__, value_fn.__name__)
                try: 
                    fit = load_fit(variant, filter_fn, value_fn, folder = folder)
                except FileNotFoundError:
                    print(f"Warning: file not found for variant {variant}, filter {filter_fn.__name__}, value {value_fn.__name__}, folder {folder}")
                    continue
                df = fit_to_dataframe(fit)
                self.model_data[f"{folder}.{filter_fn.__name__}.{value_fn.__name__}"] = df
                self.model_fits[f"{folder}.{filter_fn.__name__}.{value_fn.__name__}"] = (Model(filter_fn, value_fn, variant), fit)
                self.folders.append(folder)


        self.colors = colors
        self.baseline = self.model_data[baseline_name]

        depths = {}
        for i, row in self.baseline.iterrows(): 
            depths[row.player] = {i: dict(row)[i] for i in self.conditions}

        reaction_time_data = []
        for player in self.data.keys(): 
            for game in format_games(self.data[player]["data"]):
                #to correct for an error where the number of trials was repeated twice
                reaction_time_data.append(pd.Series({
                    "player": player, 
                    "game": game["name"],
                    "log_first_rt": np.log(np.array([trial["rt"] for trial in game["trials"]])[0]/ 1000),
                    "log_total_rt": np.log((np.array([trial["rt"] for trial in game["trials"]])/ 1000).sum()),
                    "condition": game["p"],
                    "depth": depths[player][game["p"]]
                }))

        self.reaction_time_data = pd.DataFrame(reaction_time_data)

    def plot_model_comparison(self, n_bootstrap = 1e6, verbose = False, kind = "nll", format = "bar", ax=None, baseline_name=None): 
        '''
        baseline_name allows for a different baseline to be used for the model comparison. If None, the baseline_name is used.
        '''

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        print("N bootstrap", n_bootstrap)
        
        def transform_name(name): 
            folder, filter, value = name.split(".")
            filter = filter.split("_")[-1]
            value = value.split("_")[-1]
            if value == "ignoreuncertain": value = "ignore-uncertain"
            if len(self.folders) > 1: 
                return f"{filter} {value} ({folder})"
            else: 
                return f"{filter} {value}"

        plot = defaultdict(lambda: [])

        if baseline_name is None:
            baseline_name = self.baseline_name

        for key in tqdm.tqdm(self.model_data.keys()):
            if key == baseline_name: continue
            baseline = self.model_data[baseline_name]
            model = self.model_data[key]

            n_model_params = len(set(model.columns) - set(["aic", "bic", "nll", "player"]))
            n_baseline_params = len(set(baseline.columns) - set(["aic", "bic", "nll", "player"]))
            
            # 150 games times 7 choices per game
            n = 150 * 7
            model["aic"] = 2 * (n_model_params + model["nll"])
            model["bic"] = n_model_params * np.log(n) + model["nll"]

            baseline["aic"] = 2 * (n_baseline_params + baseline["nll"])
            baseline["bic"] = n_baseline_params * np.log(n) + baseline["nll"]


            # bootstrapped NLL model
            if kind == "nll":
                diff = np.array(model["nll"] - baseline["nll"])
            elif kind == "aic":
                diff = np.array(model["aic"] - baseline["aic"])
            elif kind == "bic":
                diff = np.array(model["bic"] - baseline["bic"])
            else:
                raise ValueError("Invalid kind: must be one of 'nll', 'aic', or 'bic'")

            # bootstrapped NLL model - NLL baseline
            bootstrap_diff = bootstrap(diff, n_bootstrap).sum(-1)

            mean = np.mean(bootstrap_diff)
            conf = np.quantile(bootstrap_diff, [0.025, 0.975])
            sig = (conf > 0).all() or (conf < 0).all()
            conf_centered = conf - mean 
            
            plot["mean"].append(mean)
            plot["conf"].append(conf_centered)
            plot["sig"].append(sig)
            plot["name"].append(transform_name(key))
            plot["dist"].append(bootstrap_diff)

            if verbose: print(key, conf)
        
        plot = Prodict(plot)

        # significant stars
        sig_star_positions = np.array(plot.conf)[:, -1] + np.array(plot.mean)
        sig_star_positions = np.max(sig_star_positions) * 0.1 + sig_star_positions
        
        sig_names = [plot.name[i] for i in range(len(plot.sig)) if plot.sig[i]]
        sig_star_positions = [sig_star_positions[i] for i in range(len(plot.sig)) if plot.sig[i]]

        plt.figure(figsize = (5, len(plot.name) * 0.5))
        if format == "violin":
            color = self.colors(0.5)
            violin = ax.violinplot(np.array(plot.dist).T,
                           showmeans=False,
                           showextrema = False,
                           vert = False,
                           widths = 0.8,
                           quantiles = [[0.025, 0.975]] * len(plot.name)
                          )

            for part in violin['bodies']:
                part.set_facecolor(color)
                part.set_alpha(0.4)

            for partname in (['cquantiles']):
                vp = violin[partname]
                vp.set_edgecolor(color)
                vp.set_linewidth(1.5)

            ax.set_yticks([y + 1 for y in range(len(plot.name))],
                  labels= plot.name)
            ax.set_ylim([0.2, len(plot.name) + 0.5])
            ax.vlines(0, -1, len(plot.name) + 1, colors=self.colors(0.5), alpha=0.3, linestyle='dotted')
            ax.grid(c = [0.95, 0.95, 0.95], axis = 'y', linewidth = 1)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_linewidth(1.5)
            ax.set_axisbelow(True)
            ax.xaxis.set_tick_params(width=1.5, length = 10)
            ax.yaxis.set_tick_params(width=1.5, length = 10)
            ax.set_xlabel(f"Model {kind.upper()} - {transform_name(self.baseline_name)} {kind.upper()}\n($\leftarrow$ better fit)")

        else:
            ax.barh(plot.name, plot.mean, align='center', color = "#a4c4eb", alpha=1)
            ax.errorbar(plot.mean, plot.name, xerr = np.abs(np.array(plot.conf)).T, ecolor = self.colors(0.5), fmt = "none", capsize = 3, elinewidth=2, markeredgewidth=2)
            # plt.scatter(sig_star_positions, sig_names, s = 100, color = 'k', marker = "*")

            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_linewidth(1.5)
            ax.set_axisbelow(True)
            ax.xaxis.set_tick_params(width=1.5, length = 10)
            ax.yaxis.set_tick_params(width=1.5, length = 10)

    def plot_stochasticity_vs_depth(self, ax=None): 
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            
        baseline = self.baseline
        
        mean = np.mean(baseline[self.conditions].values, axis = 0)
        std = np.std(baseline[self.conditions].values, axis = 0, ddof = 1)/np.sqrt(len(baseline))
        ax.errorbar(np.array(self.conditions) * 100, mean, std, capsize = 3, elinewidth=2, markeredgewidth=2, linewidth=2, color = self.colors(0.5))
        ax.set_xlabel("Stochasticity Level (%)\n")
        ax.set_ylabel("Planning Depth")
        ax.set_xticks(np.array(self.conditions) * 100)
            
        df = pd.melt(baseline, "player", value_vars = self.conditions)

        df["condition"] = df["variable"].astype(float)
        df["depth"] = df["value"]
        
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(1.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_tick_params(width=1.5, length = 10)
        ax.yaxis.set_tick_params(width=1.5, length = 10)
        ax.grid(c = [0.95, 0.95, 0.95], axis = 'both', linewidth = 1)

        out_df = df.copy()
        out_df = out_df.rename(columns = {"condition": "conditions", "depth": "y", "player": "participants"})

        # plt.savefig(f"figures/{self.variant}_stochasticity_vs_depth.svg", bbox_inches='tight')
        return out_df[["participants", "conditions", "y"]]

    def plot_stochasticity_vs_rt(self, yspace = np.linspace(0.8, 1.8, 6), ax=None, first_rt = True): 
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        self.reaction_time_data["x"] = self.reaction_time_data["condition"]
        if first_rt:
            self.reaction_time_data["y"] = self.reaction_time_data["log_first_rt"]
        else:
            self.reaction_time_data["y"] = self.reaction_time_data["log_total_rt"]
        group = self.reaction_time_data.groupby(["player", "condition"])
        mean_x, mean_rt, sem_rt = heirarchical_means(group)

        ax.errorbar(mean_x * 100, mean_rt, sem_rt, capsize = 3, elinewidth=2, markeredgewidth=2, linewidth=2, color = self.colors(0.5))
        ax.set_xticks(mean_x * 100)
        ax.set_xlabel("Stochasticity Level (%)\n")
        ax.set_ylabel("First Choice RT (s)" if first_rt else "Total RT (s)")

        ax.set_yticks(np.log(yspace))
        ax.set_yticklabels(np.round(yspace, 2))

        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(1.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_tick_params(width=1.5, length = 10)
        ax.yaxis.set_tick_params(width=1.5, length = 10)
        ax.grid(c = [0.95, 0.95, 0.95], axis = 'both', linewidth = 1)

        out_df = self.reaction_time_data.copy()
        out_df = out_df.rename(columns = {"condition": "conditions", "player": "participants"})
        return out_df[["participants", "conditions", "y"]]

    def plot_depth_vs_rt(self, yspace = np.linspace(0.8, 1.8, 6), ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            
        ax.scatter(self.reaction_time_data["depth"], self.reaction_time_data["log_first_rt"], alpha = 0.005, s = 30, color = self.colors(0.5))
        ax.axis("square")
        ax.set_xlabel("Planning Depth")
        ax.set_ylabel("Log First-choice RT")
        ax.set_xlim(-1, 8)
        
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(1.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_tick_params(width=1.5, length = 10)
        ax.yaxis.set_tick_params(width=1.5, length = 10)
        ax.grid(c = [0.95, 0.95, 0.95], axis = 'both', linewidth = 1)

        md = smf.mixedlm("log_first_rt ~ depth", self.reaction_time_data, groups = self.reaction_time_data["player"])
        mdf = md.fit()
        return mdf

    def plot_checking_condition(self, y_fun, show_model = True, ax=None): 
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            
        sim_x, sim_y = None, None

        model, fit_params = self.model_fits[self.baseline_name]
        
        if show_model:
            _, sim_y = simulate_model(data = self.data, 
                model = model, 
                fitted_params = fit_params,
                x_fun = trialwise_ignore,
                y_fun = y_fun
                )

            sim_x, participants = (xr.ones_like(sim_y) * sim_y.conditions), (xr.ones_like(sim_y) * sim_y.participants)
            model_x, model_y, model_sem = aggregate_data(participants.values.flatten(), sim_x.values.flatten(), sim_y.values.flatten())

            ax.fill_between(model_x * 100, model_y - model_sem, model_y + model_sem, alpha = 0.5, color = self.colors(0.5))

        _, y = empirical(
            data = self.data,
            x_fun = trialwise_ignore, 
            y_fun = y_fun
        )
        x, participants = (xr.ones_like(y) * y.conditions), (xr.ones_like(y) * y.participants)
        emp_x, emp_y, emp_sem = aggregate_data(participants.values.flatten(), x.values.flatten(), y.values.flatten())

        ax.errorbar(emp_x * 100 , emp_y, yerr=emp_sem, elinewidth=2, markeredgewidth=2, capsize = 3, linestyle='None', color=self.colors(0.5))
        if not show_model: ax.plot(emp_x * 100, emp_y, color=self.colors(0.5), linestyle='-', linewidth=2)  # Connect error bars with a line

        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(1.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_tick_params(width=1.5, length = 10)
        ax.yaxis.set_tick_params(width=1.5, length = 10)
        ax.grid(c = [0.95, 0.95, 0.95], axis = 'both', linewidth = 1)

        if show_model:
            df = y.to_dataframe(name="y").reset_index()
            df_sim = sim_y.to_dataframe(name="y").reset_index()
            return df, df_sim

    def plot_checking(self, x_fun, y_fun, n_bins = None, show_model = True, ax=None): 
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            
        sim_x, sim_y = None, None
        model, fit_params = self.model_fits[self.baseline_name]

        if show_model:
            sim_x, sim_y = simulate_model(data = self.data, 
                model = model, 
                fitted_params = fit_params,
                x_fun = x_fun, 
                y_fun = y_fun
                )

            model_x, model_y, model_sem = summary_statistics(sim_x, sim_y, n_bins = n_bins)
            num_model_keys = len(model_x.keys())
            for i, k in enumerate(model_x.keys()): 
                # Get color from colormap based on index
                color = self.colors(i / (num_model_keys - 1)) if num_model_keys > 1 else self.colors(0.5)
                ax.fill_between(model_x[k], model_y[k] - model_sem[k], model_y[k] + model_sem[k], alpha = 0.5, color = color)

        x, y = empirical(
            data = self.data,
            x_fun = x_fun, 
            y_fun = y_fun
        )

        emp_x, emp_y, emp_sem = summary_statistics(x, y, n_bins = n_bins)
        num_emp_keys = len(emp_x.keys())

        colorlegend = {}

        for i, k in enumerate(emp_x.keys()): 
            # Get color from colormap based on index
            color = self.colors(i / (num_emp_keys - 1)) if num_emp_keys > 1 else self.colors(0.5)
            ax.errorbar(emp_x[k], emp_y[k], yerr = emp_sem[k], elinewidth=2, markeredgewidth=2, capsize = 3, linestyle = 'None', label = k, color = color)
            if not show_model: ax.plot(emp_x[k], emp_y[k], color=color, linestyle='-', linewidth=2)  # Connect error bars with a line
            colorlegend[f"{strsimplify(k * 100)}%"] = color

        # After plotting, manually create a patch-based legend
        patches = [mpatches.Patch(color=color, label=label) for label, color in colorlegend.items()]

        ax.legend(handles=patches, loc='lower right', frameon=False, fontsize=13, title="$q$", title_fontsize = 13)

        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(1.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_tick_params(width=1.5, length = 10)
        ax.yaxis.set_tick_params(width=1.5, length = 10)
        ax.grid(c = [0.95, 0.95, 0.95], axis = 'both', linewidth = 1)

        if show_model:
            df = pd.concat([x.to_dataframe(name="x"), y.to_dataframe(name="y")], axis=1).reset_index()
            df_sim = pd.concat([sim_x.to_dataframe(name="x"), sim_y.to_dataframe(name="y")], axis=1).reset_index()
            return df, df_sim
