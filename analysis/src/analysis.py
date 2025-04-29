from modeling import *
from modelchecking import *
from scipy import stats
import statsmodels.api as sm 
import statsmodels.formula.api as smf 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from pymer4.models import Lmer
from scipy.stats import chi2
from utils import report_p_value


def lmm(df): 
    # combined = xr.Dataset({"y": y})
    # df = combined.stack(flat_dim=list(y.coords)).to_dataframe().reset_index(drop = True)

    model = Lmer("y ~ conditions + (1|participants)", data=df)
    result = model.fit()

    beta = model.coefs.loc['conditions', 'Estimate']
    t_val = model.coefs.loc['conditions', 'T-stat']
    dof = model.coefs.loc['conditions', 'DF']
    p_val = model.coefs.loc['conditions', 'P-val']
    return f"\\beta = {beta:.2f}, t_{dof:.2f} = {t_val:.2f}, {report_p_value(p_val)}\n"

def glmm(df): 
    # combined = xr.Dataset({"x": x, "y": y.astype(int)})
    # df = combined.stack().to_dataframe().reset_index(drop = True)

    # Create the full model with interaction
    full_model = Lmer("y ~ x * conditions + (1|participants)", data=df, family="binomial")
    out = full_model.fit(verbose = True);

    # Create reduced model without interaction
    reduced_model = Lmer("y ~ x + conditions + (1|participants)", data=df, family="binomial")
    reduced_model.fit();

    beta = out.loc['x:conditions', 'Estimate']
    chi_squared = 2 * (full_model.logLike - reduced_model.logLike)          
    dof = full_model.coefs.shape[0] - reduced_model.coefs.shape[0]
    p_value = chi2.sf(chi_squared, dof)

    return f"\\beta = {beta}, \\chi^2({dof}) = {chi_squared:.2f}, {report_p_value(p_value)}\n"

def bootstrap(x, n = 1e4): 
    n_samps = len(x)
    samp_indices = np.random.choice(np.arange(n_samps), (int(n), n_samps), replace = True)
    return x[samp_indices]

class Analyzer(): 
    def __init__(self, baseline_name, filter_fns, value_fns, variant, colors, folders = ["fit"], verbose = False):
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

        self.model_data = defaultdict()
        self.model_fits = defaultdict()

        for folder in folders:
            for f, filter_fn in filter_fns: 
                for v, value_fn in value_fns: 
                    if verbose: print(variant, f, v)  

                    try:
                        fit = load_fit(variant, filter_fn, value_fn, folder = folder)
                    except FileNotFoundError:
                        print(f"Warning: file not found for variant {variant}, filter {filter_fn.__name__}, value {value_fn.__name__}, folder {folder}")
                        continue

                    df = fit_to_dataframe(fit)
                    self.model_data[f"{folder}.{filter_fn.__name__}.{value_fn.__name__}"] = df
                    self.model_fits[f"{folder}.{filter_fn.__name__}.{value_fn.__name__}"] = (Model(filter_fn, value_fn, variant), fit)

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
                    "condition": game["p"],
                    "depth": depths[player][game["p"]]
                }))

        self.reaction_time_data = pd.DataFrame(reaction_time_data)

    def plot_model_comparison(self, n_bootstrap = 1e6, verbose = False, kind = "nll", format = "bar"): 
        print("N bootstrap", n_bootstrap)
        
        def transform_name(name): 
            folder, filter, value = name.split(".")
            filter = filter.split("_")[-1]
            value = value.split("_")[-1]
            if value == "ignoreuncertain": value = "ignore-uncertain"
            return f"{filter} {value}"

        plot = defaultdict(lambda: [])

        for key in tqdm.tqdm(self.model_data.keys()):
            if key == self.baseline_name: continue
            baseline = self.baseline
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
            plt.subplot(1, 1, 1)
            color = self.colors(0.5)
            violin = plt.violinplot(np.array(plot.dist).T, 
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
            
            plt.gca().set_yticks([y + 1 for y in range(len(plot.name))],
                  labels= plot.name)
            plt.ylim([0.2, len(plot.name) + 0.5])
            plt.vlines(0, -1, len(plot.name) + 1, colors=self.colors(0.5), alpha=0.3, linestyle='dotted')
            plt.gca().grid(c = [0.95, 0.95, 0.95], axis = 'y', linewidth = 1)
            gca = plt.gca()
            gca.spines[['top', 'right']].set_visible(False)
            gca.spines[['left', 'bottom']].set_linewidth(1.5)
            gca.set_axisbelow(True)
            gca.xaxis.set_tick_params(width=1.5, length = 10)
            gca.yaxis.set_tick_params(width=1.5, length = 10)
            gca.set_xlabel(f"Model {kind.upper()} - {transform_name(self.baseline_name)} {kind.upper()}\n($\leftarrow$ better fit)")

        else: 
            fig = plt.subplot(1, 1, 1)
            plt.barh(plot.name, plot.mean, align='center', color = "#a4c4eb", alpha=1)
            plt.errorbar(plot.mean, plot.name, xerr = np.abs(np.array(plot.conf)).T, ecolor = self.colors(0.5), fmt = "none", capsize = 3, elinewidth=2, markeredgewidth=2)
            # plt.scatter(sig_star_positions, sig_names, s = 100, color = 'k', marker = "*")
            
            gca = plt.gca()
            gca.spines[['top', 'right']].set_visible(False)
            gca.spines[['left', 'bottom']].set_linewidth(1.5)
            gca.set_axisbelow(True)
            gca.xaxis.set_tick_params(width=1.5, length = 10)
            gca.yaxis.set_tick_params(width=1.5, length = 10)


    def plot_stochasticity_vs_depth(self): 
        baseline = self.baseline
        
        fig, ax = plt.subplots(1, 1, figsize = (4, 4))
        mean = np.mean(baseline[self.conditions].values, axis = 0)
        std = np.std(baseline[self.conditions].values, axis = 0, ddof = 1)/np.sqrt(len(baseline))
        plt.errorbar(np.array(self.conditions) * 100, mean, std, capsize = 3, elinewidth=2, markeredgewidth=2, linewidth=2, color = self.colors(0.5))
        plt.xlabel("Stochasticity Level (%)\n")
        plt.ylabel("Planning Depth")
        plt.xticks(np.array(self.conditions) * 100);
            
        df = pd.melt(baseline, "player", value_vars = self.conditions)

        df["condition"] = df["variable"].astype(float)
        df["depth"] = df["value"]
        
        gca = plt.gca()
        gca.spines[['top', 'right']].set_visible(False)
        gca.spines[['left', 'bottom']].set_linewidth(1.5)
        gca.set_axisbelow(True)
        gca.xaxis.set_tick_params(width=1.5, length = 10)
        gca.yaxis.set_tick_params(width=1.5, length = 10)
        gca.grid(c = [0.95, 0.95, 0.95], axis = 'both', linewidth = 1)

        out_df = df.copy()
        out_df = out_df.rename(columns = {"condition": "conditions", "depth": "y", "player": "participants"})

        # plt.savefig(f"figures/{self.variant}_stochasticity_vs_depth.svg", bbox_inches='tight')
        return out_df[["participants", "conditions", "y"]]

    def plot_stochasticity_vs_rt(self, yspace = np.linspace(0.8, 1.8, 6)): 

        self.reaction_time_data["x"] = self.reaction_time_data["condition"]
        self.reaction_time_data["y"] = self.reaction_time_data["log_first_rt"]
        group = self.reaction_time_data.groupby(["player", "condition"])
        mean_x, mean_rt, sem_rt = heirarchical_means(group)

        fig, ax = plt.subplots(1, 1, figsize = (4, 4))
        plt.errorbar(mean_x * 100, mean_rt, sem_rt, capsize = 3, elinewidth=2, markeredgewidth=2, linewidth=2, color = self.colors(0.5))
        plt.xticks(mean_x * 100)
        plt.xlabel("Stochasticity Level (%)\n")
        plt.ylabel("First Choice RT (s)")

        plt.yticks(np.log(yspace))
        ax.set_yticklabels(np.round(yspace, 2))

        gca = plt.gca()
        gca.spines[['top', 'right']].set_visible(False)
        gca.spines[['left', 'bottom']].set_linewidth(1.5)
        gca.set_axisbelow(True)
        gca.xaxis.set_tick_params(width=1.5, length = 10)
        gca.yaxis.set_tick_params(width=1.5, length = 10)
        gca.grid(c = [0.95, 0.95, 0.95], axis = 'both', linewidth = 1)

        out_df = self.reaction_time_data.copy()
        out_df = out_df.rename(columns = {"condition": "conditions", "log_first_rt": "y", "player": "participants"})
        return out_df[["participants", "conditions", "y"]]

    def plot_depth_vs_rt(self, yspace = np.linspace(0.8, 1.8, 6)):
        # print("hello")
        fig, ax = plt.subplots(1, 1, figsize = (4, 4))
        plt.scatter(self.reaction_time_data["depth"], self.reaction_time_data["log_first_rt"], alpha = 0.005, s = 30, color = self.colors(0.5))
        plt.axis("square")
        plt.xlabel("Planning Depth")
        plt.ylabel("Log First-choice RT")
        plt.xlim(-1, 8)
        gca = plt.gca()
        gca.spines[['top', 'right']].set_visible(False)
        gca.spines[['left', 'bottom']].set_linewidth(1.5)
        gca.set_axisbelow(True)
        gca.xaxis.set_tick_params(width=1.5, length = 10)
        gca.yaxis.set_tick_params(width=1.5, length = 10)
        gca.grid(c = [0.95, 0.95, 0.95], axis = 'both', linewidth = 1)
        # plt.savefig(f"figures/{self.variant}_depth_vs_rt.svg", bbox_inches='tight')

        md = smf.mixedlm("log_first_rt ~ depth", self.reaction_time_data, groups = self.reaction_time_data["player"])
        mdf = md.fit()
        # print("Group Std", np.sqrt(mdf.cov_re.values[0]))
        return mdf

    # def plot_trial_vs_rt(self, yspace = np.linspace(0.8, 1.8, 6)): 
    #     x, y = empirical(
    #         data = self.data,
    #         x_fun = trialwise_trialnumber, 
    #         y_fun = trialwise_reactiontime
    #     )

    #     plt.figure(figsize = (4, 4))

    #     emp_x, emp_y, emp_sem = summary_statistics(x, y, n_bins = None)
    #     num_keys = len(emp_x.keys())
    #     for i, k in enumerate(emp_x.keys()): 
    #         # Get color from colormap based on index
    #         color = self.colors(i / (num_keys - 1)) if num_keys > 1 else self.colors(0.5)
    #         plt.errorbar(emp_x[k], emp_y[k], yerr = emp_sem[k], elinewidth=2, markeredgewidth=2, linewidth = 3, capsize = 3, color = color)
        
    #     plt.xlabel("Trial")
    #     plt.ylabel("Log First-choice RT")

    #     gca = plt.gca()
    #     gca.spines[['top', 'right']].set_visible(False)
    #     gca.spines[['left', 'bottom']].set_linewidth(1.5)
    #     gca.set_axisbelow(True)
    #     gca.xaxis.set_tick_params(width=1.5, length = 10)
    #     gca.yaxis.set_tick_params(width=1.5, length = 10)
    #     gca.grid(c = [0.95, 0.95, 0.95], axis = 'both', linewidth = 1)
        
    #     # plt.savefig(f"figures/{self.variant}_trialvsrt.svg", bbox_inches='tight')

    def plot_checking_condition(self, y_fun, show_model = True): 
        sim_x, sim_y = None, None

        model, fit_params = self.model_fits[self.baseline_name]
        plt.figure(figsize = (4, 4))
        
        if show_model:
            _, sim_y = simulate_model(data = self.data, 
                model = model, 
                fitted_params = fit_params,
                x_fun = trialwise_ignore,
                y_fun = y_fun
                )

            sim_x, participants = (xr.ones_like(sim_y) * sim_y.conditions), (xr.ones_like(sim_y) * sim_y.participants)
            model_x, model_y, model_sem = aggregate_data(participants.values.flatten(), sim_x.values.flatten(), sim_y.values.flatten())

            plt.fill_between(model_x * 100, model_y - model_sem, model_y + model_sem, alpha = 0.5, color = self.colors(0.5))

        _, y = empirical(
            data = self.data,
            x_fun = trialwise_ignore, 
            y_fun = y_fun
        )
        x, participants = (xr.ones_like(y) * y.conditions), (xr.ones_like(y) * y.participants)
        emp_x, emp_y, emp_sem = aggregate_data(participants.values.flatten(), x.values.flatten(), y.values.flatten())

        plt.errorbar(emp_x * 100 , emp_y, yerr=emp_sem, elinewidth=2, markeredgewidth=2, capsize = 3, linestyle='None', color=self.colors(0.5))
        if not show_model: plt.plot(emp_x * 100, emp_y, color=self.colors(0.5), linestyle='-', linewidth=2)  # Connect error bars with a line

        gca = plt.gca()
        gca.spines[['top', 'right']].set_visible(False)
        gca.spines[['left', 'bottom']].set_linewidth(1.5)
        gca.set_axisbelow(True)
        gca.xaxis.set_tick_params(width=1.5, length = 10)
        gca.yaxis.set_tick_params(width=1.5, length = 10)
        gca.grid(c = [0.95, 0.95, 0.95], axis = 'both', linewidth = 1)

        if show_model:
            df = y.to_dataframe(name="y").reset_index()
            df_sim = sim_y.to_dataframe(name="y").reset_index()
            return df, df_sim
        

    def plot_checking(self, x_fun, y_fun, n_bins = None, show_model = True): 
        sim_x, sim_y = None, None
        model, fit_params = self.model_fits[self.baseline_name]

        plt.figure(figsize = (4, 4))

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
                plt.fill_between(model_x[k], model_y[k] - model_sem[k], model_y[k] + model_sem[k], alpha = 0.5, color = color)

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
            plt.errorbar(emp_x[k], emp_y[k], yerr = emp_sem[k], elinewidth=2, markeredgewidth=2, capsize = 3, linestyle = 'None', label = k, color = color)
            if not show_model: plt.plot(emp_x[k], emp_y[k], color=color, linestyle='-', linewidth=2)  # Connect error bars with a line
            colorlegend[f"{strsimplify(k * 100)}%"] = color

        # After plotting, manually create a patch-based legend
        patches = [mpatches.Patch(color=color, label=label) for label, color in colorlegend.items()]

        plt.legend(handles=patches, loc='lower right', frameon=False, fontsize=13, title="$q$", title_fontsize = 13)

        gca = plt.gca()
        gca.spines[['top', 'right']].set_visible(False)
        gca.spines[['left', 'bottom']].set_linewidth(1.5)
        gca.set_axisbelow(True)
        gca.xaxis.set_tick_params(width=1.5, length = 10)
        gca.yaxis.set_tick_params(width=1.5, length = 10)
        gca.grid(c = [0.95, 0.95, 0.95], axis = 'both', linewidth = 1)

        if show_model:
            df = pd.concat([x.to_dataframe(name="x"), y.to_dataframe(name="y")], axis=1).reset_index()
            df_sim = pd.concat([sim_x.to_dataframe(name="x"), sim_y.to_dataframe(name="y")], axis=1).reset_index()
            return df, df_sim
                

