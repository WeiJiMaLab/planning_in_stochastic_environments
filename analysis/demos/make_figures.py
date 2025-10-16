import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir) 

from modeling import get_effort_filter_value_options
from modelchecking import trialwise_rewards, trialwise_greedydiff, trialwise_chooseleft
from analysis import Analyzer, lmm, glmm
from plots import set_helvetica_style
from utils import colormaps, report_p_value, strsimplify, get_stochasticity_levels, alphabet


def get_colormap(type_):
    return {"R": colormaps["arctic"], "T": colormaps["berry"], "V": colormaps["grass"]}[type_]


def total_rt_analysis(folder="raw", effort_version="policy_compress", filter_fn="filter_depth", value_fn="value_path", plot_fns=["greedydiff", "rewards", "rt"]):
    # Plot total RT analysis
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 5), gridspec_kw={'hspace': 0.5, 'wspace': 0.4})
    result_df, log_df, glmm_result_df = [], [], []

    # Plot total RT for each condition type
    for col, type_ in enumerate(["R", "V", "T"]):
        analyzer = Analyzer(f"{folder}.{effort_version}.{filter_fn}.{value_fn}", 
                          *get_effort_filter_value_options(type_),
                          type_, colors=get_colormap(type_), folders=[folder])
        
        # Plot total RT
        df_rt = analyzer.plot_stochasticity_vs_rt(ax=axs[col], yspace=np.linspace(2.8, 6.8, 6), first_rt=False)
        axs[col].set(xticklabels=[strsimplify(x) for x in axs[col].get_xticks()],
                    xlabel="Stochasticity Level (%)", 
                    ylabel="Total RT (s)")
        axs[col].text(-0.3, 1.15, alphabet(col), transform=axs[col].transAxes, 
                     fontsize=28, fontproperties=helvetica_bold, va='top', ha='left')

        # Run LMM analysis
        rt_result, rt_log = lmm(df_rt)
        rt_result.update({"Model Name": "empirical", "Stochasticity Type": type_, "Variable": "total_rt"})
        result_df.append(rt_result)

        # Log analysis details
        for log in rt_log:
            formula, status = log.split(":")
            condition = {"R": "Reliability", "V": "Volatility", "T": "Controllability"}[type_]
            log_df.append({
                "Model Name": "empirical",
                "Condition": condition,
                "Variable": "y = log(totalRT)",
                "Formula / Status": f"\\texttt{{{formula}:{status}}}"
            })

    fig.savefig(f"figures/empirical_total_rt.png", bbox_inches='tight', dpi=600)

    # Plot model checking analyses
    fig, axs = plt.subplots(len(plot_fns), 3, figsize=(15, 5 * len(plot_fns)), 
                           gridspec_kw={'hspace': 0.5, 'wspace': 0.4})
    
    for col, type_ in enumerate(["R", "V", "T"]):
        analyzer = Analyzer(f"{folder}.{effort_version}.{filter_fn}.{value_fn}",
                          *get_effort_filter_value_options(type_),
                          type_, colors=get_colormap(type_), folders=[folder])
        
        for row, plot_fn in enumerate(plot_fns):
            ax = axs[col] if len(plot_fns) == 1 else axs[row, col]
            
            if plot_fn == "greedydiff":
                df_value, _ = analyzer.plot_checking(trialwise_greedydiff, trialwise_chooseleft, 
                                                   n_bins=5, show_model=False, ax=ax)
                ax.set(xticks=[-6, -3, 0, 3, 6], xlabel="Label Difference\n(Left - Right)", 
                      ylabel="P(Choice = Left)", ylim=[0, 1])
                
                glmm_result, glmm_log = glmm(df_value)
                glmm_result.update({"Model Name": "empirical", "Stochasticity Type": type_, 
                                  "Variable": "greedydiff", "isGLMM": True})
                glmm_result_df.append(glmm_result)
                
            elif plot_fn == "rewards":
                df_reward, _ = analyzer.plot_checking_condition(trialwise_rewards, show_model=False, ax=ax)
                ax.set(ylabel="Points Earned", xlabel="Stochasticity Level (%)",
                      yticks=[4.8, 5.2, 5.6, 6, 6.4], 
                      xticks=np.array(get_stochasticity_levels(type_)) * 100,
                      xticklabels=[strsimplify(x) for x in ax.get_xticks()],
                      yticklabels=[strsimplify(y) for y in ax.get_yticks()])
                
                reward_result, reward_log = lmm(df_reward)
                reward_result.update({"Model Name": "empirical", "Stochasticity Type": type_, 
                                    "Variable": "points"})
                result_df.append(reward_result)
                
            elif plot_fn == "rt":
                df_rt = analyzer.plot_stochasticity_vs_rt(ax=ax, first_rt=True)
                ax.set(xticklabels=[strsimplify(x) for x in ax.get_xticks()],
                      xlabel="Stochasticity Level (%)", 
                      ylabel="First Choice RT (s)")
                
                rt_result, rt_log = lmm(df_rt)
                rt_result.update({"Model Name": "empirical", "Stochasticity Type": type_, 
                                   "Variable": "rt"})
                result_df.append(rt_result)

            # Add labels and titles
            ax.text(-0.3, 1.15, alphabet(row * 3 + col), transform=ax.transAxes,
                   fontsize=28, fontproperties=helvetica_bold, va='top', ha='left')
            if row == 0:
                title = {"R": "Reliability", "V": "Volatility", "T": "Controllability"}[type_]
                ax.set_title(title, color=analyzer.colors(0.5), pad=15)

    # Save results
    if log_df and result_df:
        save_dir = f"figures/{folder}/empirical"
        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame(log_df).to_csv(f"{save_dir}/model_log.csv", index=False)
        pd.DataFrame(result_df).to_csv(f"{save_dir}/model_result.csv", index=False)
        pd.DataFrame(glmm_result_df).to_csv(f"{save_dir}/glmm_result.csv", index=False)

    os.makedirs(f"figures/{folder}/", exist_ok=True)
    fig.savefig(f"figures/{folder}/empirical_first_rt.png", bbox_inches='tight', dpi=600)


def model_comparison_analysis(folders=["raw"],
                          effort_version = "policy_compress",
                          filter_fn="filter_depth", 
                          value_fn="value_path",
                          types=["R", "V", "T"],
                          full_range=(-1000, 20000),
                          zoom_range=(-100, 200),
                          kind="nll",
                          save_name=None):
    
    # Create figure with two panels per condition type
    fig, axes = plt.subplots(len(types), 2, 
                          figsize=(8, 12 * len(types)),
                          gridspec_kw={'width_ratios': [0.6, 0.4], 'wspace': 0.05},
                          constrained_layout=True)
    
    axes = [axes] if len(types) == 1 else axes

    for i, type_ in enumerate(types):
        # Initialize analyzer
        analyzer = Analyzer(f"{folders[0]}.{effort_version}.{filter_fn}.{value_fn}",
                        *get_effort_filter_value_options(type_),
                        type_, colors=get_colormap(type_), folders=folders)

        # Main panel showing full range
        ax_main = axes[i][0]
        analyzer.plot_model_comparison(ax=ax_main, kind=kind)
        ax_main.set_xlim(*full_range)
        ax_main.grid(True, axis='y')
        ax_main.text(-0.3, 1.15, alphabet(i), transform=ax_main.transAxes, 
                    fontsize=32, fontproperties=helvetica_bold, va='top', ha='left')

        # Zoomed panel
        ax_zoom = axes[i][1] 
        analyzer.plot_model_comparison(ax=ax_zoom, kind=kind)
        ax_zoom.set_xlim(*zoom_range)
        ax_zoom.set_xlabel("(Zoomed in)")
        ax_zoom.set_yticklabels([])
        ax_zoom.tick_params(axis='y', length=0)
        ax_zoom.grid(True, axis='y')

    # Save figure
    save_name = f"model_comparison_{kind}" if save_name is None else save_name
    save_dir = f"figures/{folders[0]}/{filter_fn}.{value_fn}"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/{save_name}.png", bbox_inches='tight', dpi=600)


def model_checking_analysis(folder="raw", 
                          effort_version="policy_compress",
                          filter_fn="filter_depth",
                          value_fn="value_path",
                          types=["R", "V", "T"],
                          plot_fns=["greedydiff", "rewards", "inv_temp"],
                          save_analysis=True,
                          save_name=None):
    """Generate model checking plots and analysis."""
    
    # Setup figure grid
    fig, axs = plt.subplots(len(plot_fns), len(types), 
                           figsize=(5 * len(types), 5 * len(plot_fns)), 
                           gridspec_kw={'hspace': 0.5, 'wspace': 0.4})
    
    log_df = []
    result_df = []

    # Plot each condition type in columns
    for col, type_ in enumerate(types):
        analyzer = Analyzer(f"{folder}.{effort_version}.{filter_fn}.{value_fn}",
                          *get_effort_filter_value_options(type_),
                          type_, colors=get_colormap(type_), 
                          folders=[folder])

        # Generate each plot type in rows
        for row, plot_fn in enumerate(plot_fns):
            ax = axs[col] if len(plot_fns) == 1 else axs[row, col]

            # Choice probability plot
            if plot_fn == "greedydiff":
                analyzer.plot_checking(trialwise_greedydiff, trialwise_chooseleft, 
                                    n_bins=5, show_model=True, ax=ax)
                ax.set(xticks=[-6, -3, 0, 3, 6], 
                      xlabel="Label Difference\n(Left - Right)",
                      ylabel="P(Choice = Left)", 
                      ylim=[0, 1])
            
            # Reward plot
            elif plot_fn == "rewards":
                analyzer.plot_checking_condition(trialwise_rewards, show_model=True, ax=ax)
                ax.set(ylabel="Points Earned",
                      xlabel="Stochasticity Level (%)",
                      yticks=[4.8, 5.2, 5.6, 6, 6.4],
                      xticks=np.array(get_stochasticity_levels(type_)) * 100)
                ax.set_xticklabels([strsimplify(x) for x in ax.get_xticks()])
                ax.set_yticklabels([strsimplify(y) for y in ax.get_yticks()])

            # Planning depth plot
            elif plot_fn == "depth":
                df_depth = analyzer.plot_stochasticity_vs_depth(ax=ax)
                ax.set(xlabel="Stochasticity Level (%)",
                      ylabel="Planning Depth")
                ax.set_xticklabels([strsimplify(x) for x in ax.get_xticks()])
                ax.set_yticklabels([strsimplify(y) for y in ax.get_yticks()])

                # Save depth analysis results
                model_result, model_log = lmm(df_depth)
                model_result.update({
                    "Model Name": analyzer.baseline_name.replace(".", "_"),
                    "Stochasticity Type": type_,
                    "Variable": "depth"
                })
                result_df.append(model_result)

                for log in model_log:
                    formula, status = log.split(":")
                    log_df.append({
                        "Model Name": analyzer.transform_name(analyzer.baseline_name),
                        "Condition": {"R": "Reliability", "V": "Volatility", "T": "Controllability"}[type_],
                        "Variable": "y = depth",
                        "Formula / Status": f"\\texttt{{{formula}:{status}}}"
                    })

            elif plot_fn == "inv_temp":
                df_invtemp = analyzer.plot_stochasticity_vs_conditional_inv_temp(ax=ax)
                ax.set(xlabel="Stochasticity Level (%)",
                      ylabel="Log $\\beta$")
                ax.set_xticklabels([strsimplify(x) for x in ax.get_xticks()])
                ax.set_yticklabels([strsimplify(y) for y in ax.get_yticks()])
                
                invtemp_result, invtemp_log = lmm(df_invtemp)
                invtemp_result.update({"Model Name": analyzer.transform_name(analyzer.baseline_name),
                                   "Stochasticity Type": type_, 
                                   "Variable": "invtemp"})
                result_df.append(invtemp_result)
                
                for log in invtemp_log:
                    formula, status = log.split(":")
                    log_df.append({
                        "Model Name": analyzer.transform_name(analyzer.baseline_name),
                        "Condition": {"R": "Reliability", "V": "Volatility", "T": "Controllability"}[type_],
                        "Variable": "y = invtemp",
                        "Formula / Status": f"\\texttt{{{formula}:{status}}}"
                    })

            # Add labels and titles
            ax.text(-0.3, 1.15, alphabet(row * len(types) + col), 
                   transform=ax.transAxes, fontsize=28, 
                   fontproperties=helvetica_bold, va='top', ha='left')
            
            if row == 0:
                if col == 0:
                    ax.text(-0.3, 1.5, f"Model: {analyzer.transform_name(analyzer.baseline_name)}", 
                           transform=ax.transAxes, fontsize=28,
                           fontproperties=helvetica_regular, va='top', ha='left')
                title = {"R": "Reliability", "V": "Volatility", "T": "Controllability"}[type_]
                ax.set_title(title, color=analyzer.colors(0.5), pad=15)

    # Save analysis results and figure
    save_dir = f"figures/{folder}/{filter_fn}.{value_fn}"
    os.makedirs(save_dir, exist_ok=True)

    if save_analysis and log_df and result_df:
        pd.DataFrame(log_df).to_csv(f"{save_dir}/model_log.csv", index=False)
        pd.DataFrame(result_df).to_csv(f"{save_dir}/model_result.csv", index=False)
    
    save_name = "grid" if save_name is None else save_name
    fig.savefig(f"{save_dir}/{save_name}.png", bbox_inches='tight', dpi=600)


def format_lmm_result(row):
    """Format LMM results into LaTeX string"""
    return (
        f"LMM, "
        f"$\\beta = {row['beta']:.2f}$, "
        f"$t_{{{int(row['dof'])}}} = {row['tstat']:.1f}$, "
        f"${report_p_value(row['pval'])}$"
    )

def format_glmm_main_effect(row):
    """Format GLMM main effect results into LaTeX string"""
    return (
        f"GLMM, "
        f"main effect $\\beta = {row['beta_main']:.2f}$, "
        f"$\\chi^2(1) = {row['chi2_main']:.1f}$, "
        f"${report_p_value(row['pval_main'])}$"
    )

def format_glmm_interaction(row):
    """Format GLMM interaction results into LaTeX string"""
    return (
        f"GLMM, "
        f"interaction $\\beta = {row['beta_inter']:.2f}$, "
        f"$\\chi^2(1) = {row['chi2_inter']:.1f}$, "
        f"${report_p_value(row['pval_inter'])}$"
    )

def write_latex_results(df_results, glmm_results, output_file):
    """Write formatted results to LaTeX file"""
    with open(output_file, 'w') as f:
        # Write LMM results
        for (model, stoch, var), row in df_results.iterrows():
            key = f"{model.replace(' ', '.')}.{stoch}.{var}"
            f.write(f"\\pgfkeyssetvalue{{{key}}}{{{format_lmm_result(row)}}}\n")

        # Write GLMM results
        for (model, stoch, var), row in glmm_results.iterrows():
            key = f"{model.replace(' ', '.')}.{stoch}.{var}"
            f.write(f"\\pgfkeyssetvalue{{{key}}}{{{format_glmm_main_effect(row)}}}\n")
            f.write(f"\\pgfkeyssetvalue{{{key + '.interaction'}}}{{{format_glmm_interaction(row)}}}\n")

def run_main_analysis(folder = "raw"): 
    total_rt_analysis()
    model_comparison_analysis()
    model_checking_analysis( 
            folder = folder,
            effort_version = "policy_compress", 
            filter_fn = "filter_depth", 
            value_fn = "value_path"
    )

    model_checking_analysis( 
            folder = folder,
            effort_version = "policy_compress", 
            filter_fn = "filter_depth", 
            value_fn = "value_levelmean"
    )

    model_checking_analysis( 
            folder = folder,
            effort_version = "policy_compress", 
            filter_fn = "filter_depth", 
            value_fn = "value_max"
    )

    model_checking_analysis( 
            folder = folder,
            effort_version = "policy_compress", 
            filter_fn = "filter_depth", 
            value_fn = "value_sum"
    )

    model_checking_analysis(
            folder = folder,
            effort_version = "policy_compress", 
            filter_fn = "filter_depth", 
            value_fn = "value_EV",
            types = ["R", "T"],
            plot_fns = ["greedydiff"]
    )

    # Process model logs into LaTeX table
    df_logs = pd.concat([pd.read_csv(f) for f in glob.glob(f"figures/{folder}/*/model_log.csv")])
    df_logs = df_logs.set_index(["Model Name", "Condition", "Variable", "Formula / Status"])
    df_logs.to_latex(f"figures/model_log.tex", index=True)
    
    # Replace \cline with \cmidrule for better formatting
    with open('figures/model_log.tex', 'r') as f:
        content = f.read().replace('\\cline', '\\cmidrule(lr)')
    with open('figures/model_log.tex', 'w') as f:
        f.write(content)

    # Process model results into LaTeX format
    # Regular LMM results
    df_results = pd.concat([
        pd.read_csv(f) for f in sorted(glob.glob(f"figures/{folder}/*/model_result.csv"))
    ])
    df_results = df_results.set_index(["Model Name", "Stochasticity Type", "Variable"])
    
    # GLMM results
    glmm_results = pd.concat([
        pd.read_csv(f) for f in sorted(glob.glob(f"figures/{folder}/*/glmm_result.csv"))
    ])
    glmm_results = glmm_results.set_index(["Model Name", "Stochasticity Type", "Variable"])

    # Write all results to LaTeX file
    write_latex_results(df_results, glmm_results, f"figures/model_result.tex")




if __name__ == "__main__":
    helvetica_regular, helvetica_bold = set_helvetica_style()
    os.makedirs(f"figures/raw", exist_ok=True)
    folder = "raw"
    model_comparison_analysis(folders = ["simulated_policy_compress"])



    


            

            

    








            





