import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir) 

from modeling import filter_depth, filter_rank, filter_value, value_path, value_EV, value_max, value_sum, value_levelmean
from modelchecking import trialwise_rewards, trialwise_greedydiff, trialwise_chooseleft
from analysis import Analyzer, lmm, glmm
from plots import set_helvetica_style
from utils import colormaps, report_p_value, strsimplify, get_conditions, alphabet

def get_colormap(type_):
    return {"R": colormaps["arctic"], "T": colormaps["berry"], "V": colormaps["grass"]}[type_]

def get_filter_and_value_functions(type_):
    # which filter functions to compare to
    compare_filter_fns = [
        ["depth", filter_depth],
        ["rank", filter_rank],
        ["value", filter_value],
    ]

    compare_value_fns = [
        ["path", value_path], 
        ["max", value_max],
        ["sum", value_sum], 
        ["level-mean", value_levelmean]
    ]

    if type_ == "R" or type_ == "T": 
        compare_value_fns.append(["EV", value_EV])
    return compare_filter_fns, compare_value_fns


def total_rt_analysis(folder = "main_fit", filter_fn = "filter_depth", value_fn = "value_path"):
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 5), gridspec_kw={'hspace': 0.5, 'wspace': 0.4})
    
    for col, type_ in enumerate(["R", "V", "T"]):
        ax = axs[col]
        analyzer = Analyzer(f"{folder}.{filter_fn}.{value_fn}", *get_filter_and_value_functions(type_), type_, colors=get_colormap(type_), folders=[folder])
        ax.text(-0.3, 1.15, alphabet(col), transform=ax.transAxes, fontsize=28, fontproperties=helvetica_bold, va='top', ha='left')
        
        df_rt = analyzer.plot_stochasticity_vs_rt(ax=ax, yspace=np.linspace(2.8, 6.8, 6), first_rt=False)
        ax.set(xticklabels=[strsimplify(x) for x in ax.get_xticks()], xlabel="Stochasticity Level (%)", ylabel="First Choice RT (s)")

        rt_result, rt_log = lmm(df_rt)
        with open(f"figures/{folder}/{filter_fn}.{value_fn}/_{type_}_total_rt_summary.txt", "w") as f:
            f.write(f"LMM (RT): {rt_result['modeltype']}\n{rt_log}\n\\newcommand\\supplmmrt{type_}{{\\textcolor{{red}}{{LMM, $\\beta = {rt_result['beta']:.2f}$, $t_{{{rt_result['dof']:.0f}}} = {rt_result['tstat']:.1f}$, ${report_p_value(rt_result['pval'])}$}}}}\n")
        
    fig.savefig(f"figures/{folder}/{filter_fn}.{value_fn}/total_rt.png", bbox_inches='tight', dpi=600)


def model_checking_analysis(folder = "main_fit", 
                            filter_fn = "filter_depth", 
                            value_fn = "value_path", 
                            types = ["R", "V", "T"], 
                            plot_fns = ["greedydiff", "rewards", "depth"],
                            save_analysis = True,
                            save_name = None):

    log_df = []
    result_df = []
    fig, axs = plt.subplots(len(plot_fns), len(types), figsize=(5 * len(types), 5 * len(plot_fns)), gridspec_kw={'hspace': 0.5, 'wspace': 0.4})

    for col, type_ in enumerate(types):
        analyzer = Analyzer(f"{folder}.{filter_fn}.{value_fn}", *get_filter_and_value_functions(type_), type_, colors=get_colormap(type_), folders=[folder])
        for row, plot_fn in enumerate(plot_fns):
            ax = axs[col] if len(plot_fns) == 1 else axs[row, col]

            if plot_fn == "greedydiff":
                analyzer.plot_checking(trialwise_greedydiff, trialwise_chooseleft, n_bins=5, show_model=True, ax=ax)
                ax.set(xticks=[-6, -3, 0, 3, 6], xlabel="Label Difference\n(Left - Right)", ylabel="P(Choice = Left)", ylim=[0, 1])
            
            if plot_fn == "rewards":
                analyzer.plot_checking_condition(trialwise_rewards, show_model=True, ax=ax)
                ax.set(ylabel="Points Earned", xlabel="Stochasticity Level (%)", yticks=[4.8, 5.2, 5.6, 6, 6.4], xticks=np.array(get_conditions(type_)) * 100, xticklabels=[strsimplify(x) for x in ax.get_xticks()], yticklabels=[strsimplify(y) for y in ax.get_yticks()])

            if plot_fn == "depth":
                df_depth = analyzer.plot_stochasticity_vs_depth(ax=ax)
                ax.set(xlabel="Stochasticity Level (%)", ylabel="Planning Depth", xticklabels=[strsimplify(x) for x in ax.get_xticks()], yticklabels=[strsimplify(y) for y in ax.get_yticks()])

                depth_result, depth_log = lmm(df_depth)
                result_df.append({"Model Name":analyzer.baseline_name.replace(".", "_"), "Stochasticity Type": type_, "Variable": "depth", "Beta": depth_result['beta'], "Dof": depth_result['dof'], "Tstat": depth_result['tstat'], "Pvalue": depth_result['pval']})

                for log in depth_log:
                    formula, status = log.split(":")
                    log_df.append({"Model Name": analyzer.transform_name(analyzer.baseline_name), "Stochasticity Type": type_, "Variable": "y = depth", "Formula": formula.replace('~', '\\sim'), "Convergence": status.replace("|", "\\mid")})

            ax.text(-0.3, 1.15, alphabet(row * len(types) + col), transform=ax.transAxes, fontsize=28, fontproperties=helvetica_bold, va='top', ha='left')
            if row == 0: 
                if col == 0:
                    ax.text(-0.3, 1.5, f"Model: {analyzer.transform_name(analyzer.baseline_name)}", transform=ax.transAxes, fontsize=28, fontproperties=helvetica_regular, va='top', ha='left')
                ax.set_title("Reliability" if type_ == "R" else "Volatility" if type_ == "V" else "Controllability", color = analyzer.colors(0.5), pad = 15)

    if save_analysis and len(log_df) > 0 and len(result_df) > 0:
        os.makedirs(f"figures/{filter_fn}.{value_fn}", exist_ok=True)
        pd.DataFrame(log_df).to_csv(f"figures/{filter_fn}.{value_fn}/depth_log.csv", index=False)
        pd.DataFrame(result_df).to_csv(f"figures/{filter_fn}.{value_fn}/depth_result.csv", index=False)
    
    if save_name is None:
        save_name = "grid"
    fig.savefig(f"figures/{folder}_{filter_fn}_{value_fn}_{save_name}.png", bbox_inches='tight', dpi=600)


import glob
if __name__ == "__main__":
    helvetica_regular, helvetica_bold = set_helvetica_style()
    folder = "main_fit"
    filter_fn = "filter_depth"
    value_fn = "value_path"

    # # total_rt_analysis(folder, filter_fn, value_fn)
    model_checking_analysis("main_fit", "filter_depth", "value_path")
    model_checking_analysis("main_fit", "filter_depth", "value_EV", types = ["R", "T"], plot_fns = ["greedydiff"], save_analysis = False)
    model_checking_analysis("main_fit", "filter_depth", "value_max")
    model_checking_analysis("main_fit", "filter_depth", "value_sum")
    model_checking_analysis("main_fit", "filter_depth", "value_levelmean", types = ["V", "T"], save_name = "vt_grid", save_analysis = False)
    model_checking_analysis("main_fit", "filter_depth", "value_levelmean")


    df_logs = []
    for filename in glob.glob(f"figures/*/depth_log.csv"):
        df_logs.append(pd.read_csv(filename))
    
    df_logs = pd.concat(df_logs)
    df_logs = df_logs.set_index(["Model Name", "Stochasticity Type", "Variable", "Formula"])
    df_logs.to_latex(f"figures/depth_log.tex", index=True)


    df_results = []
    for filename in glob.glob(f"figures/*/depth_result.csv"):
        df_results.append(pd.read_csv(filename))
    df_results = pd.concat(df_results)
    df_results = df_results.set_index(["Model Name", "Stochasticity Type", "Variable"])
    
    # Write to tex file with pgf format
    with open(f"figures/depth_result.tex", 'w') as f:
        for idx, row in df_results.iterrows():
            model, stoch, var = idx
            key = f"{model}.{stoch}.{var}"
            beta = row['Beta']
            dof = row['Dof']
            tstat = row['Tstat']
            pval = row['Pvalue']
            
            f.write(f"\\pgfkeyssetvalue{{{key}}}{{LMM, $\\beta = {beta:.2f}$, ")
            f.write(f"$t_{{{int(dof)}}} = {tstat:.1f}$, ")
            f.write(f"${report_p_value(pval)}$}}\n")








            





