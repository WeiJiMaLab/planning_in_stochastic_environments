import sys, os
import numpy as np
import matplotlib.pyplot as plt
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir) 

from modeling import filter_depth, filter_rank, filter_value, value_path, value_EV, value_max, value_sum
from modelchecking import trialwise_rewards, trialwise_greedydiff, trialwise_chooseleft
from analysis import Analyzer, lmm, glmm
from plots import set_helvetica_style
from utils import colormaps, report_p_value, strsimplify, get_conditions


def set_axis_labels(ax, type_, color):
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_xlabel("Label Difference\n(Left - Right)")
    ax.set_ylabel("P(Choice = Left)")
    ax.set_ylim([0, 1])
    ax.set_title(
        "Reliability" if type_ == "R" else "Volatility" if type_ == "V" else "Controllability",
        color=color, pad=15
    )

def set_condition_labels(ax, type_):
    ax.set_ylabel("Points Earned")
    ax.set_xlabel("Stochasticity Level (%)")
    ax.set_yticks([4.8, 5.2, 5.6, 6, 6.4])
    ax.set_xticks(np.array(get_conditions(type_)) * 100)
    ax.set_xticklabels([strsimplify(x) for x in ax.get_xticks()])
    ax.set_yticklabels([strsimplify(y) for y in ax.get_yticks()])

def set_rt_labels(ax, type_):
    ax.set_xticklabels([strsimplify(x) for x in ax.get_xticks()])

FIG_LABELS_GRID = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
FIG_LABELS_VIOLIN = ['A', 'B', 'C']
FIG_LABELS_MODEL = ['A', 'B', 'C', 'D', 'E', 'F']

def get_filter_and_value_functions(type_):
    # which filter functions to compare to
    compare_filter_fns = [
        ["depth", filter_depth],
        ["rank", filter_rank],
        ["value", filter_value],
    ]
    # exclude EV from volatility condition
    if type_ == "V": 
        compare_value_fns = [
            ["path", value_path], 
            ["max", value_max],
            ["sum", value_sum]
        ]
    else: 
        compare_value_fns = [
            ["path", value_path], 
            ["EV", value_EV],
            ["max", value_max],
            ["sum", value_sum]
        ]
    return compare_filter_fns, compare_value_fns

    

if __name__ == "__main__":
    helvetica_regular, helvetica_bold = set_helvetica_style()
    folder = "main_fit"
    for filter_fn in ["filter_depth"]:
        for value_fn in ["value_path"]:
            os.makedirs(f"figures/{folder}/supplementary/{filter_fn}.{value_fn}", exist_ok=True)
            fig, ax = plt.subplots(1, 3, figsize=(12.5, 5), gridspec_kw={'hspace': 0.5, 'wspace': 0.4})

            for col, type_ in enumerate(["R", "V", "T"]):
                colormap_ = {"R": colormaps["arctic"], "T": colormaps["berry"], "V": colormaps["grass"]}
                colormap = colormap_[type_]

                compare_filter_fns, compare_value_fns = get_filter_and_value_functions(type_)
                
                # load data into analyzer with baseline filter_depth.value_path
                analyzer = Analyzer(
                    f"{folder}.{filter_fn}.{value_fn}",
                    compare_filter_fns,
                    compare_value_fns,
                    type_,
                    colors=colormap,
                    folders=[folder]
                )

                ax[0].text(-0.3, 1.15, 'A', transform=ax[0].transAxes,
                                    fontsize=28, fontproperties=helvetica_bold, va='top', ha='left')
                ax[1].text(-0.3, 1.15, 'B', transform=ax[1].transAxes,
                                    fontsize=28, fontproperties=helvetica_bold, va='top', ha='left')
                ax[2].text(-0.3, 1.15, 'C', transform=ax[2].transAxes,
                                    fontsize=28, fontproperties=helvetica_bold, va='top', ha='left')

                df_rt = analyzer.plot_stochasticity_vs_rt(ax=ax[col], yspace=np.linspace(2.8, 6.8, 6), first_rt=False)
                set_rt_labels(ax[col], type_)

                with open(f"figures/{folder}/supplementary/{filter_fn}.{value_fn}/_{type_}_supp_summary.txt", "w") as f:
                    # LMM (RT)
                    rt_result, rt_log = lmm(df_rt)
                    f.write(f"LMM (RT): {rt_result['modeltype']}\n{rt_log}\n")
                    # LaTeX commands
                    f.write(
                        f"\\newcommand\\supplmmrt{type_}{{\\textcolor{{red}}{{LMM, $\\beta = {rt_result['beta']:.2f}$, $t_{{{rt_result['dof']:.0f}}} = {rt_result['tstat']:.1f}$, ${report_p_value(rt_result['pval'])}$}}}}\n"
                    )
            #save figures
            fig.savefig(f"figures/{folder}/supplementary/{filter_fn}.{value_fn}/supp_rt.png", bbox_inches='tight', dpi=600)
        
        for value_fn in ["value_EV"]:
            os.makedirs(f"figures/{folder}/supplementary/{filter_fn}.{value_fn}", exist_ok=True)

            fig, ax = plt.subplots(1, 2, figsize=(12.5/3 * 2, 5), gridspec_kw={'hspace': 0.5, 'wspace': 0.4})

            ax[0].text(-0.3, 1.15, 'A', transform=ax[0].transAxes,
                                    fontsize=28, fontproperties=helvetica_bold, va='top', ha='left')
            ax[1].text(-0.3, 1.15, 'B', transform=ax[1].transAxes,
                                    fontsize=28, fontproperties=helvetica_bold, va='top', ha='left')

            for col, type_ in enumerate(["R", "T"]):
                colormap_ = {"R": colormaps["arctic"], "T": colormaps["berry"], "V": colormaps["grass"]}
                colormap = colormap_[type_]

                compare_filter_fns, compare_value_fns = get_filter_and_value_functions(type_)
                
                # load data into analyzer with baseline filter_depth.value_path
                analyzer = Analyzer(
                    f"{folder}.{filter_fn}.{value_fn}",
                    compare_filter_fns,
                    compare_value_fns,
                    type_,
                    colors=colormap,
                    folders=[folder]
                )

                df_value, df_value_sim = analyzer.plot_checking(trialwise_greedydiff, trialwise_chooseleft, n_bins=5, show_model=True, ax=ax[col])
                set_axis_labels(ax[col], type_, analyzer.colors(0.5))
            fig.savefig(f"figures/{folder}/supplementary/{filter_fn}.{value_fn}/supp_{value_fn}_model_checking.png", bbox_inches='tight', dpi=600)

        for value_fn in ["value_max", "value_sum"]:
            os.makedirs(f"figures/{folder}/supplementary/{filter_fn}.{value_fn}", exist_ok=True)

            fig, ax = plt.subplots(3, 3, figsize=(12.5, 15), gridspec_kw={'hspace': 0.5, 'wspace': 0.4})

            for col, type_ in enumerate(["R", "V", "T"]):
                colormap_ = {"R": colormaps["arctic"], "T": colormaps["berry"], "V": colormaps["grass"]}
                colormap = colormap_[type_]

                compare_filter_fns, compare_value_fns = get_filter_and_value_functions(type_)
                
                # load data into analyzer with baseline filter_depth.value_path
                analyzer = Analyzer(
                    f"{folder}.{filter_fn}.{value_fn}",
                    compare_filter_fns,
                    compare_value_fns,
                    type_,
                    colors=colormap,
                    folders=[folder]
                )

                for row in range(3):
                    idx = row * 3 + col
                    ax[row, col].text(-0.3, 1.2, FIG_LABELS_GRID[idx], transform=ax[row, col].transAxes,
                                    fontsize=28, fontproperties=helvetica_bold, va='top', ha='left')

                df_value, df_value_sim = analyzer.plot_checking(trialwise_greedydiff, trialwise_chooseleft, n_bins=5, show_model=True, ax=ax[0, col])
                set_axis_labels(ax[0, col], type_, analyzer.colors(0.5))

                df_reward, df_reward_sim = analyzer.plot_checking_condition(trialwise_rewards, show_model=True, ax=ax[1, col])
                set_condition_labels(ax[1, col], type_)
                
                df_depth = analyzer.plot_stochasticity_vs_depth(ax=ax[2, col])
                ax[2, col].set_xticklabels([strsimplify(x) for x in ax[2, col].get_xticks()])
                ax[2, col].set_yticklabels([strsimplify(y) for y in ax[2, col].get_yticks()])
                ax[2, col].set_xlabel("Stochasticity Level (%)")
                ax[2, col].set_ylabel("Planning Depth")

                with open(f"figures/{folder}/supplementary/{filter_fn}.{value_fn}/_{type_}_supp_summary.txt", "w") as f:
                    print("=" * 48)
                    print(f"Summary file: figures/{folder}/supplementary/{filter_fn}.{value_fn}/_{type_}_supp_summary.txt")

                    # LMM (Depth)
                    depth_result, depth_log = lmm(df_depth)
                    f.write(f"LMM (Depth): {depth_result['modeltype']}\n{depth_log}\n\n")
                    print("=" * 48)
                    f.write(
                        f"\\newcommand\\{value_fn.replace('_', '')}lmmdepth{type_}{{\\textcolor{{red}}{{LMM, $\\beta = {depth_result['beta']:.2f}$, $t_{{{depth_result['dof']:.0f}}} = {depth_result['tstat']:.1f}$, ${report_p_value(depth_result['pval'])}$}}}}\n"
                    )
                    print("=" * 48)
            fig.savefig(f"figures/{folder}/supplementary/{filter_fn}.{value_fn}/supp_{value_fn}_model_checking.png", bbox_inches='tight', dpi=600)




            





