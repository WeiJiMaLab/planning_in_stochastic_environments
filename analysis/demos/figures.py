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
    ax.set_xlabel("Stochasticity Level (%)")
    ax.set_ylabel("First Choice RT (s)")

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
            os.makedirs(f"figures/{folder}/{filter_fn}.{value_fn}", exist_ok=True)

            fig, ax = plt.subplots(3, 3, figsize=(15, 15), gridspec_kw={'hspace': 0.5, 'wspace': 0.4})
            fig2, ax2 = plt.subplots(3, 1, figsize=(4, 20), gridspec_kw={'hspace': 0.5})
            fig3, ax3 = plt.subplots(2, 3, figsize=(12.5, 10), gridspec_kw={'hspace': 0.5, 'wspace': 0.4})
            fig4, ax4 = plt.subplots(1, 3, figsize=(12.5, 5), gridspec_kw={'hspace': 0.5, 'wspace': 0.4})

            figs = [fig, fig2, fig3, fig4]
            axes = [ax, ax2, ax3, ax4]

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

                analyzer.plot_checking(trialwise_greedydiff, trialwise_chooseleft, n_bins=5, show_model=False, ax=ax[0, col])
                set_axis_labels(ax[0, col], type_, analyzer.colors(0.5))

                analyzer.plot_checking_condition(trialwise_rewards, show_model=False, ax=ax[1, col])
                set_condition_labels(ax[1, col], type_)

                df_rt = analyzer.plot_stochasticity_vs_rt(ax=ax[2, col])
                set_rt_labels(ax[2, col], type_)

                analyzer.plot_model_comparison(format="violin", ax=ax2[col])
                ax2[col].text(-0.3, 1.2, FIG_LABELS_VIOLIN[col], transform=ax2[col].transAxes,
                            fontsize=28, fontproperties=helvetica_bold, va='top', ha='left')
                ax2[col].set_xscale("log")
                ax2[col].set_xlim(10, 10**5)

                for row in range(2):
                    idx = row * 3 + col
                    ax3[row, col].text(-0.3, 1.2, FIG_LABELS_MODEL[idx], transform=ax3[row, col].transAxes,
                                    fontsize=28, fontproperties=helvetica_bold, va='top', ha='left')

                df_value, df_value_sim = analyzer.plot_checking(trialwise_greedydiff, trialwise_chooseleft, n_bins=5, show_model=True, ax=ax3[0, col])
                set_axis_labels(ax3[0, col], type_, analyzer.colors(0.5))

                df_reward, df_reward_sim = analyzer.plot_checking_condition(trialwise_rewards, show_model=True, ax=ax3[1, col])
                set_condition_labels(ax3[1, col], type_)

                ax4[col].text(-0.3, 1.2, FIG_LABELS_MODEL[col], transform=ax4[col].transAxes,
                            fontsize=28, fontproperties=helvetica_bold, va='top', ha='left')
                df_depth = analyzer.plot_stochasticity_vs_depth(ax=ax4[col])
                ax4[col].set_xticklabels([strsimplify(x) for x in ax4[col].get_xticks()])
                ax4[col].set_yticklabels([strsimplify(y) for y in ax4[col].get_yticks()])
                ax4[col].set_xlabel("Stochasticity Level (%)")
                ax4[col].set_ylabel("Planning Depth")

                with open(f"figures/{folder}/{filter_fn}.{value_fn}/_{type_}_summary.txt", "w") as f:
                    print("=" * 48)
                    print(f"Summary file: figures/{folder}/{filter_fn}.{value_fn}/_{type_}_summary.txt")

                    # GLMM
                    glmm_result, glmm_log = glmm(df_value)
                    f.write(f"GLMM: {glmm_result['modeltype']}\n{glmm_log}\n\n")

                    # LMM (Reward)
                    reward_result, reward_log = lmm(df_reward)
                    f.write(f"LMM (Reward): {reward_result['modeltype']}\n{reward_log}\n\n")

                    # LMM (RT)
                    rt_result, rt_log = lmm(df_rt)
                    f.write(f"LMM (RT): {rt_result['modeltype']}\n{rt_log}\n\n")

                    # LMM (Depth)
                    depth_result, depth_log = lmm(df_depth)
                    f.write(f"LMM (Depth): {depth_result['modeltype']}\n{depth_log}\n\n")

                    print("=" * 48)

                    # LaTeX commands
                    f.write(
                        f"\\newcommand\\glmm{type_}{{\\textcolor{{red}}{{GLMM, $\\beta = {glmm_result['beta_main']:.2f}$, $\\chi^2(1) = {glmm_result['chi2_main']:.1f}$, ${report_p_value(glmm_result['pval_main'])}$}}}}\n"
                    )
                    f.write(
                        f"\\newcommand\\glmminteraction{type_}{{\\textcolor{{red}}{{GLMM, interaction $\\beta = {glmm_result['beta_inter']:.2f}$, $\\chi^2(1) = {glmm_result['chi2_inter']:.1f}$, ${report_p_value(glmm_result['pval_inter'])}$}}}}\n"
                    )
                    f.write(
                        f"\\newcommand\\lmmreward{type_}{{\\textcolor{{red}}{{LMM, $\\beta = {reward_result['beta']:.2f}$, $t_{{{reward_result['dof']:.0f}}} = {reward_result['tstat']:.1f}$, ${report_p_value(reward_result['pval'])}$}}}}\n"
                    )
                    f.write(
                        f"\\newcommand\\lmmrt{type_}{{\\textcolor{{red}}{{LMM, $\\beta = {rt_result['beta']:.2f}$, $t_{{{rt_result['dof']:.0f}}} = {rt_result['tstat']:.1f}$, ${report_p_value(rt_result['pval'])}$}}}}\n"
                    )
                    f.write(
                        f"\\newcommand\\lmmdepth{type_}{{\\textcolor{{red}}{{LMM, $\\beta = {depth_result['beta']:.2f}$, $t_{{{depth_result['dof']:.0f}}} = {depth_result['tstat']:.1f}$, ${report_p_value(depth_result['pval'])}$}}}}\n"
                    )
                    print("=" * 48)

            #save figures
            fig.savefig(f"figures/{folder}/{filter_fn}.{value_fn}/grid.png", bbox_inches='tight')
            fig2.savefig(f"figures/{folder}/{filter_fn}.{value_fn}/violin.png", bbox_inches='tight')
            fig3.savefig(f"figures/{folder}/{filter_fn}.{value_fn}/model_checking.png", bbox_inches='tight')
            fig4.savefig(f"figures/{folder}/{filter_fn}.{value_fn}/depth.png", bbox_inches='tight')


            





