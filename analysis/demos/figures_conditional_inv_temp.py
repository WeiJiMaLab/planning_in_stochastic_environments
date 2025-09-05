import sys, os
import numpy as np
import matplotlib.pyplot as plt
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir)
from modeling import filter_depth, filter_rank, filter_value, value_path, value_EV, value_max, value_sum
from modelchecking import trialwise_rewards, trialwise_greedydiff, trialwise_chooseleft
from analysis import Analyzer, lmm
from plots import set_helvetica_style
from utils import colormaps, report_p_value, strsimplify, get_conditions, alphabet


def get_filter_and_value_functions(type_):
    filter_fns = [
        ["depth", filter_depth],
        ["rank", filter_rank], 
        ["value", filter_value]
    ]
    
    value_fns = [
        ["path", value_path],
        ["max", value_max],
        ["sum", value_sum]
    ]
    
    # add EV to value functions for non-volatility conditions
    if type_ != "V":
        value_fns.insert(1, ["EV", value_EV])
        
    return filter_fns, value_fns

if __name__ == "__main__":
    helvetica_regular, helvetica_bold = set_helvetica_style()
    
    # Setup output directory
    folder = "conditional_inv_temp"
    filter_fn = "filter_depth" 
    value_fn = "value_path"
    os.makedirs(f"figures/{folder}/{filter_fn}.{value_fn}", exist_ok=True)

    # Create figure layouts
    fig, ax = plt.subplots(3, 3, figsize=(15, 15), gridspec_kw={'hspace': 0.5, 'wspace': 0.4})
    fig2, ax2 = plt.subplots(3, 1, figsize=(4, 20), gridspec_kw={'hspace': 0.5})

    # Plot for each condition type
    for col, type_ in enumerate(["R", "V", "T"]):
        colormap = colormaps[{"R": "arctic", "T": "berry", "V": "grass"}[type_]]
        
        # Initialize analyzer
        analyzer = Analyzer(
            f"{folder}.filter_depth.value_path",
            *get_filter_and_value_functions(type_),
            type_,
            colors=colormap,
            folders=[folder],
            supplementary_models=[("main_fit", filter_depth, value_path)]
        )

        # Add labels to grid plots
        for row in range(3):
            ax[row, col].text(-0.3, 1.2, alphabet(row * 3 + col), 
                            transform=ax[row, col].transAxes,
                            fontsize=28, fontproperties=helvetica_bold, 
                            va='top', ha='left')

        # Plot checking analysis
        analyzer.plot_checking(trialwise_greedydiff, trialwise_chooseleft, 
                             n_bins=5, show_model=True, ax=ax[0, col])
        ax[0, col].set(xticks=[-6, -3, 0, 3, 6], 
                      xlabel="Label Difference\n(Left - Right)",
                      ylabel="P(Choice = Left)", 
                      ylim=[0, 1])

        ax[0, col].set_title("Reliability" if type_ == "R" else "Volatility" if type_ == "V" else "Controllability", color = analyzer.colors(0.5), pad = 15)

        # Plot condition checking
        analyzer.plot_checking_condition(trialwise_rewards, show_model=True, ax=ax[1, col])
        ax[1, col].set(ylabel="Points Earned",
                      xlabel="Stochasticity Level (%)", 
                      yticks=[4.8, 5.2, 5.6, 6, 6.4],
                      xticks=np.array(get_conditions(type_)) * 100)
        ax[1, col].set_xticklabels([strsimplify(x) for x in ax[1, col].get_xticks()])
        ax[1, col].set_yticklabels([strsimplify(y) for y in ax[1, col].get_yticks()])

        # Plot depth analysis
        df_depth = analyzer.plot_stochasticity_vs_depth(ax=ax[2, col])
        ax[2, col].set(xlabel="Stochasticity Level (%)",
                      ylabel="Planning Depth")
        ax[2, col].set_xticklabels([strsimplify(x) for x in ax[2, col].get_xticks()])
        ax[2, col].set_yticklabels([strsimplify(y) for y in ax[2, col].get_yticks()])

        # Plot model comparison
        analyzer.plot_model_comparison(format="violin", ax=ax2[col], 
                                    baseline_name="main_fit.filter_depth.value_path", kind = "nll")
        ax2[col].text(-0.3, 1.2, alphabet(col), transform=ax2[col].transAxes,
                     fontsize=28, fontproperties=helvetica_bold, va='top', ha='left')
        ax2[col].set_xlabel("Model NLL - NLL Main")

        # Save analysis results
        with open(f"figures/{folder}/{filter_fn}.{value_fn}/_{type_}_summary.txt", "w") as f:
            depth_result, depth_log = lmm(df_depth)
            f.write(f"LMM (Depth): {depth_result['modeltype']}\n{depth_log}\n\n")
            f.write(f"\\newcommand\\lmminvtempdepth{type_}{{\\textcolor{{red}}{{LMM, $\\beta = {depth_result['beta']:.2f}$, $t_{{{depth_result['dof']:.0f}}} = {depth_result['tstat']:.1f}$, ${report_p_value(depth_result['pval'])}$}}}}\n")

    # Save figures
    fig.savefig(f"figures/supp_conditional_inv_temp_grid.png", bbox_inches='tight')
    fig2.savefig(f"figures/supp_conditional_inv_temp_nll.png", bbox_inches='tight')
