import sys, os
#this is a comment
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir) 

from modeling import *
from analysis import *
from plots import *
from utils import colormaps

folder = "final_fit"

for filter_fn in ["filter_depth"]:
    for value_fn in ["value_path", "value_sum", "value_max"]:
        for type_ in ["R", "V", "T"]:
            os.makedirs(f"figures/{folder}/{filter_fn}.{value_fn}", exist_ok=True)
            colormap_ = {"R": colormaps["arctic"], "T": colormaps["berry"], "V": colormaps["grass"]}
            colormap = colormap_[type_]

            filter_fns = [
                ["depth", filter_depth],
                ["rank", filter_rank],
                ["value", filter_value],
            ]

            if type_ == "V": 
                value_fns = [
                    ["path", value_path], 
                    ["max", value_max],
                    ["sum", value_sum]
                ]
            else: 
                value_fns = [
                    ["path", value_path], 
                    ["EV", value_EV],
                    ["max", value_max],
                    ["sum", value_sum]
                ]

            a = Analyzer(f"{folder}.{filter_fn}.{value_fn}", filter_fns, value_fns, type_, colors = colormap, folders = [folder])

            df_rt = a.plot_stochasticity_vs_rt()
            plt.gca().set_xticklabels([strsimplify(x) for x in plt.gca().get_xticks()]);
            plt.savefig(f"figures/{folder}/{filter_fn}.{value_fn}/{type_}_stochasticity_vs_rt.svg", transparent=True)

            a.plot_checking_condition(trialwise_rewards, show_model = False)
            plt.ylabel("Points Earned"); plt.xlabel("Stochasticity Level (%)"); plt.xticks(np.array(get_conditions(type_)) * 100);
            plt.yticks([4.8, 5.2, 5.6, 6, 6.4])
            plt.gca().set_yticklabels([strsimplify(y) for y in plt.gca().get_yticks()]);
            plt.gca().set_xticklabels([strsimplify(x) for x in plt.gca().get_xticks()]);
            plt.savefig(f"figures/{folder}/{filter_fn}.{value_fn}/{type_}_stochasticity_vs_reward.svg", transparent=True)

            a.plot_checking(trialwise_greedydiff, trialwise_chooseleft, n_bins = 5, show_model = False)
            plt.xlabel("Label Difference\n(Left - Right)"); plt.ylabel("P(Choice = Left)")
            plt.ylim([0, 1])
            plt.xticks([-6, -3, 0, 3, 6])
            plt.savefig(f"figures/{folder}/{filter_fn}.{value_fn}/{type_}_label_diff_vs_choice.svg", transparent=True)

            a.plot_model_comparison(format = "violin")
            plt.xscale("log")
            plt.xlim(10, 10**5)
            plt.gca().set_xticks((10 ** np.arange(1, 6)).tolist())
            plt.savefig(f"figures/{folder}/{filter_fn}.{value_fn}/{type_}_modelcomparison.pdf", bbox_inches="tight", transparent=True)

            df_depth = a.plot_stochasticity_vs_depth()
            plt.gca().set_yticklabels([strsimplify(y) for y in plt.gca().get_yticks()]);
            plt.gca().set_xticklabels([strsimplify(x) for x in plt.gca().get_xticks()]);
            plt.savefig(f"figures/{folder}/{filter_fn}.{value_fn}/{type_}_stochasticity_vs_depth.svg", transparent=True)

            df_reward, df_reward_sim = a.plot_checking_condition(trialwise_rewards, show_model = True)
            plt.ylabel("Points Earned"); plt.xlabel("Stochasticity Level (%)"); plt.xticks(np.array(get_conditions(type_)) * 100);
            plt.yticks([4.8, 5.2, 5.6, 6, 6.4])
            plt.gca().set_yticklabels([strsimplify(y) for y in plt.gca().get_yticks()]);
            plt.gca().set_xticklabels([strsimplify(x) for x in plt.gca().get_xticks()]);
            plt.savefig(f"figures/{folder}/{filter_fn}.{value_fn}/{type_}_stochasticity_vs_reward_model.svg", transparent=True)

            df_value, df_value_sim = a.plot_checking(trialwise_greedydiff, trialwise_chooseleft, n_bins = 5, show_model = True)
            plt.xlabel("Label Difference\n(Left - Right)"); plt.ylabel("P(Choice = Left)")
            plt.ylim([0, 1])
            plt.xticks([-6, -3, 0, 3, 6])
            plt.savefig(f"figures/{folder}/{filter_fn}.{value_fn}/{type_}_label_diff_vs_choice_model.svg", transparent=True)

            with open(f"figures/{folder}/{filter_fn}.{value_fn}/_{type_}_summary.txt", "w") as f:
                f.write(f"Stochasticity vs Reaction Time Regression Analysis\n")
                f.write(lmm(df_rt))

                f.write(f"Stochasticity vs Reward Regression Analysis LMM\n")
                f.write(lmm(df_reward))

                f.write(f"Greedy Value vs Choice LeftRegression Analysis GLMM\n")
                f.write(glmm(df_value))

                f.write(f"-----------------------------------\n")

                f.write(f"[Model] Stochasticity vs Depth Regression Analysis\n")
                f.write(lmm(df_depth))

                f.write(f"[Model] Stochasticity vs Reward Regression Analysis LMM\n")
                f.write(lmm(df_reward_sim))

                f.write(f"[Model] Greedy Value vs Choice LeftRegression Analysis GLMM\n")
                f.write(glmm(df_value_sim))


