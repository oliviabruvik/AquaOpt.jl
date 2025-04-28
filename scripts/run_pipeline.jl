include("../src/cleaning.jl")
include("../src/optimize_mdp.jl")
include("../src/optimize_sarsop.jl")
include("../src/plot_views.jl")
include("../src/utils.jl")

using .Cleaning, .PlotViews, .OptimizeMDP, .Utils, Plots
using DiscreteValueIteration
using POMDPs
ENV["PLOTS_BROWSER"] = "true"

# Clean data
# Data downloaded from https://lusedata.hubocean.earth/
# Data limited to production area 5: Stadt til Hustadvika for 2012-2025
df = Cleaning.load_and_clean("data/raw/licedata.csv")

# Plot sealice levels over time
sealice_levels_over_time_plot = PlotViews.plot_sealice_levels_over_time(df)
Plots.savefig(sealice_levels_over_time_plot, "results/figures/sealice_levels_over_time.png")

# Run optimizations
#sarsop_policy = Optimize.sarsop_optimize(df)
mdp_policy = OptimizeMDP.mdp_optimize(df)

# Evaluate policies
lambda_values = 0.0:0.05:1.0
num_episodes = 1000
steps_per_episode = 500
sarsop_results = evaluate_sarsop_policy(lambda_values, episodes=num_episodes, steps_per_episode=steps_per_episode, heuristic_policy=false)
heuristic_sarsop_results = evaluate_sarsop_policy(lambda_values, episodes=num_episodes, steps_per_episode=steps_per_episode, heuristic_policy=true)
mdp_results = OptimizeMDP.evaluate_mdp_policy(lambda_values, episodes=num_episodes, steps_per_episode=steps_per_episode, heuristic_policy=false)
heuristic_mdp_results = OptimizeMDP.evaluate_mdp_policy(lambda_values, episodes=num_episodes, steps_per_episode=steps_per_episode, heuristic_policy=true)

# Plot results
mdp_results_plot = PlotViews.plot_mdp_results(mdp_results, "MDP Policy")
heuristic_mdp_results_plot = PlotViews.plot_mdp_results(heuristic_mdp_results, "Heuristic Policy")
sarsop_results_plot = PlotViews.plot_mdp_results(sarsop_results, "SARSOP Policy")
heuristic_sarsop_results_plot = PlotViews.plot_mdp_results(heuristic_sarsop_results, "Heuristic SARSOP Policy")
Plots.savefig(mdp_results_plot, "results/figures/mdp_results_$(num_episodes)_$(steps_per_episode).png")
Plots.savefig(heuristic_mdp_results_plot, "results/figures/heuristic_mdp_results_$(num_episodes)_$(steps_per_episode).png")
Plots.savefig(sarsop_results_plot, "results/figures/sarsop_results_$(num_episodes)_$(steps_per_episode).png")
Plots.savefig(heuristic_sarsop_results_plot, "results/figures/heuristic_sarsop_results_$(num_episodes)_$(steps_per_episode).png")