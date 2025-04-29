include("../src/cleaning.jl")
include("../src/optimize_mdp.jl")
include("../src/optimize_sarsop.jl")
include("../src/plot_views.jl")
include("../src/utils.jl")

# Import required packages
using DiscreteValueIteration
using POMDPs
using Plots
using NativeSARSOP: SARSOPSolver

ENV["PLOTS_BROWSER"] = "true"

# Load and clean data
df = load_and_clean("data/raw/licedata.csv")

# Plot sealice levels over time
sealice_levels_over_time_plot = plot_sealice_levels_over_time(df)
savefig(sealice_levels_over_time_plot, "results/figures/sealice_levels_over_time.png")

# Evaluate policies
lambda_values = 0.0:0.2:1.0 # 0.0:0.05:1.0
num_episodes = 10 # 1000
steps_per_episode = 50 # 500

# Create heuristic policies
heuristic_policies_dict = create_heuristic_policy_dict(lambda_values)
heuristic_results = calculate_avg_rewards(heuristic_policies_dict, episodes=num_episodes, steps_per_episode=steps_per_episode)
heuristic_mdp_results_plot = plot_mdp_results(heuristic_results, "Heuristic Policy")
savefig(heuristic_mdp_results_plot, "results/figures/heuristic_mdp_results_$(num_episodes)_$(steps_per_episode).png")

# Find MDP policies
mdp_policies_dict = find_policies_across_lambdas(lambda_values, solver=ValueIterationSolver(max_iterations=30), convert_to_mdp=true)
mdp_results = calculate_avg_rewards(mdp_policies_dict, episodes=num_episodes, steps_per_episode=steps_per_episode)
mdp_results_plot = plot_mdp_results(mdp_results, "MDP Policy")
savefig(mdp_results_plot, "results/figures/mdp_results_$(num_episodes)_$(steps_per_episode).png")

# Find SARSOP policies
sarsop_policies_dict = find_policies_across_lambdas(lambda_values, solver=SARSOPSolver(; max_time=10.0), convert_to_mdp=false)
sarsop_results = calculate_avg_rewards(sarsop_policies_dict, episodes=num_episodes, steps_per_episode=steps_per_episode)
sarsop_results_plot = plot_mdp_results(sarsop_results, "SARSOP Policy")
savefig(sarsop_results_plot, "results/figures/sarsop_results_$(num_episodes)_$(steps_per_episode).png")

