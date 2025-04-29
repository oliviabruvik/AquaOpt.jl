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

# Heuristic Policy
test_optimizer(
    lambda_values,
    ValueIterationSolver(max_iterations=30),
    episodes=num_episodes,
    steps_per_episode=steps_per_episode,
    convert_to_mdp=true,
    plot_name="Heuristic Policy"
)

# MDP
test_optimizer(
    lambda_values,
    ValueIterationSolver(max_iterations=30),
    episodes=num_episodes,
    steps_per_episode=steps_per_episode,
    convert_to_mdp=true,
    plot_name="MDP Policy"
)

# SARSOP
test_optimizer(
    lambda_values,
    SARSOPSolver(; max_time=10.0),
    episodes=num_episodes,
    steps_per_episode=steps_per_episode,
    convert_to_mdp=false,
    plot_name="SARSOP Policy"
)

# QMDP
test_optimizer(
    lambda_values,
    QMDPSolver(max_iterations=30),
    episodes=num_episodes,
    steps_per_episode=steps_per_episode,
    convert_to_mdp=false,
    plot_name="QMDP Policy"
)