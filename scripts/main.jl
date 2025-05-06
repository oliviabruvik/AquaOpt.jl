include("../src/cleaning.jl")
include("../src/SeaLicePOMDP.jl")
include("../src/plot_views.jl")
include("../src/optimization.jl")

# Environment variables
ENV["PLOTS_BROWSER"] = "true"
ENV["PLOTS_BACKEND"] = "plotlyjs"

# Import required packages
using Logging
using DiscreteValueIteration
using NativeSARSOP: SARSOPSolver
using POMDPs
using Plots

plotlyjs()  # Activate Plotly backend

# ----------------------------
# Configuration
# ----------------------------
struct Config
    lambda_values::Vector{Float64} # 0.0:0.05:1.0
    num_episodes::Int # 1000
    steps_per_episode::Int # 100
end

const CONFIG = Config(
    collect(0.0:0.2:1.0), # 0.0:0.05:1.0
    10, # 1000
    5 # 100
)

# ----------------------------
# Policy struct
# ----------------------------
struct Algorithm{S<:Solver}
    solver::S
    convert_to_mdp::Bool
    solver_name::String
    heuristic_threshold::Float64
end

# Create results directories
mkpath("results/figures")
mkpath("results/data")

# ----------------------------
# Main function
# ----------------------------
function main(run_algorithms=false)

    @info "Loading and cleaning data"
    df = load_and_clean("data/raw/licedata.csv")
    sealice_levels_over_time_plot = plot_sealice_levels_over_time(df)

    if run_algorithms
        algorithms = [
            Algorithm(ValueIterationSolver(max_iterations=30), true, "Heuristic_Policy", 0.5),
            Algorithm(ValueIterationSolver(max_iterations=30), true, "VI_Policy", 0.0),
            Algorithm(SARSOPSolver(max_time=10.0), false, "SARSOP_Policy", 0.0),
            Algorithm(QMDPSolver(max_iterations=30), false, "QMDP_Policy", 0.0)
        ]

        for algo in algorithms
            @info "Running $(algo.solver_name)"
            test_optimizer(algo, CONFIG)
        end
    end

    @info "Generating result plots"
    overlay_plot = plot_mdp_results_overlay(CONFIG.num_episodes, CONFIG.steps_per_episode)
    comparison_plot = plot_policy_sealice_levels(CONFIG.num_episodes, CONFIG.steps_per_episode)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main("--run" in ARGS)
end