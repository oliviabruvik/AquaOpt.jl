include("Algorithms/Evaluation.jl")
include("Algorithms/Policies.jl")
include("Algorithms/Simulation.jl")
include("Data/Cleaning.jl")
include("Models/SeaLicePOMDP.jl")
include("Plotting/Heatmaps.jl")
include("Plotting/Timeseries.jl")
include("Plotting/Comparison.jl")
include("Utils/Config.jl")
include("Utils/ExperimentTracking.jl")

# Environment variables
ENV["PLOTS_BROWSER"] = "true"
ENV["PLOTS_BACKEND"] = "plotlyjs"

# Import required packages
using Logging
using DiscreteValueIteration
using GridInterpolations
using NativeSARSOP
using SARSOP
using POMDPs
using POMDPTools
using POMDPXFiles
using Plots: plot, plot!, scatter, scatter!, heatmap, heatmap!, histogram, histogram!, savefig
using LocalFunctionApproximation
using LocalApproximationValueIteration
using Dates

plotlyjs()  # Activate Plotly backend

# ----------------------------
# Main function
# ----------------------------
function main(;run_algorithms=true, log_space=true, experiment_name="exp", skew=false, mode="light")

    EXPERIMENT_CONFIG, HEURISTIC_CONFIG = setup_experiment_configs(
        experiment_name,
        log_space,
        skew,
        mode
    )

    algorithms = define_algorithms(EXPERIMENT_CONFIG, HEURISTIC_CONFIG)

    # Solve POMDPs and simulate policies
    if run_algorithms

        # Solve POMDPs
        for algo in algorithms
            @info "Solving $(algo.solver_name)"
            generate_mdp_pomdp_policies(algo, EXPERIMENT_CONFIG)
        end

        # Simulate policies
        for algo in algorithms
            @info "Simulating $(algo.solver_name)"
            histories = simulate_policy(algo, EXPERIMENT_CONFIG)
            avg_results = evaluate_simulation_results(EXPERIMENT_CONFIG, algo, histories)
        end
    end

    # Plot results
    plot_results(algorithms, EXPERIMENT_CONFIG)

end


# ----------------------------
# Plot results
# ----------------------------
function plot_results(algorithms, EXPERIMENT_CONFIG)

    @info "Generating result plots"

    # Plot individual policy plots
    for algo in algorithms

        # Load histories
        histories_dir = joinpath(EXPERIMENT_CONFIG.data_dir, "simulation_histories", "$(algo.solver_name)")
        histories_filename = "$(algo.solver_name)_histories"
        @load joinpath(histories_dir, "$(histories_filename).jld2") histories

        # Load avg results
        results_dir = joinpath(EXPERIMENT_CONFIG.data_dir, "avg_results")
        avg_results_filename = "$(algo.solver_name)_avg_results"
        @load joinpath(results_dir, "$(avg_results_filename).jld2") avg_results

        # Plot policy cost vs sealice
        plot_policy_cost_vs_sealice(histories, avg_results, algo.solver_name, EXPERIMENT_CONFIG)

        # Plot policy belief levels
        plot_policy_belief_levels(histories, algo.solver_name, EXPERIMENT_CONFIG, 0.6)

        # Plot treatment heatmap
        plot_treatment_heatmap(algo, EXPERIMENT_CONFIG)

        # Plot simulation treatment heatmap
        plot_simulation_treatment_heatmap(algo, EXPERIMENT_CONFIG; use_observations=false, n_bins=50)

    end
    
    # Plot comparison plots
    plot_all_cost_vs_sealice(EXPERIMENT_CONFIG)
    plot_policy_sealice_levels_over_lambdas(EXPERIMENT_CONFIG)
    plot_policy_treatment_cost_over_lambdas(EXPERIMENT_CONFIG)
    plot_policy_sealice_levels_over_time(EXPERIMENT_CONFIG, 0.6)
    plot_policy_treatment_cost_over_time(EXPERIMENT_CONFIG, 0.6)
    plot_policy_reward_over_lambdas(EXPERIMENT_CONFIG)

    # Generate Pareto frontier
    plot_pareto_frontier(EXPERIMENT_CONFIG)

end

# ----------------------------
# Set up and save experiment configuration
# ----------------------------
function setup_experiment_configs(experiment_name, log_space, skew=false, mode="light")

    @info "Setting up experiment configurations"
    @info "Experiment name: $experiment_name"
    @info "Log space: $log_space"
    @info "Skew: $skew"
    @info "Mode: $mode"

    # Define experiment configuration
    exp_name = string(Dates.today(), "/", Dates.now(), "_", experiment_name, "_mode_", mode)

    if mode == "light"
        EXPERIMENT_CONFIG = ExperimentConfig(
            num_episodes=10,
            steps_per_episode=52,
            log_space=log_space,
            skew=skew,
            experiment_name=exp_name,
            verbose=false,
            step_through=false,
            lambda_values=[0.4, 0.6],
            sarsop_max_time=5.0,
            VI_max_iterations=10,
            QMDP_max_iterations=10,
        )
    elseif mode == "debug"
        EXPERIMENT_CONFIG = ExperimentConfig(
            num_episodes=1,
            steps_per_episode=20,
            log_space=log_space,
            skew=skew,
            experiment_name=exp_name,
            verbose=true,
            step_through=true,
            lambda_values=[0.2, 0.4, 0.6, 0.8],
            sarsop_max_time=5.0,
            VI_max_iterations=10,
            QMDP_max_iterations=10,
        )
    elseif mode == "full"
        EXPERIMENT_CONFIG = ExperimentConfig(
            num_episodes=10,
            steps_per_episode=52,
            log_space=log_space,
            skew=skew,
            experiment_name=exp_name,
            verbose=false,
            step_through=false,
        )
    end
        

    # Define heuristic configuration
    HEURISTIC_CONFIG = HeuristicConfig(
        raw_space_threshold=EXPERIMENT_CONFIG.heuristic_threshold,
        belief_threshold=EXPERIMENT_CONFIG.heuristic_belief_threshold,
        rho=EXPERIMENT_CONFIG.heuristic_rho
    )

    # Write EXPERIMENT_CONFIG to file
    mkpath(joinpath(EXPERIMENT_CONFIG.data_dir, "config"))
    @save joinpath(EXPERIMENT_CONFIG.data_dir, "config", "experiment_config.jld2") EXPERIMENT_CONFIG
    open(joinpath(EXPERIMENT_CONFIG.data_dir, "config", "experiment_config.txt"), "w") do io
        for field in fieldnames(typeof(EXPERIMENT_CONFIG))
            value = getfield(EXPERIMENT_CONFIG, field)
            println(io, "$field: $value")
        end
    end

    # Log experiment configuration
    save_experiment_config(EXPERIMENT_CONFIG, HEURISTIC_CONFIG)
    latest_experiment = find_latest_experiment(EXPERIMENT_CONFIG, HEURISTIC_CONFIG)
    @info "Latest experiment: $latest_experiment"

    return EXPERIMENT_CONFIG, HEURISTIC_CONFIG

end

# ----------------------------
# Define algorithms
# ----------------------------

function define_algorithms(EXPERIMENT_CONFIG, HEURISTIC_CONFIG)

    @info "Defining algorithms"

    algorithms = [
        Algorithm(solver_name="NeverTreat_Policy"),
        Algorithm(solver_name="AlwaysTreat_Policy"),
        Algorithm(solver_name="Random_Policy"),
        Algorithm(solver_name="Heuristic_Policy", heuristic_config=HEURISTIC_CONFIG),
        Algorithm(
            solver=NativeSARSOP.SARSOPSolver(max_time=EXPERIMENT_CONFIG.sarsop_max_time, verbose=false), #true),
            solver_name="SARSOP_Policy",
        ),
        Algorithm(
            solver=SARSOP.SARSOPSolver(
                timeout=EXPERIMENT_CONFIG.sarsop_max_time,
                verbose=false, #true),
                policy_filename=joinpath(EXPERIMENT_CONFIG.data_dir, "policies/NUS_SARSOP_Policy/policy.out"),
                pomdp_filename=joinpath(EXPERIMENT_CONFIG.data_dir, "pomdp_mdp/pomdp.pomdpx")
            ),
            solver_name="NUS_SARSOP_Policy",
        ),
        Algorithm(solver=ValueIterationSolver(max_iterations=EXPERIMENT_CONFIG.VI_max_iterations), solver_name="VI_Policy"),
        Algorithm(solver=QMDPSolver(max_iterations=EXPERIMENT_CONFIG.QMDP_max_iterations), solver_name="QMDP_Policy"),
    ]

    return algorithms

end


if abspath(PROGRAM_FILE) == @__FILE__

    run_algorithms_flag = true
    log_space_flag = true
    skew_flag = false
    experiment_name_flag = "exp"
    mode_flag = "light"

    for arg in ARGS
        if occursin("--experiment_name=", arg)
            global experiment_name_flag = split(arg, "=")[2]
        elseif occursin("--mode=", arg)
            global mode_flag = split(arg, "=")[2]
        elseif arg == "--no-run_algorithms"
            global run_algorithms_flag = false
        elseif arg == "--no-log_space"
            global log_space_flag = false
        elseif arg == "--skew"
            global skew_flag = true
        end
    end

    println("Running with mode: $mode_flag")

    main(run_algorithms=run_algorithms_flag, log_space=log_space_flag, experiment_name=experiment_name_flag, mode=mode_flag, skew=skew_flag)
end