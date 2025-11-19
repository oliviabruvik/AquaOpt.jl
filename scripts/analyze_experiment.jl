#!/usr/bin/env julia

using AquaOpt
using DataFrames
using JLD2
using Printf

struct AnalyzeOptions
    experiment_path::String
    output_dir::String
    generate_plots::Bool
    policies_to_plot::Union{Nothing, Vector{String}}
end

function usage()
    println("Usage: julia --project scripts/analyze_experiment.jl --experiment <path> [options]")
    println("  --experiment, -e  Path to a finished experiment directory (required).")
    println("  --output-dir DIR  Directory where plots should be saved (default: final_plots/ inside experiment).")
    println("  --no-plots        Skip generating plots.")
    println("  --policies LIST   Comma-separated list of policy names to include in multi-policy plots.")
    println("You can also pass the experiment directory as the first positional argument.")
end

function parse_cli_args(args::Vector{String})
    experiment_path::Union{Nothing, String} = nothing
    output_dir::Union{Nothing, String} = nothing
    generate_plots = true
    policies_to_plot = nothing
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-e", "--experiment")
            i += 1
            if i > length(args)
                usage()
                error("Missing value for --experiment")
            end
            experiment_path = args[i]
        elseif arg in ("--output-dir", "--output")
            i += 1
            if i > length(args)
                usage()
                error("Missing value for --output-dir")
            end
            output_dir = args[i]
        elseif arg == "--no-plots"
            generate_plots = false
        elseif arg == "--policies"
            i += 1
            if i > length(args)
                usage()
                error("Missing value for --policies")
            end
            policies_to_plot = split(strip(args[i]), ',')
            policies_to_plot = [strip(p) for p in policies_to_plot if !isempty(strip(p))]
            isempty(policies_to_plot) && (policies_to_plot = nothing)
        elseif startswith(arg, "--policies=")
            value = split(arg, "=", limit=2)[2]
            policies_to_plot = split(strip(value), ',')
            policies_to_plot = [strip(p) for p in policies_to_plot if !isempty(strip(p))]
            isempty(policies_to_plot) && (policies_to_plot = nothing)
        elseif startswith(arg, "-")
            usage()
            error("Unknown argument: $arg")
        elseif isnothing(experiment_path)
            experiment_path = arg
        else
            usage()
            error("Unexpected positional argument: $arg")
        end
        i += 1
    end

    if isnothing(experiment_path)
        usage()
        error("An experiment directory must be provided.")
    end

    exp_path = abspath(experiment_path)
    default_output = joinpath(exp_path, "final_plots")
    resolved_output = isnothing(output_dir) ? default_output : abspath(output_dir)

    return AnalyzeOptions(exp_path, resolved_output, generate_plots, policies_to_plot)
end

function adjust_config_paths!(config, experiment_root::String)
    config.experiment_dir = experiment_root
    config.policies_dir = joinpath(experiment_root, "policies")
    config.simulations_dir = joinpath(experiment_root, "simulation_histories")
    config.results_dir = joinpath(experiment_root, "avg_results")
    config.figures_dir = joinpath(experiment_root, "figures")
    return config
end

function load_experiment_config(experiment_root::String)
    cfg_path = joinpath(experiment_root, "config", "experiment_config.jld2")
    isfile(cfg_path) || error("Could not find config file at $cfg_path")
    @load cfg_path config
    return adjust_config_paths!(config, experiment_root)
end

function ensure_dataframe(data)
    df = data isa DataFrame ? data : DataFrame(data)
    rename!(df, Symbol.(names(df)))
    return df
end

function load_parallel_data(experiment_root::String)
    data_path = joinpath(experiment_root, "simulation_histories", "all_policies_simulation_data.jld2")
    isfile(data_path) || error("Could not find simulation data at $data_path")
    @load data_path data
    return ensure_dataframe(data)
end

function summarize_experiment(config, parallel_data; output_dir::Union{Nothing,String}=nothing, generate_plots::Bool=true, policies_to_plot=nothing)
    processed_data = extract_reward_metrics(parallel_data, config)
    println("Loaded $(nrow(processed_data)) episodes across $(length(unique(processed_data.policy))) policies.")

    mkpath(config.results_dir)
    # display_reward_metrics(processed_data, config, false)

    if generate_plots
        target_dir = isnothing(output_dir) ? config.figures_dir : output_dir
        mkpath(target_dir)
        temp_config = deepcopy(config)
        temp_config.figures_dir = target_dir
        plot_plos_one_plots(processed_data, temp_config; policies_to_plot=policies_to_plot)
        println("Plots saved to $(temp_config.figures_dir).")
    else
        println("Skipping plot generation (--no-plots flag).")
    end

    return processed_data
end

function main()
    opts = parse_cli_args(ARGS)
    config = load_experiment_config(opts.experiment_path)
    parallel_data = load_parallel_data(opts.experiment_path)
    summarize_experiment(
        config,
        parallel_data;
        output_dir = opts.output_dir,
        generate_plots = opts.generate_plots,
        policies_to_plot = opts.policies_to_plot,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
