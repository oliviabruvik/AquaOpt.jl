### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ 20527f16-ba0c-11f0-b1c2-e3cc93d0c374
begin
    using Pkg #, Revise
	import Revise
    Pkg.activate("/Users/oliviabeyerbruvik/Desktop/AquaOpt")
    Pkg.instantiate()
	using AquaOpt
end

# ╔═╡ 8e3f0645-0f96-4913-baa5-b2950a18c455
begin
	using JLD2

	# Load parallel data
	parallel_data_path = "results/experiments/2025-11-06/2025-11-06T10:30:10.403_exp_mode_debug//simulation_histories/all_policies_simulation_data.jld2"
	saved_parallel_data = load(parallel_data_path)

	# Load config
	config_path = "results/experiments/2025-11-06/2025-11-06T10:30:10.403_exp_mode_debug//config/experiment_config.jld2"
	saved_config = load(config_path)
end

# ╔═╡ a96b743f-c18a-4fff-8222-4f83f3152e7d
begin
	# Test that we can access exported types and functions
	@info "Testing AquaOpt imports..."

	# Available actions
	actions = [NoTreatment, Treatment, ThermalTreatment]
	@info "Available actions: $actions"

	# Get treatment costs
	for action in actions
		cost = get_treatment_cost(action)
		@info "Cost of $action: $cost"
	end

	md"""
	## Available Functions and Types

	The AquaOpt module exports:
	- **Actions**: `NoTreatment`, `Treatment`, `ThermalTreatment`
	- **Config Types**: `ExperimentConfig`, `HeuristicConfig`, `Algorithm`
	- **Main Functions**: `main()`, `run_experiments()`, `setup_experiment_configs()`, `define_algorithms()`
	"""
end

# ╔═╡ 76d768c5-f387-4dda-af1f-c3ad8df923bc
begin
	# Create a custom experiment configuration
	my_config = ExperimentConfig(
		num_episodes = 100,
		steps_per_episode = 52,
		log_space = true,
		ekf_filter = true,
		experiment_name = "pluto_experiment",
		verbose = true,
		step_through = false,
		reward_lambdas = [0.7, 0.2, 0.1, 0.1, 0.8],
		sarsop_max_time = 5.0,
		VI_max_iterations = 10,
		QMDP_max_iterations = 10,
		location = "north",
	)

	md"""
	## Custom Experiment Configuration

	Created a configuration with:
	- **Episodes**: $(my_config.num_episodes)
	- **Steps per episode**: $(my_config.steps_per_episode)
	- **Location**: $(my_config.location)
	- **Log space**: $(my_config.log_space)
	"""
end

# ╔═╡ e2f5b90b-acbd-4741-929e-42609f6f8b53
begin
	# Example: Run a simple experiment
	# Uncomment to run (this will take some time)

	# main(
	#     first_step_flag="solve",
	#     log_space=true,
	#     experiment_name="pluto_test",
	#     mode="debug",
	#     ekf_filter=true
	# )

	md"""
	## Running Experiments

	To run an experiment, uncomment the code above or call:
	```julia
	main(
		first_step_flag="solve",
		log_space=true,
		experiment_name="pluto_test",
		mode="debug",
		ekf_filter=true
	)
	```

	Available modes:
	- `"debug"`: Fast mode with 10 episodes, 20 steps
	- `"light"`: 1000 episodes, 104 steps
	- `"paper"`: Full paper configuration
	- `"VIdebug"`: Debug Value Iteration
	"""
end

# ╔═╡ c8496cff-6478-4ace-a026-54c7cb6ddb40
begin
	Revise.revise()
	experiment_name = "exp"
	log_space = true
	ekf_filter = false
	mode = "debug"
	
	config, heuristic_config = setup_experiment_configs(experiment_name, log_space, ekf_filter, mode)

	algorithms = define_algorithms(config, heuristic_config)

	 @info "Solving policies"
    for algo in algorithms
        generate_mdp_pomdp_policies(algo, config)
    end

	@info "Simulating policies"
    parallel_data = simulate_all_policies(algorithms, config)

    # If we are simulating on high fidelity model, we want to evaluate the simulation results
    if config.high_fidelity_sim
        for algo in algorithms
            histories = extract_simulation_histories(config, algo, parallel_data)
            evaluate_simulation_results(config, algo, histories)
        end
    else
        print_reward_metrics_for_vi_policy(parallel_data, config)
        exit()
    end

    # Extract reward metrics
    processed_data = extract_reward_metrics(parallel_data, config)

    # Display reward metrics
    display_reward_metrics(processed_data, config, false)

	
end

# ╔═╡ 1afa29b0-bef0-4f7a-8902-d0fd29c8ea9c
processed_data

# ╔═╡ 9fd7e94c-b94e-47d9-9c26-f3ac56261db9
begin
	Revise.revise()
	plot_plos_one_plots(parallel_data, config)
end


# ╔═╡ 0cf9258b-e7ae-4d9d-b28c-47bff2cc6e81
plot_policy_sealice_levels_over_time(config, 0.6)

# ╔═╡ 4ff65366-2bce-4777-a57c-b91e96b527ef
plot_policy_treatment_cost_over_time(config, 0.6)

# ╔═╡ Cell order:
# ╠═20527f16-ba0c-11f0-b1c2-e3cc93d0c374
# ╠═a96b743f-c18a-4fff-8222-4f83f3152e7d
# ╠═76d768c5-f387-4dda-af1f-c3ad8df923bc
# ╠═e2f5b90b-acbd-4741-929e-42609f6f8b53
# ╠═8e3f0645-0f96-4913-baa5-b2950a18c455
# ╠═c8496cff-6478-4ace-a026-54c7cb6ddb40
# ╠═1afa29b0-bef0-4f7a-8902-d0fd29c8ea9c
# ╠═9fd7e94c-b94e-47d9-9c26-f3ac56261db9
# ╠═0cf9258b-e7ae-4d9d-b28c-47bff2cc6e81
# ╠═4ff65366-2bce-4777-a57c-b91e96b527ef
