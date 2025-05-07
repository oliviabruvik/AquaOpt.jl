include("SeaLicePOMDP.jl")

using POMDPs
using POMDPModels
using POMDPTools
using DataFrames
using JLD2
using Plots

# ----------------------------
# Algorithm struct
# ----------------------------
struct Algorithm{S<:Solver}
    solver::S
    convert_to_mdp::Bool
    solver_name::String
    heuristic_threshold::Union{Float64, Nothing}
end

# ----------------------------
# Policy Saving & Loading
# ----------------------------
function save_policy(policy, pomdp, mdp, solver_name, lambda, config)
    mkpath(config.policies_dir)
    save(joinpath(config.policies_dir, "$(solver_name)/$(lambda)_lambda_policy.jld2"), "policy", policy)
    save(joinpath(config.policies_dir, "$(solver_name)/$(lambda)_lambda_pomdp.jld2"), "pomdp", pomdp)
    save(joinpath(config.policies_dir, "$(solver_name)/$(lambda)_lambda_mdp.jld2"), "mdp", mdp)
end

function load_policy(solver_name, lambda, config)
    policy = load(joinpath(config.policies_dir, "$(solver_name)/$(lambda)_lambda_policy.jld2"), "policy")
    pomdp = load(joinpath(config.policies_dir, "$(solver_name)/$(lambda)_lambda_pomdp.jld2"), "pomdp")
    mdp = load(joinpath(config.policies_dir, "$(solver_name)/$(lambda)_lambda_mdp.jld2"), "mdp")
    return (policy, pomdp, mdp)
end

# ----------------------------
# Policy Generation
# ----------------------------
function generate_policy(algorithm, λ)
    pomdp = SeaLiceMDP(lambda=λ)
    mdp = UnderlyingMDP(pomdp)

    policy = if algorithm.solver_name == "Heuristic Policy"
        HeuristicPolicy(mdp, algorithm.heuristic_threshold)
    elseif algorithm.convert_to_mdp
        solve(algorithm.solver, mdp)
    else
        solve(algorithm.solver, pomdp)
    end
    
    return (policy, pomdp, mdp)
end

# ----------------------------
# Simulation & Evaluation
# ----------------------------
function simulate(sim::RolloutSimulator, pomdp::POMDP, policy::Policy, updater::Updater, initial_belief, s, algorithm)
    disc = 1.0
    r_total = 0.0
    total_cost = 0.0
    total_sealice = 0.0
    steps = 1

    b = initialize_belief(updater, initial_belief)

    while disc > sim.eps && !isterminal(pomdp, s) && steps <= sim.max_steps
        
        # Choose next action
        a = if algorithm.convert_to_mdp
            action(policy, s)  # Use concrete state for MDP policies
        else
            action(policy, b)  # Use belief state for POMDP policies
        end
        
        # Track costs and sea lice levels
        total_cost += (a == Treatment ? pomdp.costOfTreatment : 0.0)
        total_sealice += s.SeaLiceLevel
        
        # Transition to next state
        sp, o, r = @gen(:sp,:o,:r)(pomdp, s, a, sim.rng)
        r_total += disc*r

        # Update belief
        s = sp
        bp = update(updater, b, a, o)
        b = bp

        disc *= discount(pomdp)
        steps += 1
    end

    return r_total, total_cost / steps, total_sealice / steps
end

function run_simulation(policy, mdp, pomdp, config, algorithm)
    total_cost, total_sealice = 0.0, 0.0
    total_reward = 0.0
    
    # Create simulator and updater
    sim = RolloutSimulator(max_steps=config.steps_per_episode)
    updater = DiscreteUpdater(pomdp)
    
    # Run simulation for each episode
    for _ in 1:config.num_episodes
        s = rand(initialstate(mdp))
        initial_belief = Deterministic(s)
        
        reward, cost, sealice = simulate(sim, pomdp, policy, updater, initial_belief, s, algorithm)
        total_reward += reward
        total_cost += cost
        total_sealice += sealice
    end

    # Return averages
    return total_cost / config.num_episodes, total_sealice / config.num_episodes
end


# ----------------------------
# Heuristic Policy
# ----------------------------
struct HeuristicPolicy{P<:MDP} <: Policy
    mdp::P
    threshold::Float64
end

# Heuristic action
function POMDPs.action(policy::HeuristicPolicy, s::SeaLiceState)
    return s.SeaLiceLevel > policy.threshold ? Treatment : NoTreatment
end

# ----------------------------
# Optimizer Wrapper
# ----------------------------
function test_optimizer(algorithm, config)

    results = DataFrame(lambda=Float64[], avg_treatment_cost=Float64[], avg_sealice=Float64[])

    # Generate policies for each lambda
    for λ in config.lambda_values

        # Generate policy
        policy, pomdp, mdp = generate_policy(algorithm, λ)
        save_policy(policy, pomdp, mdp, algorithm.solver_name, λ, config)

        # Run simulation to calculate average cost and average sea lice level
        avg_cost, avg_sealice = run_simulation(policy, mdp, pomdp, config, algorithm)
        push!(results, (λ, avg_cost, avg_sealice))
    end

    # Plot results
    results_plot = plot_mdp_results(results, algorithm.solver_name)
    
    # Save results
    mkpath(joinpath(config.figures_dir, algorithm.solver_name))
    mkpath(joinpath(config.data_dir, algorithm.solver_name))
    @save joinpath(config.data_dir, "$(algorithm.solver_name)/results_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.jld2") results
    savefig(results_plot, joinpath(config.figures_dir, "$(algorithm.solver_name)/results_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
end