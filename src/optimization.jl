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
    heuristic_threshold::Float64
end

# ----------------------------
# Policy Saving & Loading
# ----------------------------
function save_policy(policy, pomdp, mdp, solver_name, lambda)
    save("results/policies/$(solver_name)/$(lambda)/policy.jld2", "policy", policy)
    save("results/policies/$(solver_name)/$(lambda)/pomdp.jld2", "pomdp", pomdp)
    save("results/policies/$(solver_name)/$(lambda)/mdp.jld2", "mdp", mdp)
end

function load_policy(solver_name, lambda)
    policy = load("results/policies/$(solver_name)_sea_lice_mdp_policy_$(lambda).jld2", "policy")
    pomdp = load("results/policies/$(solver_name)_sea_lice_pomdp_$(lambda).jld2", "pomdp")
    mdp = load("results/policies/$(solver_name)_sea_lice_mdp_$(lambda).jld2", "mdp")
    return (policy, pomdp, mdp)
end

# ----------------------------
# Policy Generation
# ----------------------------
function generate_policy(algorithm, λ)
    pomdp = SeaLiceMDP(lambda=λ)
    mdp = UnderlyingMDP(pomdp)

    if algorithm.solver_name == "Heuristic Policy"
        policy = HeuristicPolicy(mdp, algorithm.heuristic_threshold)
    elseif algorithm.convert_to_mdp
        policy = solve(algorithm.solver, mdp)
    else
        policy = solve(algorithm.solver, pomdp)
    end
    
    return (policy, pomdp, mdp)
end

# ----------------------------
# Simulation & Evaluation
# ----------------------------
function run_simulation(policy, mdp, pomdp, config)
    
    total_cost, total_sealice = 0.0, 0.0
    total_steps = config.num_episodes * config.steps_per_episode

    for _ in 1:config.num_episodes
        s = rand(initialstate(mdp))
        b = Deterministic(s)
        
        for _ in 1:config.steps_per_episode
            a = policy isa AlphaVectorPolicy ? action(policy, b) : action(policy, s)
            total_cost += (a == Treatment ? pomdp.costOfTreatment : 0.0)
            total_sealice += s.SeaLiceLevel
            
            # Transition to next state
            s = rand(transition(pomdp, s, a))

            # TODO: save the state and action to a file
            # TODO: use discrete updator
            
            # Update belief
            if policy isa AlphaVectorPolicy
                o = rand(observation(pomdp, a, s))
                b = update_belief(b, a, o, pomdp)
            end
        end
    end

    return total_cost / total_steps, total_sealice / total_steps
end


"Very simple belief updater (placeholder for future particle filter)."
function update_belief(b, a, o, pomdp)
    # TODO: Update belief based on observation (avoided now because of complexity)
    # TODO: Review discretization of state space (now 0.1)
    # TODO: Implement particle filter for belief update?
    return Deterministic(SeaLiceState(o.SeaLiceLevel))
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
        save_policy(policy, pomdp, mdp, algorithm.solver_name, λ)

        # Run simulation to calculate average cost and average sea lice level
        avg_cost, avg_sealice = run_simulation(policy, mdp, pomdp, config)
        push!(results, (λ, avg_cost, avg_sealice))
    end

    # Plot results
    results_plot = plot_mdp_results(results, algorithm.solver_name)
    
    # Save results
    @save "results/data/$(algorithm.solver_name)_$(config.num_episodes)_$(config.steps_per_episode).jld2" results
    savefig(results_plot, "results/figures/$(algorithm.solver_name)_$(config.num_episodes)_$(config.steps_per_episode).png")
end