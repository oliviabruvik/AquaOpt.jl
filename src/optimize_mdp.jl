using DataFrames
using DiscreteValueIteration
import Distributions: Normal, Uniform
using JLD2
using POMDPLinter
using POMDPModels
using POMDPs
using POMDPTools
using QMDP
using QuickPOMDPs
using NativeSARSOP

# Define the state
struct SeaLiceState
	SeaLiceLevel::Float64
end

# Define the observation
struct SeaLiceObservation
	SeaLiceLevel::Float64
end

# Define the action
@enum Action NoTreatment Treatment

# Define the MDP
struct SeaLiceMDP <: POMDP{SeaLiceState, Action, SeaLiceObservation}
	lambda::Float64
	costOfTreatment::Float64
	growthRate::Float64
	rho::Float64
    discount_factor::Float64
end

# Constructor
function SeaLiceMDP(;
    lambda::Float64 = 0.5,
    costOfTreatment::Float64 = 10.0,
    growthRate::Float64 = 1.2,
    rho::Float64 = 0.7,
    discount_factor::Float64 = 0.95
)
    return SeaLiceMDP(
        lambda,
        costOfTreatment,
        growthRate,
        rho,
        discount_factor
    )
end

# POMDP interface functions
POMDPs.actions(mdp::SeaLiceMDP) = [NoTreatment, Treatment]
POMDPs.states(mdp::SeaLiceMDP) = [SeaLiceState(round(i, digits=1)) for i in 0:0.1:10]
POMDPs.observations(mdp::SeaLiceMDP) = [SeaLiceObservation(round(i, digits=1)) for i in 0:0.1:10]
POMDPs.isterminal(mdp::SeaLiceMDP, s::SeaLiceState) = false
POMDPs.discount(mdp::SeaLiceMDP) = mdp.discount_factor

# stateindex and actionindex functions
function POMDPs.stateindex(mdp::SeaLiceMDP, s::SeaLiceState)
    return clamp(round(Int, s.SeaLiceLevel * 10) + 1, 1, 101)
end

# Convert action to index (1 for NoTreatment, 2 for Treatment)
function POMDPs.actionindex(mdp::SeaLiceMDP, a::Action)
    return Int(a) + 1
end

# Convert observation to index
function POMDPs.obsindex(mdp::SeaLiceMDP, o::SeaLiceObservation)
    return clamp(round(Int, o.SeaLiceLevel * 10) + 1, 1, 101)
end

function POMDPs.transition(mdp::SeaLiceMDP, s::SeaLiceState, a::Action)
    growth_rate = mdp.growthRate
    rho = a == Treatment ? mdp.rho : 0.0
    next_sea_lice_mean = (1-rho) * exp(growth_rate) * s.SeaLiceLevel
    
    std_dev = 1.0
    points = [
        next_sea_lice_mean - 2*std_dev,
        next_sea_lice_mean - std_dev,
        next_sea_lice_mean,
        next_sea_lice_mean + std_dev,
        next_sea_lice_mean + 2*std_dev
    ]
    
    # Calculate probabilities using normal distribution
    probs = [exp(-(x - next_sea_lice_mean)^2 / (2*std_dev^2)) for x in points]
    
    # Normalize
    probs = probs / sum(probs)
    
    # Create states and clamp/round values
    states = [SeaLiceState(round(clamp(x, 0.0, 10.0), digits=1)) for x in points]
    
    return SparseCat(states, probs)
end

function POMDPs.reward(mdp::SeaLiceMDP, s::SeaLiceState, a::Action)
    if a == Treatment
        return - (mdp.lambda * s.SeaLiceLevel + (1 - mdp.lambda) * mdp.costOfTreatment)
    else
        return - (mdp.lambda * s.SeaLiceLevel)
    end
end

# Create a uniform distribution over initial states from 0.0 to 1.0
function POMDPs.initialstate(mdp::SeaLiceMDP)
    states = [SeaLiceState(round(i, digits=1)) for i in 0:0.1:1.0]
    probs = ones(length(states)) / length(states)
    return SparseCat(states, probs)
end

function POMDPs.observation(mdp::SeaLiceMDP, a::Action, s::SeaLiceState)
    std_dev = 1.0
    points = [
        s.SeaLiceLevel - 2*std_dev,
        s.SeaLiceLevel - std_dev,
        s.SeaLiceLevel,
        s.SeaLiceLevel + std_dev,
        s.SeaLiceLevel + 2*std_dev
    ]
    
    # Calculate probabilities using normal distribution
    probs = [exp(-(x - s.SeaLiceLevel)^2 / (2*std_dev^2)) for x in points]
    
    # Normalize
    probs = probs / sum(probs)
    
    # Create observations and clamp/round values
    observations = [SeaLiceObservation(round(clamp(x, 0.0, 10.0), digits=1)) for x in points]
    
    return SparseCat(observations, probs)
end

function find_policies_across_lambdas(lambda_values; solver, convert_to_mdp=false)
    policies = Dict{Float64, Tuple{Policy, SeaLiceMDP, MDP}}()
    for λ in lambda_values
        pomdp = SeaLiceMDP(lambda=λ)
        mdp = UnderlyingMDP(pomdp)

        # Convert to MDP if requested
        if convert_to_mdp
            policy = solve(solver, mdp)
        else
            policy = solve(solver, pomdp)
        end
        policies[λ] = (policy, pomdp, mdp)

        # save policy
        save("results/policies/sea_lice_mdp_policy_$(λ).jld2", "policy", policy)
    end
    return policies
end

# Calculate average cost and average sea lice level for each lambda
function calculate_avg_rewards(policies_dict; episodes=100, steps_per_episode=50)
    results = DataFrame(lambda=Float64[], avg_treatment_cost=Float64[], avg_sealice=Float64[])
    
    for (λ, (policy, pomdp, mdp)) in pairs(policies_dict)
        
        # optimize mdp
        avg_cost, avg_sealice = run_simulation(policy, mdp, pomdp, episodes, steps_per_episode)

        # add results to dataframe
        push!(results, (λ, avg_cost, avg_sealice))

    end

    rename!(results, [:lambda, :avg_treatment_cost, :avg_sealice])
    return results
end

# Run simulations of policy passed in
function run_simulation(policy, mdp, pomdp, episodes=100, steps_per_episode=50)
    
    total_cost = 0.0
    total_sealice = 0.0
    total_steps = episodes * steps_per_episode

    for _ in 1:episodes
        s = rand(initialstate(mdp))
        b = Deterministic(s)  # For MDP policies, use a deterministic belief
        
        for _ in 1:steps_per_episode
            if policy isa AlphaVectorPolicy
                a = action(policy, b)
            else
                a = action(policy, s)
            end
            
            # Apply action and get reward
            total_cost += (a == Treatment ? pomdp.costOfTreatment : 0.0)
            total_sealice += s.SeaLiceLevel
            
            # Transition to next state
            s = rand(transition(pomdp, s, a))
            
            # Update belief
            if policy isa AlphaVectorPolicy
                o = rand(observation(pomdp, a, s))
                b = update_belief(b, a, o, pomdp)
            end
        end
    end

    avg_cost = total_cost / total_steps
    avg_sealice = total_sealice / total_steps

    return avg_cost, avg_sealice
end



# Helper function to update belief state
function update_belief(b, a, o, pomdp)
    # TODO: Update belief based on observation (avoided now because of complexity)
    # TODO: Review discretization of state space (now 0.1)
    # TODO: Implement particle filter for belief update?
    return Deterministic(SeaLiceState(o.SeaLiceLevel))
end

# Add heuristic policy
struct HeuristicPolicy{P<:MDP} <: Policy
    mdp::P
end

# Heuristic action
function POMDPs.action(policy::HeuristicPolicy, s::SeaLiceState)
    if s.SeaLiceLevel > 0.5
        return Treatment
    else
        return NoTreatment
    end
end

# Create heuristic policy dict
function create_heuristic_policy_dict(lambda_values)
    policies = Dict{Float64, Tuple{Policy, SeaLiceMDP, MDP}}()

    for λ in lambda_values
        pomdp = SeaLiceMDP(lambda=λ)
        mdp = UnderlyingMDP(pomdp)
        policy = HeuristicPolicy(mdp)
        policies[λ] = (policy, pomdp, mdp)
    end
    return policies
end

function test_optimizer(lambda_values, solver; episodes=100, steps_per_episode=50, convert_to_mdp=false, plot_name="MDP Policy")
    if solver isa Nothing
        policies_dict = create_heuristic_policy_dict(lambda_values)
    else
        policies_dict = find_policies_across_lambdas(lambda_values, solver=solver, convert_to_mdp=convert_to_mdp)
    end
    results = calculate_avg_rewards(policies_dict, episodes=episodes, steps_per_episode=steps_per_episode)
    results_plot = plot_mdp_results(results, plot_name)
    savefig(results_plot, "results/figures/$(plot_name)_$(episodes)_$(steps_per_episode).png")
end