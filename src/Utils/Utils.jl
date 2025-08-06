using Distributions
using POMDPs

# -------------------------
# Discretized Normal Distribution
# -------------------------
function discretize_distribution(dist::Distribution, space::Any, skew::Bool=false)

    probs = zeros(length(space))
    past_cdf = 0.0
    
    # Add safety check for NaN values
    for (i, s) in enumerate(space)
        curr_cdf = cdf(dist, s.SeaLiceLevel)
        
        # Check for NaN values
        if isnan(curr_cdf)
            @warn("NaN detected in CDF calculation. Distribution: $dist, State: $s")
            # Fallback to uniform distribution
            return fill(1.0/length(space), length(space))
        end
        
        probs[i] = curr_cdf - past_cdf
        past_cdf = curr_cdf
    end

    # Check for negative probabilities (shouldn't happen with CDF)
    if any(x -> x < -1e-10, probs)
        @warn("Negative probabilities detected, clamping to 0")
        probs = max.(probs, 0.0)
    end

    if sum(probs) < 0.01
        println("type of space: $(typeof(space))")
        println("dist: $dist")
        @warn("Probs over distribution very small")
    end

    probs = normalize(probs, 1)

    # Check that the probs sum to 1.0
    try
        @assert sum(probs) ≈ 1.0
    catch
        println("probs: $probs")
        println("sum(probs): $(sum(probs))")
        error("Probs do not sum to 1.0")
    end

    return probs
end

# -------------------------
# Temperature model: Return estimated weekly sea surface temperature (°C) for Norwegian salmon farms.
# -------------------------
function get_temperature(annual_week::Int64)
    T_mean = 9.0      # average annual temperature (°C)
    T_amp = 4.5       # amplitude (°C)
    peak_week = 27      # aligns peak with July (week ~27)
    return T_mean + T_amp * cos(2π * (annual_week - peak_week) / 52)
end
