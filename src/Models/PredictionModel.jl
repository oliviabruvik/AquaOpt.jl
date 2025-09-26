using Distributions

"""
Sea lice prediction model based on Stige et al. 2025
"A salmon lice prediction model"
https://www.sciencedirect.com/science/article/pii/S0167587724002915

This model predicts sea lice abundance one week ahead based on current abundances
and environmental conditions, following the exact equations from the paper.
"""

# -------------------------
# Model Parameters from Stige et al. 2025 Table 1
# -------------------------

# Weekly survival probabilities (Table 1)
const s1 = 0.49  # sessile survival
const s2 = 2.3   # sessile → motile scaling  
const s3 = 0.88  # motile survival
const s4 = 0.61  # adult survival

# Development rate parameters (Table 1)
const d1_intercept = -2.4
const d1_temp_coeff = 0.37
const d2_intercept = -2.1
const d2_temp_coeff = 0.037

# Temperature reference point
const temp_ref = 9.0  # °C

# -------------------------
# Development Rate Functions (from paper equations)
# -------------------------

"""
Calculate development rate d1 (sessile to motile) based on temperature.
From Stige et al. 2025 equation: d1 = 1 / (1 + exp(-(-2.4 + 0.37*(T-9))))
"""
function d1(temperature::Float64)
    return 1.0 / (1.0 + exp(-(d1_intercept + d1_temp_coeff * (temperature - temp_ref))))
end

"""
Calculate development rate d2 (motile to adult) based on temperature.
From Stige et al. 2025 equation: d2 = 1 / (1 + exp(-(-2.1 + 0.037*(T-9))))
"""
function d2(temperature::Float64)
    return 1.0 / (1.0 + exp(-(d2_intercept + d2_temp_coeff * (temperature - temp_ref))))
end

# -------------------------
# One-Week Prediction Model (exact equations from paper)
# -------------------------

"""
Predict sea lice abundances one week ahead based on current abundances and temperature.
Following the exact equations from Stige et al. 2025.

Args:
    adult::Float64: Current adult sea lice abundance
    sessile::Float64: Current sessile sea lice abundance  
    motile::Float64: Current motile sea lice abundance
    temperature::Float64: Current temperature (°C)

Returns:
    Tuple{Float64, Float64, Float64}: Predicted (adult, sessile, motile) abundances
"""
function predict_next_lice(adult::Float64, motile::Float64, sessile::Float64, temp::Float64)
    
    # Calculate development rates
    d1_val = d1(temp)
    d2_val = d2(temp)
    
    # Stage transitions following exact equations from the paper:
    
    # Sessile stage: next_sessile = s1 * sessile
    next_sessile = s1 * sessile
    
    # Motile stage: next_motile = s3 * (1 - d2) * motile + s2 * d1 * sessile
    next_motile = s3 * (1 - d2_val) * motile + s2 * d1_val * sessile
    
    # Adult stage: next_adult = s4 * adult + d2 * 0.5 * (s3 + s4) * motile
    next_adult = s4 * adult + d2_val * 0.5 * (s3 + s4) * motile
    
    return next_sessile, next_motile, next_adult
end

"""
Predict sea lice abundances one week ahead with treatment effect.
Treatment only affects adult stage according to the paper.

Args:
    adult::Float64: Current adult sea lice abundance
    sessile::Float64: Current sessile sea lice abundance  
    motile::Float64: Current motile sea lice abundance
    temperature::Float64: Current temperature (°C)
    treatment_efficacy::Float64: Treatment efficacy (0.0 to 1.0, default 0.95)

Returns:
    Tuple{Float64, Float64, Float64}: Predicted (adult, sessile, motile) abundances
"""
function predict_next_lice_with_treatment(
    adult::Float64, 
    motile::Float64, 
    sessile::Float64, 
    temp::Float64;
    treatment_efficacy::Float64 = 0.95
)
    
    # Get base predictions
    next_sessile, next_motile, next_adult = predict_next_lice(adult, motile, sessile, temp)
    
    # Apply treatment effect (only affects adult stage)
    next_adult *= (1.0 - treatment_efficacy)
    
    return next_sessile, next_motile, next_adult
end

# -------------------------
# Temperature Model (from paper)
# -------------------------

"""
Calculate weekly temperature based on annual cycle.
From Stige et al. 2025 temperature model.

Args:
    annual_week::Int: Week of the year (1-52)
    
Returns:
    Float64: Predicted temperature (°C)
"""
function temperature_model(annual_week::Int64)
    T_mean = 9.0      # average annual temperature (°C)
    T_amp = 4.5       # amplitude (°C)
    peak_week = 27    # aligns peak with July (week ~27)
    
    return T_mean + T_amp * cos(2π * (annual_week - peak_week) / 52)
end

# -------------------------
# Validation and Utility Functions
# -------------------------

"""
Validate that input abundances are non-negative.
"""
function validate_inputs(adult::Float64, sessile::Float64, motile::Float64, temperature::Float64)
    if adult < 0 || sessile < 0 || motile < 0
        error("Abundances must be non-negative")
    end
    if temperature < -10 || temperature > 30
        @warn "Temperature $temperature is outside expected range (-10 to 30°C)"
    end
end

"""
Print model parameters for reference.
"""
function print_model_parameters()
    println("Sea Lice Prediction Model Parameters (Stige et al. 2025):")
    println("========================================================")
    println("Survival Probabilities (Table 1):")
    println("  s1 (sessile survival): $s1")
    println("  s2 (sessile → motile scaling): $s2")
    println("  s3 (motile survival): $s3")
    println("  s4 (adult survival): $s4")
    println()
    println("Development Rate Parameters (Table 1):")
    println("  d1 intercept: $d1_intercept")
    println("  d1 temperature coefficient: $d1_temp_coeff")
    println("  d2 intercept: $d2_intercept")
    println("  d2 temperature coefficient: $d2_temp_coeff")
    println("  Temperature reference: $(temp_ref)°C")
    println()
    println("Stage Transition Equations:")
    println("  next_sessile = s1 * sessile")
    println("  next_motile = s3 * (1 - d2) * motile + s2 * d1 * sessile")
    println("  next_adult = s4 * adult + d2 * 0.5 * (s3 + s4) * motile")
end

# -------------------------
# Example Usage
# -------------------------

"""
Example usage of the prediction model.
"""
function example_usage()
    println("Example: One-week sea lice prediction (Stige et al. 2025)")
    println("=========================================================")
    
    # Current abundances
    adult = 0.5
    sessile = 0.3
    motile = 0.2
    temp = 12.0
    
    println("Current abundances:")
    println("  Adult: $adult")
    println("  Sessile: $sessile")
    println("  Motile: $motile")
    println("  Temperature: $(temp)°C")
    println()
    
    # Calculate development rates
    d1_val = d1(temp)
    d2_val = d2(temp)
    println("Development rates at $(temp)°C:")
    println("  d1 (sessile→motile): $(round(d1_val, digits=3))")
    println("  d2 (motile→adult): $(round(d2_val, digits=3))")
    println()
    
    # Predict without treatment
    next_sessile, next_motile, next_adult = predict_next_lice(adult, motile, sessile, temp)
    
    println("Predicted abundances (no treatment):")
    println("  Adult: $(round(next_adult, digits=3))")
    println("  Sessile: $(round(next_sessile, digits=3))")
    println("  Motile: $(round(next_motile, digits=3))")
    println()
    
    # Predict with treatment
    next_sessile_treat, next_motile_treat, next_adult_treat = predict_next_lice_with_treatment(
        adult, motile, sessile, temp
    )
    
    println("Predicted abundances (with treatment):")
    println("  Adult: $(round(next_adult_treat, digits=3))")
    println("  Sessile: $(round(next_sessile_treat, digits=3))")
    println("  Motile: $(round(next_motile_treat, digits=3))")
end

# Export main functions
export predict_next_lice, predict_next_lice_with_treatment, temperature_model
export d1, d2, print_model_parameters, example_usage
