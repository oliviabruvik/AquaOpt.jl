include("cleaning.jl")
include("SeaLicePOMDP.jl")
include("plot_views.jl")
include("optimization.jl")

# Environment variables
ENV["PLOTS_BROWSER"] = "true"
ENV["PLOTS_BACKEND"] = "plotlyjs"

# ----------------------------# Import required packages
using Logging
using DiscreteValueIteration
using GridInterpolations
using NativeSARSOP: SARSOPSolver
using POMDPs
using POMDPTools
using Plots
using StatsPlots  # Add StatsPlots for boxplot support
using LocalFunctionApproximation
using LocalApproximationValueIteration
using StatsPlots
using Optim
using DataFrames

plotlyjs()  # Activate Plotly backend

LUSEDATA_PATH = "data/processed/sealice_data.csv"
BARETSWATCH_PATH = "data/processed/bayesian_outer_data.csv"

function main()
    @info "Running MLE analysis for Luse data"
    luse_data_df = CSV.read(LUSEDATA_PATH, DataFrame)
    luse_data_df = convert_lusedata_to_baretswatch_format(luse_data_df)
    baretswatch_mle_analysis(luse_data_df)
    
    @info "Running MLE analysis for Baretswatch data"
    baretswatch_data_df = CSV.read(BARETSWATCH_PATH, DataFrame)
    baretswatch_mle_analysis(baretswatch_data_df)

end

########################################################
# MLE Analysis for Sea Lice Growth Rate - Lusedata
########################################################
function convert_lusedata_to_baretswatch_format(df)

    # Rename mechanical_removal to treatment
    df.treatment = df.mechanical_removal

    # Rename location_number to site_number
    df.site_number = df.location_number

    return df
end

function lusedata_mle_analysis()

    # Load data
    df = CSV.read(LUSEDATA_PATH, DataFrame)

    # Handle missing values and convert treatment column to binary
    df.treatment = ifelse.(df.treatment .== "None", false, true)

    # Drop rows where lice count is not available or relevant for analyses
    df = dropmissing(df, [:adult_sealice])

    # Split into separate dataframes for each location
    locations = unique(df.location_number)
    location_dfs = [df[df.location_number .== location, :] for location in locations]

    growth_rates = []
    for location_df in location_dfs

        # Filter out rows where treatment is conducted and the row after as well
        # location_df = filter(row -> row.treatment == false, location_df)
        
        # Filter out rows where adult sea lice is 0
        # Without: 0.05
        # With: 0.00184
        # location_df = filter(row -> row.adult_sealice > 0, location_df)

        # Add epsilon to the adult sea lice count: 0.087
        location_df.adult_sealice = location_df.adult_sealice .+ 1e-6

        result = baretswatch_mle_analysis(location_df)
        rounded_result = round(result[1], digits=4)
        # println("MLE for growth rate at location $location: $rounded_result")
        push!(growth_rates, result)
    end

    #println(growth_rates)
    println(mean(growth_rates))
end

# Define log likelihood function for MLE analysis
function log_likelihood(r, y)
    log_likelihood = 0.0
    for t in 2:length(y)
        μ = log(y[t-1]) + r[1]
        log_likelihood += logpdf(Normal(μ, 0.1), log(y[t]))
    end
    return -log_likelihood
end

# Run minimization to find the MLE for the growth rate r
function mle_analysis(df)
    result = optimize(r -> log_likelihood(r, df.adult_sealice), [0.0])
    r_hat = Optim.minimizer(result)
    return r_hat
end

########################################################
# MLE Analysis for Sea Lice Growth Rate - Baretswatchdata
########################################################

function baretswatch_mle_analysis(df)

    # Handle missing values and convert treatment column to binary
    df.treatment = ifelse.(df.treatment .== "None", false, true)

    # Drop rows where lice count is not available or relevant for analyses
    df = dropmissing(df, [:adult_sealice])

    # Split into separate dataframes for each location
    locations = unique(df.site_number)
    location_dfs = [df[df.site_number .== location, :] for location in locations]

    growth_rates = []
    for location_df in location_dfs

        # Filter out rows where treatment is conducted and the row after as well
        # location_df = filter(row -> row.treatment == false, location_df)
        
        # Filter out rows where adult sea lice is 0
        # Without: 0.05
        # With: 0.00184
        # location_df = filter(row -> row.adult_sealice > 0, location_df)

        # Add epsilon to the adult sea lice count: 0.087
        location_df.adult_sealice = location_df.adult_sealice .+ 1e-6

        result = baretswatch_optimization(location_df)
        rounded_result = round(result[1], digits=4)
        # println("MLE for growth rate at location $location: $rounded_result")
        push!(growth_rates, result)
    end

    #println(growth_rates)
    println(mean(growth_rates))
end

# Define log likelihood function for MLE analysis
function baretswatch_log_likelihood(r, df)
    log_likelihood = 0.0
    for t in 2:length(df.adult_sealice)
        # Without constraints, the MLE is 0.009131332484441012]
        # With constraints, the MLE is 0.050504687554673175
        # With week constraint, the MLE is 0.24
        if use_data_for_mle_treatment(df, t) && use_data_for_mle_week(df, t)
            μ = log(df.adult_sealice[t-1]) + r[1]
            log_likelihood += logpdf(Normal(μ, 0.1), log(df.adult_sealice[t]))
        end
    end
    return -log_likelihood
end

# Run minimization to find the MLE for the growth rate r
function baretswatch_optimization(df)
    result = optimize(r -> baretswatch_log_likelihood(r, df), [0.0])
    r_hat = Optim.minimizer(result)
    return r_hat
end

# Constraints for MLE analysis
function use_data_for_mle_treatment(df, t)

    # If treatment is not conducted, then count
    if df.treatment[t] == false
        return true
    end

    # If treatment is conducted, then don't count if sea lice is not increasing
    if df.treatment[t] == true && df.adult_sealice[t-1] >= df.adult_sealice[t]
        # If treatment is conducted and sea lice is not increasing, then don't count
        return false
    elseif df.treatment[t-1] == true && df.treatment[t] == true
        # If treatment is conducted and the previous week was also treated, then don't count
        return false
    elseif df.treatment[t] == true && df.adult_sealice[t-1] < df.adult_sealice[t]
        # Count if treatment is conducted and sea lice is increasing
        # Represents that they counted sea lice next week before the treatment was conducted
        return true
    else
        return true
    end
end

function use_data_for_mle_week(df, t)

    # If week is not available, then don't count
    if df.total_week[t] != df.total_week[t-1] + 1
        return false
    else
        return true
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end