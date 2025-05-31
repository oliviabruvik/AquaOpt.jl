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
    # @info "Running MLE analysis for Luse data"
    # luse_data_df = CSV.read(LUSEDATA_PATH, DataFrame)
    # luse_data_df = convert_lusedata_to_baretswatch_format(luse_data_df)
    # mle_analysis(luse_data_df)

    @info "Running MLE analysis for Baretswatch data in in normal space"
    # Natural for growth in log
    baretswatch_data_df = CSV.read(BARETSWATCH_PATH, DataFrame)
    mle_analysis(baretswatch_data_df, "log-space-normal")

    @info "Running MLE analysis for Baretswatch data in in log normal space"
    # Best when working in count space
    baretswatch_data_df = CSV.read(BARETSWATCH_PATH, DataFrame)
    mle_analysis(baretswatch_data_df, "raw-space-log-normal")

    @info "Running MLE analysis for Baretswatch data in in raw-space"
    # Not appropriate - can go negative
    baretswatch_data_df = CSV.read(BARETSWATCH_PATH, DataFrame)
    mle_analysis(baretswatch_data_df, "raw-space-normal")

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

########################################################
# MLE Analysis for Sea Lice Growth Rate - Baretswatchdata
########################################################
function mle_analysis(df, space="log-space-normal")

    # Handle missing values and convert treatment column to binary
    df.treatment = ifelse.(df.treatment .== "None", false, true)

    # Drop rows where lice count is not available or relevant for analyses
    df = dropmissing(df, [:adult_sealice])

    # Split into separate dataframes for each location
    locations = unique(df.site_number)
    location_dfs = [df[df.site_number .== location, :] for location in locations]

    growth_rates = []
    for location_df in location_dfs
        
        # Filter out rows where adult sea lice is 0
        # Without: 0.002
        # With: 0.10
        # location_df = filter(row -> row.adult_sealice > 0, location_df)

        # Add epsilon to the adult sea lice count: 0.087
        # location_df.adult_sealice = location_df.adult_sealice .+ 1e-1

        # Run optimization: # Initial guess: r = 0.1, σ = 0.1
        # result = optimize(p -> baretswatch_log_likelihood(p, location_df, space), [0.1, 0.1],
        #          lower=[-1.0, 1e-6], upper=[1.0, 5.0], autodiff = :forward)

        result = optimize(r -> baretswatch_log_likelihood(r, location_df, space), [0.0], autodiff = :forward)
        r_hat = Optim.minimizer(result)

        rounded_result = round(r_hat[1], digits=4)

        # println("MLE for growth rate at location $location: $rounded_result")
        push!(growth_rates, r_hat)
    end

    #println(growth_rates)
    println(mean(growth_rates))
end

# Define log likelihood function for MLE analysis
function baretswatch_log_likelihood(r, df, space="log-space-normal")

    use_consecutive_weeks = false
    # r = r[1]
    σ = 0.1

    log_likelihood = 0.0
    for t in 2:length(df.adult_sealice)
        if use_data_for_mle_treatment(df, t) && use_data_zeros(df, t)
            if use_consecutive_weeks && use_data_for_mle_week(df, t)
                log_likelihood += calculate_log_likelihood(r, df, t, σ, 1, space)
            elseif !use_consecutive_weeks
                # Decreases from 0.10 to 0.05
                week_delta = df.total_week[t] - df.total_week[t-1]
                log_likelihood += calculate_log_likelihood(r, df, t, σ, week_delta, space)
            end
        end
    end
    return -log_likelihood
end

function calculate_log_likelihood(r, df, t, σ, week_delta, space="log-space-normal")
    # Convert week_delta to Float64 to ensure consistent types
    week_delta = Float64(week_delta)
    
    if space == "log-space-normal"
        # Log-space normal model: log(N_k+1) ~ N(log(N_k) + r * week_delta, sd)
        # This models exponential growth in log space
        μ = log(df.adult_sealice[t-1]) + r[1] * week_delta
        return logpdf(Normal(μ, σ), log(df.adult_sealice[t]))
    elseif space == "raw-space-log-normal"
        # Raw-space log normal model: N_k+1 ~ LogNormal(log(N_k) + r * week_delta, sd)
        # This models multiplicative growth with log-normal errors
        μ = log(df.adult_sealice[t-1]) + r[1] * week_delta
        return logpdf(LogNormal(μ, σ), df.adult_sealice[t])
    elseif space == "raw-space-normal"
        # Raw-space normal model: N_k+1 ~ N(N_k * exp(r * week_delta), sd)
        # This models exponential growth with normal errors
        μ = df.adult_sealice[t-1] * exp(r[1] * week_delta)
        return logpdf(Normal(μ, σ), df.adult_sealice[t])
    else
        error("Invalid space type: $space. Must be one of: log-space-normal, raw-space-log-normal, raw-space-normal")
    end
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

function use_data_zeros(df, t)

    # If adult sea lice is 0, then don't count
    if df.adult_sealice[t] == 0 || df.adult_sealice[t-1] == 0
        return false
    else
        return true
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end