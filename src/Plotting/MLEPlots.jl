using Logging
using DiscreteValueIteration
using GridInterpolations
using NativeSARSOP: SARSOPSolver
using POMDPs
using POMDPTools
using Plots
using StatsPlots
using LocalFunctionApproximation
using LocalApproximationValueIteration
using StatsPlots
using Optim
using DataFrames
using Statistics

plotlyjs()  # Activate Plotly backend

function plot_predictions(preds, actuals, space="log-space-normal")

    title_name = "Predicted vs Actual Lice Counts"
    if space == "log-space-normal"
        title_name = "Predicted vs Actual Lice Counts (Log Space)"
    elseif space == "raw-space-log-normal"
        title_name = "Predicted vs Actual Lice Counts (Raw Space Log-Normal)"
    elseif space == "raw-space-normal"
        title_name = "Predicted vs Actual Lice Counts (Raw Space)"
    end

    p = scatter(actuals, preds,
            xlabel="Actual Sea Lice Level",
            ylabel="Predicted Sea Lice Level",
            title="$title_name",
            label="Predicted",
            ylims=(0, 30),
            legend=:bottomright)
    plot!(x -> x, label="Perfect Prediction", linestyle=:dash)
    return p
end

function plot_growth_rates_by_year_and_location(location_year_to_growth_rate, df, space="log-space-normal")

    # Get unique years
    years = sort(unique(df.year))

    # Get unique locations
    locations = unique(df.site_number)

    p = plot(xlabel="Year", ylabel="Growth Rate", title="Growth Rate by Year and Location", legend=:topleft)

    # For each location, get the growth rate for each year
    for location in locations
        x = Int64[]
        y = Float64[]
        for year in years
            if (location, year) in keys(location_year_to_growth_rate)
                push!(x, year)
                push!(y, location_year_to_growth_rate[(location, year)])
            end
        end
        if length(x) > 0
            plot!(p, x, y, label="Location $location")
        end
    end

    # Save plot
    mkpath("results/figures/MLE/$space")
    savefig(p, "results/figures/MLE/$space/growth_rates_by_year_and_location.png")

    return p
end

function plot_mean_growth_rates_by_year_and_location(location_year_to_growth_rate, df, space="log-space-normal")

    # Get unique years and locations
    years = sort(unique(df.year))
    locations = unique(df.site_number)

    title_name = "Mean Growth Rate by Year with 95% CI"
    if space == "log-space-normal"
        title_name = "Mean Growth Rate by Year with 95% CI (Log Space)"
    elseif space == "raw-space-log-normal"
        title_name = "Mean Growth Rate by Year with 95% CI (Raw Space Log-Normal)"
    elseif space == "raw-space-normal"
        title_name = "Mean Growth Rate by Year with 95% CI (Raw Space)"
    end

    p = plot(xlabel="Year", ylabel="Growth Rate", title="$title_name", legend=:bottomright)

    x = Int64[]
    y = Float64[]
    lower_ci = Float64[]
    upper_ci = Float64[]

    for year in years
        push!(x, year)
        ys = Float64[]
        
        for location in locations
            if (location, year) in keys(location_year_to_growth_rate)
                push!(ys, location_year_to_growth_rate[(location, year)])
            end
        end
        
        mean_growth = mean(ys)
        push!(y, mean_growth)
        
        # Calculate 95% confidence interval
        n = length(ys)
        se = std(ys) / sqrt(n)
        t_value = quantile(TDist(n-1), 0.975)
        margin_error = t_value * se
        
        push!(lower_ci, mean_growth - margin_error)
        push!(upper_ci, mean_growth + margin_error)
    end

    plot!(p, x, y, ribbon=(y .- lower_ci, upper_ci .- y), fillalpha=0.3, label="Mean Growth Rate with 95% CI")

    # Save plot
    mkpath("results/figures/MLE/$space")
    savefig(p, "results/figures/MLE/$space/mean_growth_rates_by_year_and_location_with_ci_$(space).png")

    return p
end

function plot_epsilon_sensitivity_analysis(epsilon_values)
    # Initialize arrays to store metrics
    location_maes = Float64[]
    location_rmses = Float64[]
    location_r2s = Float64[]
    location_year_maes = Float64[]
    location_year_rmses = Float64[]
    location_year_r2s = Float64[]
    
    # Read results for each epsilon value
    for epsilon in epsilon_values
        # Read location results
        location_df = CSV.read("results/figures/MLE/epsilon_$(epsilon)_location_predictions.csv", DataFrame)
        push!(location_maes, location_df.mae)
        push!(location_rmses, location_df.rmse)
        push!(location_r2s, location_df.r2)
        
        # Read location-year results
        location_year_df = CSV.read("results/figures/MLE/epsilon_$(epsilon)_location_year_predictions.csv", DataFrame)
        push!(location_year_maes, location_year_df.mae[1])
        push!(location_year_rmses, location_year_df.rmse[1])
        push!(location_year_r2s, location_year_df.r2[1])
    end
    
    # Create plots
    p1 = plot(epsilon_values, [location_maes, location_year_maes],
             xscale=:log10,
             xlabel="Epsilon",
             ylabel="MAE",
             label=["By Location" "By Location-Year"],
             title="MAE vs Epsilon",
             legend=:topleft)
    
    p2 = plot(epsilon_values, [location_rmses, location_year_rmses],
             xscale=:log10,
             xlabel="Epsilon",
             ylabel="RMSE",
             label=["By Location" "By Location-Year"],
             title="RMSE vs Epsilon",
             legend=:topleft)
    
    p3 = plot(epsilon_values, [location_r2s, location_year_r2s],
             xscale=:log10,
             xlabel="Epsilon",
             ylabel="R²",
             label=["By Location" "By Location-Year"],
             title="R² vs Epsilon",
             legend=:topleft)
    
    # Combine plots
    p = plot(p1, p2, p3, layout=(3,1), size=(800,1200))
    
    # Save plot
    mkpath("results/figures/MLE")
    savefig(p, "results/figures/MLE/epsilon_sensitivity_analysis.png")
    
    return p
end