# AquaOpt: Sea Lice Management Optimization

A Julia-based project for optimizing sea lice management strategies in aquaculture using various reinforcement learning and control algorithms.

## Project Structure

```
AquaOpt/
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data files
├── results/
│   ├── figures/               # Generated plots and visualizations
│   ├── policies/              # Saved policy files
│   ├── data/                  # Simulation results
│   └── sensitivity_analysis/  # Sensitivity analysis results
├── src/
│   ├── AquaOpt.jl             # Main module entry point
│   ├── Models/                # POMDP model definitions
│   │   ├── SeaLicePOMDP.jl    # Base POMDP model
│   │   ├── SeaLiceLogPOMDP.jl # Log-space POMDP model
│   │   ├── SimulationPOMDP.jl # Simulation environment
│   │   ├── SimulationLogPOMDP.jl # Log-space simulation
│   │   └── KalmanFilter.jl    # State estimation filters
│   ├── Algorithms/            # Algorithm implementations
│   │   ├── Policies.jl        # Policy implementations
│   │   ├── Solvers.jl         # POMDP solvers
│   │   ├── Simulation.jl      # Simulation engine
│   │   ├── Evaluation.jl      # Policy evaluation
│   │   ├── Optimization.jl    # High-level optimization
│   │   └── SensitivityAnalysis.jl # Sensitivity analysis
│   ├── Plotting/              # Visualization modules
│   │   ├── TimeSeries.jl      # Time series plots
│   │   ├── Comparison.jl      # Policy comparison plots
│   │   ├── Heatmaps.jl        # Heatmap visualizations
│   │   └── Convergence.jl     # Convergence analysis plots
│   ├── Data/                  # Data handling
│   │   ├── Loading.jl         # Data loading utilities
│   │   └── Cleaning.jl        # Data preprocessing
│   ├── Utils/                 # Utility modules
│   │   ├── Config.jl          # Configuration management
│   │   ├── Logging.jl         # Logging utilities
│   │   └── Validation.jl      # Input validation
│   └── MLE/                   # Maximum Likelihood Estimation
├── scripts/
│   └── main.jl                # Main execution script
├── tests/                     # Test suite
├── Project.toml               # Package dependencies
└── Manifest.toml              # Dependency lock file
```

## Overview

This project implements various algorithms for optimizing sea lice management in aquaculture facilities. It uses Partially Observable Markov Decision Processes (POMDPs) to model the sea lice population dynamics and treatment decisions.

## Key Components

### 1. Models (`src/Models/`)
- **SeaLicePOMDP.jl**: Base POMDP model for sea lice management
- **SeaLiceLogPOMDP.jl**: Log-space POMDP model for better numerical stability
- **SimulationPOMDP.jl**: Simulation environment for policy testing
- **KalmanFilter.jl**: State estimation using EKF and UKF

### 2. Algorithms (`src/Algorithms/`)
The project implements several algorithms for policy optimization:

1. **Value Iteration (VI)**
   - Classic dynamic programming approach
   - Solves the underlying MDP

2. **SARSOP**
   - Point-based POMDP solver
   - Efficient for large state spaces

3. **QMDP**
   - Approximate POMDP solver
   - Assumes full observability for next step

4. **Heuristic Policy**
   - Simple threshold-based policy
   - Uses belief state for decision making

5. **Random Policy**
   - Baseline policy for comparison

### 3. Plotting (`src/Plotting/`)
- **TimeSeries.jl**: Sea lice levels and costs over time
- **Comparison.jl**: Policy comparison and Pareto frontiers
- **Heatmaps.jl**: Treatment decision heatmaps
- **Convergence.jl**: Algorithm convergence analysis

### 4. Data Handling (`src/Data/`)
- **Loading.jl**: Data loading and preprocessing utilities
- **Cleaning.jl**: Data cleaning and transformation functions

### 5. Utilities (`src/Utils/`)
- **Config.jl**: Configuration management and validation
- **Logging.jl**: Structured logging and progress tracking
- **Validation.jl**: Input validation and error handling

## Usage

### Basic Usage

Run the main optimization script:
```julia
julia scripts/main.jl
```

### Sensitivity Analysis

Run sensitivity analysis across multiple parameters:
```julia
julia -e "using AquaOpt; main_sensitivity()"
```

### Custom Configuration

Create a custom configuration:
```julia
using AquaOpt

# Define algorithms
algorithms = [
    Algorithm(solver_name="Heuristic_Policy"),
    Algorithm(solver=ValueIterationSolver(max_iterations=30), solver_name="VI_Policy")
]

# Define configuration
config = Config(num_episodes=1000, steps_per_episode=52)
pomdp_config = POMDPConfig(log_space=true)

# Run optimization
results = test_optimizer(algorithms[1], config, pomdp_config)
```

## Configuration

### Main Configuration (`Config`)
- `lambda_values`: Trade-off between treatment cost and sea lice levels
- `num_episodes`: Number of episodes for policy evaluation
- `steps_per_episode`: Simulation horizon
- `heuristic_threshold`: Threshold for heuristic policy
- `process_noise`: Process noise for simulation
- `observation_noise`: Observation noise for simulation

### POMDP Configuration (`POMDPConfig`)
- `costOfTreatment`: Cost of treatment action
- `growthRate`: Sea lice growth rate
- `rho`: Treatment effectiveness
- `discount_factor`: Discount factor for future rewards
- `log_space`: Whether to use log-space model

## Results

Results are automatically saved in the `results/` directory:

- **Policies**: Saved as JLD2 files in `results/policies/`
- **Simulation Data**: Saved in `results/data/`
- **Plots**: Generated in `results/figures/`
- **Sensitivity Analysis**: Results in `results/sensitivity_analysis/`

## Dependencies

### Core Dependencies
- **POMDPs.jl**: POMDP framework
- **POMDPTools**: POMDP utilities and tools
- **POMDPModels**: POMDP model implementations
- **DiscreteValueIteration**: Value iteration solver
- **NativeSARSOP**: SARSOP solver
- **QMDP**: QMDP solver

### Data and Visualization
- **DataFrames**: Data manipulation
- **Plots**: Plotting framework
- **JLD2**: Data serialization
- **CSV**: CSV file handling

### Utilities
- **Parameters**: Parameterized structs
- **GaussianFilters**: Kalman filter implementations
- **Distributions**: Probability distributions
- **Statistics**: Statistical functions

```