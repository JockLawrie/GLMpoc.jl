module GLMpoc

export GLMconfig, GLMfitted,  # Types
       LogitLink, LogLink,    # Link functions
       fit, coef, loglikelihood, nobs, vcov  # StatsAPI

using Dates
using Distributions
using Distributions: digamma, trigamma
using LinearAlgebra
using Statistics
using StatsAPI

# API
include("api.jl")

# Fit
include("fit/check_input_data.jl")
include("fit/linkfunctions.jl")
include("fit/blockwise_coordinate_descent.jl"); using .blockwise_coord_descent
include("fit/fit.jl")

# Response distributions
include("responsedistributions/beta.jl")
include("responsedistributions/categorical.jl")

end
