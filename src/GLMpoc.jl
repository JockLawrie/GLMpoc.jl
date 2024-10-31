module GLMpoc

export LogitLink, LogLink

using Dates
using Distributions
using Distributions: digamma, trigamma
using LinearAlgebra
using Statistics

# Core
include("check_input_data.jl")
include("linkfunctions.jl")
include("blockwise_coordinate_descent.jl"); using .blockwise_coord_descent
include("fit.jl")

# Response distributions
include("responsedistributions/beta.jl")
include("responsedistributions/categorical.jl")

end
