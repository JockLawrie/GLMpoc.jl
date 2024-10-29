module GLMpoc

export LogitLink, LogLink

using Distributions
using Distributions: digamma, trigamma
using LinearAlgebra

# Core
include("linkfunctions.jl")
include("blockwise_coordinate_descent.jl"); using .blockwise_coord_descent
include("fit.jl")

# Response distributions
include("responsedistributions/beta.jl")
#include("responsedistributions/categorical.jl")

end
