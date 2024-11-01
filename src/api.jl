const FLOAT = typeof(0.0)

import StatsAPI: coef, loglikelihood, nobs, vcov

struct GLMconfig{L,D,F}
    responsename::String
    responselevels::Vector{L}  # For categorical response variables
    coefnames::Vector{Vector{String}}
    distribution::D
    linkfunctions::F
end

"Set levels to nothing for numeric response variables"
GLMconfig(responsename, coefnames, distribution, linkfunctions) = GLMconfig(responsename, Nothing[], coefnames, distribution, linkfunctions)

struct GLMfitted{L,D,F}  <: StatsAPI.RegressionModel
    config::GLMconfig{L,D,F}
    coefs::Vector{Vector{FLOAT}}
    nobs::Int
    loglikelihood::FLOAT
    vcov::Matrix{FLOAT}
end

StatsAPI.coef(fitted::GLMfitted) = fitted.coefs
StatsAPI.loglikelihood(fitted::GLMfitted) = fitted.loglikelihood
StatsAPI.nobs(fitted::GLMfitted) = fitted.nobs
StatsAPI.vcov(fitted::GLMfitted) = fitted.vcov
