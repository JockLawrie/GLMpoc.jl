###############################################################################
# GLMconfig

struct GLMconfig{L,D,F}
    responsename::String
    responselevels::Vector{L}  # For categorical response variables
    coefnames::Vector{Vector{String}}
    weightname::String  # Empty string indicates unit weights
    distribution::D
    linkfunctions::F
end

# weightname defaults to ""
GLMconfig(yname, xnames, distribution, linkfunctions) = GLMconfig(yname, Nothing[], xnames, "", distribution, linkfunctions)

# responselevels defaults to Nothing[] for numeric response variables
GLMconfig(yname, xnames, wname::String, distribution, linkfunctions) = GLMconfig(yname, Nothing[], xnames, wname, distribution, linkfunctions)

GLMconfig(yname, ylevels, xnames, distribution, linkfunctions) = GLMconfig(yname, ylevels, xnames, "", distribution, linkfunctions)

###############################################################################
# GLMfitted

const FLOAT = typeof(0.0)

struct GLMfitted{L,D,F}  <: StatsAPI.RegressionModel
    config::GLMconfig{L,D,F}
    coefs::Vector{Vector{FLOAT}}
    nobs::Int
    loglikelihood::FLOAT
    vcov::Matrix{FLOAT}
end

###############################################################################
# API

import StatsAPI: fit, coef, loglikelihood, nobs, vcov

StatsAPI.coef(fitted::GLMfitted) = fitted.coefs
StatsAPI.loglikelihood(fitted::GLMfitted) = fitted.loglikelihood
StatsAPI.nobs(fitted::GLMfitted) = fitted.nobs
StatsAPI.vcov(fitted::GLMfitted) = fitted.vcov
