###############################################################################
# Required methods
# logpdf_firstderiv and logpdf_secondderiv are used in the calculate_working_weights function.

function loglikelihood(d::Beta, y, prms)
    μ, ϕ = prms
    a = μ*ϕ  # For Beta(a, b), we have mu = a/(a+b), phi = a + b
    logpdf(Beta(μ*ϕ, ϕ - a), y)
end

"""
First derivative of the loglikelihood for 1 observation.
If blocknumber is 1, return d_loglikelihood/d_μ, else d_loglikelihood/d_ϕ
"""
function logpdf_firstderiv(d::Beta, y, prms, blocknumber)
    μ, ϕ   = prms
    ystar  = log(y/(1.0 - y))
    dq     = digamma(ϕ - μ*ϕ)
    mustar = digamma(μ*ϕ) - dq
    blocknumber == 1 ? ϕ*(ystar - mustar) : μ*(ystar - mustar) + digamma(ϕ) - dq + log(1.0 - y)
end

"""
Second derivative of the loglikelihood for 1 observation.
If blocknumber is 1, return d^2_loglikelihood/d_μ^2, else d^2_loglikelihood/d_ϕ^2
"""
function logpdf_secondderiv(d::Beta, y, prms, blocknumber)
    μ, ϕ  = prms
    μta   = μ * trigamma(μ*ϕ)
    omμtb = (1.0 - μ) * trigamma(ϕ - μ*ϕ)
    blocknumber == 1 ? omμtb - ϕ*μta : trigamma(ϕ) - μ*μta - (1.0 - μ)*omμtb
end

###############################################################################
# Optional: override the default initcoefs method defined in fit.jl

# Needs cleaning up, but ok for POC.
function initcoefs(d::Beta, links, w, y, Xs)
    z   = [link(links[1], yi) for yi in y]
    X   = Xs[1]
    b   = inv(transpose(X)*X)*transpose(X)*z
    eta = X*b
    ms  = [invlink(links[1], η) for η in eta]  # mu
    rs  = z .- eta
    sse = dot(rs, rs)/(length(y) - 1)
    vs  = [sse/(linkderiv1(links[1], m)^2) for m in ms]  # sigma^2
    phi = sum(m*(1.0 - m)/v for (m, v) in zip(ms, vs))/length(y) - 1.0
    result = [b, fill(0.0, size(Xs[2], 2))]
    result[2][1] = link(links[2], phi)
    result
end
