initθ(d::Beta, Xs) = [[-0.62255, -0.01230, 0.11846], [log(35.0)]]  # Hard-coded for POC

function construct_distribution(d::Beta, prms)
    μ, ϕ = prms  # mean = a/(a+b), precision = a + b
    a = μ*ϕ
    Beta(μ*ϕ, ϕ - a)
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

function logpdf_secondderiv(d::Beta, y, prms, blocknumber)
    μ, ϕ  = prms
    μta   = μ * trigamma(μ*ϕ)
    omμtb = (1.0 - μ) * trigamma(ϕ - μ*ϕ)
    blocknumber == 1 ? omμtb - ϕ*μta : trigamma(ϕ) - μ*μta - (1.0 - μ)*omμtb
end
