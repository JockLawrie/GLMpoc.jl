#=
link(g, θ)       = g(θ)
invlink(g, eta)  = g^(-1)(eta) 
linkderiv1(g, θ) = g'(θ)
linkderiv2over1(g, θ, firstderiv) = g''(θ)/g'(θ)  # firstderiv included as arg because this is often used
=#

struct LogitLink end
link(g::LogitLink, θ)       = log(θ/(1.0 - θ))
invlink(g::LogitLink, eta)  = 1.0 / (1.0 + exp(-eta))
linkderiv1(g::LogitLink, θ) = 1.0 / (θ - θ*θ)
linkderiv2over1(g::LogitLink, θ, firstderiv) = 2.0*θ*firstderiv - firstderiv  # 1.0/(1.0 - θ) - 1.0/θ

struct LogLink end
link(g::LogLink, θ)       = log(θ)
invlink(g::LogLink, eta)  = exp(eta)
linkderiv1(g::LogLink, θ) = 1.0/θ
linkderiv2over1(g::LogLink, θ, firstderiv) = -firstderiv