################################################################################
# Fit

"Returns an instance of GLMfitted"
function fit(cfg::GLMconfig, y, Xs, w, opts=nothing)
    check_input_data(y, Xs, w)
    coefs = initcoefs(cfg, y, Xs, w)
    cache = initcache(cfg, y, Xs)
    opts  = isnothing(opts) ? defaultopts() : merge(defaultopts(), opts)
    loss, coefs = blockwise_coordinate_descent(coefs, loss!, block_gradient!, block_hessian!, y, Xs, w, opts, cache)
    nobs = isnothing(w) ? length(y) : sum(w)
    vcov = construct_vcov(cache.d, cache, coefs, y, Xs, w)
    GLMfitted(cfg, coefs, nobs, -loss, vcov)
end

defaultopts() = (iterations=500, g_abstol=1e-8)  # Use the same terminology as Optim.jl

################################################################################
# Init θ and cache 

"""
Default initial value for coefs given the response distribution d,
namely a list of vectors filled with zeros.
Can be overridden for specific distributions.
"""
function initcoefs(cfg::GLMconfig, y, Xs, w)
    result = Vector{typeof(0.0)}[]
    for X in Xs
        p = size(X, 2)
        push!(result, fill(0.0, p))
    end
    result
end

function initcache(cfg, y, Xs)
    n      = length(y)
    pmax   = maximum(size(X, 2) for X in Xs)
    prms   = fill(0.0, n, length(Xs))
    wwgrad = fill(0.0, n)        # Working weights for the gradient: gradient = transpose(X)*wwgrad
    wwhess = fill(0.0, n)        # Working weights for the hessian:  hessian  = transpose(X)*Diagonal(wwhess)*X
    XtW    = fill(0.0, pmax, n)  # Space to hold transpose(X)*Diagonal(wwhess) when computing the hessian
    (d=cfg.distribution, links=cfg.linkfunctions, prms=prms, wwgrad=wwgrad, wwhess=wwhess, XtW=XtW)
end

################################################################################
# Loss function

function loss!(y, Xs, w, coefs, cache)
    d    = cache.d
    prms = cache.prms
    update_eta!(prms, Xs, coefs)
    eta_to_prms!(d, prms, cache.links)
    result = 0.0
    if isnothing(w)
        for (i, yi) in enumerate(y)
            result -= loglikelihood(d, yi, view(prms, i, :))
        end
    else
        for (i, yi) in enumerate(y)
            result -= w[i] * loglikelihood(d, yi, view(prms, i, :))
        end
    end
    result
end

"Set cache.prms = eta = X*coefs"
function update_eta!(prms, Xs, coefs)
    for (blocknumber, X) in enumerate(Xs)
        mul!(view(prms, :, blocknumber), X, coefs[blocknumber])
    end
    nothing
end

"For each observation, update the estimated parameters of the response distribution d."
function eta_to_prms!(d, prms, links)
    for (blocknumber, lnk) in enumerate(links)
        vw = view(prms, :, blocknumber)
        for (i, eta) in enumerate(vw)
            vw[i] = invlink(lnk, eta)  # Transform to parameter by applying the inverse link function
        end
    end
    nothing
end

################################################################################
# Gradient and hessian

function block_gradient!(g, blocknumber, y, Xs, w, coefs, cache)
    update_working_weights!(cache, blocknumber, y, w)
    mul!(g[blocknumber], transpose(Xs[blocknumber]), cache.wwgrad)  # g = transpose(X)*wwgrad
end

function block_hessian!(H, blocknumber, Xs, w, cache)
    X    = Xs[blocknumber]
    n, p = size(X)
    XtW  = view(cache.XtW, 1:p, 1:n)
    mul!(XtW, transpose(X), Diagonal(cache.wwhess))  # W = Diagonal(wwhess), with wwhess updated during block_gradient!.
    mul!(H[blocknumber], XtW, X)  # H = XtWX
    nothing
end

#################################################################################################

apply_observation_weights!(a, w) = a .*= w
apply_observation_weights!(a, w::Nothing) = a

"Update the working weights for the block"
function update_working_weights!(cache, blocknumber, y, w, expected=true)
    d, links, prms, wwgrad, wwhess, XtW = cache
    for (i, yi) in enumerate(y)
        wwg, wwh = calculate_working_weights(d, links, yi, view(prms, i, :), blocknumber, expected)
        wwgrad[i] = wwg
        wwhess[i] = wwh
    end
    apply_observation_weights!(wwgrad, w)
    apply_observation_weights!(wwhess, w)
    nothing
end

"""
Calculate the working weights for gradient(-LL) and hessian(-LL) for 1 observation.

If `expected` is true, the logpdf_firstderiv term in the hessian working weights is set to 0,
since the expected value of this term (with respect to the response y) is 0 for each individual.

The default value of `expected` is true because it doesn't seem to adversely affect the search direction
during the fitting process, and is used to calculate vcov(coefs) after fitting (via the Fisher Information matrix).
"""
function calculate_working_weights(d, links, y, prms, blocknumber, expected=true)
    lnk  = links[blocknumber]
    θ    = prms[blocknumber]
    d1LL = logpdf_firstderiv(d, y, prms, blocknumber)
    d2LL = logpdf_secondderiv(d, y, prms, blocknumber)
    d1g  = linkderiv1(lnk, θ)
    wwg  = -d1LL/d1g
    if expected
        wwh  = -d2LL/(d1g*d1g)
    else
        d21g = linkderiv2over1(lnk, θ, d1g)
        wwh  = (d1LL*d21g - d2LL)/(d1g*d1g)
    end
    wwg, wwh
end

#################################################################################################

function construct_vcov(d, cache, coefs, y, Xs, w)
    # Update prms
    update_eta!(cache.prms, Xs, coefs)
    eta_to_prms!(d, cache.prms, cache.links)

    # Compute expected hessian (which is block diagonal)
    H = [fill(0.0, length(b), length(b)) for b in coefs]
    for (blocknumber, b) in enumerate(coefs)
        update_working_weights!(cache, blocknumber, y, w, true)
        block_hessian!(H, blocknumber, Xs, w, cache)
    end

    # Construct vcov from hessian (which is also block diagonal)
    p      = sum(length(b) for b in coefs)
    result = fill(0.0, p, p)
    psum   = 0
    for Hb in H
        pb    = size(Hb, 1)
        vw    = view(result, (psum+1):(psum+pb), (psum+1):(psum+pb))
        vw   .= inv(Hb)
        psum += pb
    end
    result
end