################################################################################
# Fit

function fit(d, links, w, y, Xs)
    θ0      = initθ(d, Xs)
    cache   = initcache(d, links, y, Xs)
    opts    = (iterations=250, g_abstol=1e-8)  # Use the same terminology as Optim.jl
    loss, θ = blockwise_coordinate_descent(θ0, loss!, block_gradient!, block_hessian!, y, Xs, w, opts, cache)
    loss, θ
end

#fit(d::Categorical, links, w, y, Xs) = TODO...

################################################################################
# Init θ and cache 

"""
Default initial value for θ given the response distribution d,
namely a list of vectors filled with 0s.
Can be overidden for specific distributions.
"""
function initθ(d, Xs)
    θ = Vector{typeof(0.0)}[]
    for X in Xs
        p = size(X, 2)
        push!(θ, fill(0.0, p))
    end
    θ
end

function initcache(d, links, y, Xs)
    n      = length(y)
    pmax   = maximum(size(X, 2) for X in Xs)
    prms   = fill(0.0, n, length(Xs))
    wwgrad = fill(0.0, n)        # Working weights for the gradient: gradient = transpose(X)*wwgrad
    wwhess = fill(0.0, n)        # Working weights for the hessian:  hessian  = transpose(X)*Diagonal(wwhess)*X
    XtW    = fill(0.0, pmax, n)  # Space to hold transpose(X)*Diagonal(wwhess) when computing the hessian
    (d=d, links=links, prms=prms, wwgrad=wwgrad, wwhess=wwhess, XtW=XtW)
end

################################################################################
# Loss function

function loss!(y, Xs, w, θ, cache)
    update_prms!(cache, Xs, θ)
    result = 0.0
    d      = cache.d
    prms   = cache.prms
    if isnothing(w)
        for (i, yi) in enumerate(y)
            di = construct_distribution(d, view(prms, i, :))
            result -= logpdf(di, yi)
        end
    else
        for (i, yi) in enumerate(y)
            di = construct_distribution(d, view(prms, i, :))
            result -= w[i]*logpdf(di, yi)
        end
    end
    result
end

"Update the parameters of cache.d"
function update_prms!(cache, Xs, θ)
    links = cache.links
    prms  = cache.prms
    for (blocknumber, X) in enumerate(Xs)
        lnk = links[blocknumber]
        vw  = view(prms, :, blocknumber)
        mul!(vw, X, θ[blocknumber])  # Update eta
        for (i, eta) in enumerate(vw)
            vw[i] = invlink(lnk, eta)  # Transform to parameter by applying the inverse link function
        end
    end
    nothing
end

################################################################################
# Gradient and hessian

function block_gradient!(g, blocknumber, y, Xs, w, θ, cache)
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

"Update the working weights for the block"
function update_working_weights!(cache, blocknumber, y, w)
    d, links, prms, wwgrad, wwhess, XtW = cache
    if isnothing(w)
        for (i, yi) in enumerate(y)
            wwg, wwh = calculate_working_weights(d, links, yi, view(prms, i, :), blocknumber)
            wwgrad[i] = wwg
            wwhess[i] = wwh
        end
    else
        for (i, yi) in enumerate(y)
            wwg, wwh = calculate_working_weights(d, links, yi, view(prms, i, :), blocknumber)
            wwgrad[i] = w[i] * wwg
            wwhess[i] = w[i] * wwh
        end
    end
    nothing
end

"""
Calculate the gradient and hessian working weights for 1 observation.
Note: The -1 multiplier at the end is because the loss function is -loglikelihood.
"""
function calculate_working_weights(d, links, y, prms, blocknumber)
    lnk  = links[blocknumber]
    θ    = prms[blocknumber]
    d1LL = logpdf_firstderiv(d, y, prms, blocknumber)
    d2LL = logpdf_secondderiv(d, y, prms, blocknumber)
    dg1  = linkderiv1(lnk, θ)
    dg21 = linkderiv2over1(lnk, θ, dg1)
    -d1LL/dg1, (d1LL*dg21 - d2LL)/(dg1*dg1)  # wwgrad, wwhess
end
