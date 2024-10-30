###############################################################################
# Required methods

construct_distribution(d::Categorical, prms) = Categorical(prms...)

"Differs from the default method only by adding a column to prms to hold Pr(reference category)"
function initcache(d::Categorical, links, y, Xs)
    n      = length(y)
    pmax   = maximum(size(X, 2) for X in Xs)
    prms   = fill(0.0, n, 1 + length(Xs))  # Contains probabilities. +1 for the reference category
    wwgrad = fill(0.0, n)        # Working weights for the gradient: gradient = transpose(X)*wwgrad
    wwhess = fill(0.0, n)        # Working weights for the hessian:  hessian  = transpose(X)*Diagonal(wwhess)*X
    XtW    = fill(0.0, pmax, n)  # Space to hold transpose(X)*Diagonal(wwhess) when computing the hessian
    (d=d, links=links, prms=prms, wwgrad=wwgrad, wwhess=wwhess, XtW=XtW)
end

###############################################################################
# Optional (override default methods defined in fit.jl)

function update_prms!(d::Categorical, cache, Xs, coefs)
    # Set eta = X*coefs
    prms = cache.prms
    fill!(view(prms, :, 1), 0.0)
    for (blocknumber, b) in enumerate(coefs)
        mul!(view(prms, :, 1 + blocknumber), Xs[blocknumber], b)
    end
    # Transform eta into probs
    n = size(prms, 1)
    for i = 1:n
        softmax!(view(prms, i, :))
    end
end

function softmax!(eta::AbstractVector)
    max_bx = -Inf
    for x in eta
        max_bx = x > max_bx ? x : max_bx
    end
    psum = 0.0
    @inbounds for (i, x) in enumerate(eta)
        eta[i] = exp(x - max_bx)
        psum += eta[i]
    end
    denom = 1.0/psum
    rmul!(eta, denom)
end

function calculate_working_weights(d::Categorical, links, y, prms, blocknumber)
    k      = blocknumber + 1  # The category that blocknumber refers to 
    probk  = prms[k]
    wwgrad = y == k ? (probk - 1.0) : probk
    wwhess = max(sqrt(eps()), probk - probk*probk)
    wwgrad, wwhess
end