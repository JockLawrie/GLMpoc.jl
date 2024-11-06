#=
Categorical regression with 3 or more categories works a bit differently to the other
GLMs. Like the other GLMs, the parameters of the response distribution (namely the
category probabilities) define the blocks when setting up the solver.

The difference is that the coefficients of a given block appear not only in the parameter 
that corresponds to the block, but also in the parameters of every other block.

Fortunately this only requires a different calculation of the eta_to_prms! and
calculate_working_weights functions, and not a completely different approach.
=#

###############################################################################
# Required methods
# Note: logpdf_firstderiv and logpdf_secondderiv not needed,
#       see the note accompanying the calculate_working_weights calculation below.

# prms = probs[2:ncategories]
loglikelihood(d::Categorical, y, prms) = y == 1 ? log(1.0 - sum(prms)) : log(prms[y - 1])

###############################################################################
# Override the default eta_to_prms! method defined in fit.jl

function eta_to_prms!(d::Categorical, prms, links)
    n = size(prms, 1)
    for i = 1:n
        eta_to_probs!(view(prms, i, :))
    end
    nothing
end

"""
Transform eta to [Pr(Category 2), ..., Pr(Category ncategories)].
Pr(Category 1) = 1 - sum(probs), because Category 1 is the reference category.
"""
function eta_to_probs!(eta::AbstractVector)
    max_bx = 0.0  # eta(reference_category)
    for x in eta
        max_bx = x > max_bx ? x : max_bx
    end
    denom = exp(-max_bx)  # exp(eta(reference_category))
    @inbounds for (i, x) in enumerate(eta)
        eta[i] = exp(x - max_bx)
        denom += eta[i]
    end
    m = 1.0/denom
    rmul!(eta, m)
end

###############################################################################
# Override the default calculate_working_weights method defined in fit.jl
# Note: This removes the need to define methods for logpdf_firstderiv and logpdf_secondderiv,
#       since these are only used in the default calculate_working_weights method.

function calculate_working_weights(d::Categorical, links, y, prms, blocknumber, expected=true)
    k      = blocknumber + 1    # The category that blocknumber refers to 
    probk  = prms[blocknumber]  # Pr(Category k), k = 2:ncategories (prms = probs[2:ncategories])
    wwgrad = y == k ? (probk - 1.0) : probk
    wwhess = max(sqrt(eps()), probk - probk*probk)
    wwgrad, wwhess
end

###############################################################################
# Override the default construct_vcov method defined in fit.jl

"For d::Categorical, the hessian is not block diagonal"
function construct_vcov(d::Categorical, cache, coefs, y, Xs, w)
    update_eta!(cache.prms, Xs, coefs)
    eta_to_prms!(d, cache.prms, cache.links)
    H = hessian(d, cache, Xs[1], w)
#    Matrix(inv(H))
    Matrix(Hermitian(inv(bunchkaufman!(H))))
end

function hessian(d::Categorical, cache, X, w)
    wwhess = cache.wwhess
    prms = cache.prms
    km1  = size(prms, 2)
    n, p = size(X)
    H    = fill(0.0, p*km1, p*km1)
    Xt   = transpose(X)
    W    = Diagonal(wwhess)
    XtW  = cache.XtW
    del  = sqrt(eps())
    for j = 1:km1
        cols = (p*(j - 1) + 1):(p*j)
        for i = j:km1  # Update block (i, j)
            rows  = (p*(i - 1) + 1):(p*i)
            Hview = view(H, rows, cols)
            set_working_weights!(wwhess, prms, i, j, del)
            apply_observation_weights!(wwhess, w)
            mul!(XtW, Xt, W)
            mul!(Hview, XtW, X)  # Hview = XtWX
        end
    end
    Hermitian(H, :L)
end

"""
wwhess .= w .* Pi .* (delta_ij - Pj)
i, j refer to categories 2:ncategories.
"""
function set_working_weights!(wwhess, probs, i, j, del)
    Pi = view(probs, :, i)
    if i == j
        wwhess .= max.(del, Pi .* (1.0 .- Pi))
    else
        Pj = view(probs, :, j)
        wwhess .= min.(-del, -Pi .* Pj)
    end
end
