function check_input_data(y, Xs, w)
    check_nobs_is_consistent(y, Xs, w)  # Raises an error
    warn_response_variable(y)
    for (blocknumber, X) in enumerate(Xs)
        warn_univariate_issues(X, blocknumber)
        warn_high_bivariate_correlations(X, blocknumber)
        warn_multivariate_correlations(X, blocknumber)
    end
    nothing
end

function check_nobs_is_consistent(y, Xs, w)
    n = length(y)
    !isnothing(w) && length(w) != n && error("y has $(n) observations, but w has length $(length(w))")
    for (k, X) in enumerate(Xs)
        size(X, 1) != n && error("y has $(n) observations, but Xs[$(k)] has $(size(X, 1)) rows")
    end
end

function warn_response_variable(y)
    length(unique(y)) == 1 && @warn "$(now()) Response variable has only 1 unique non-missing value"
end

function warn_univariate_issues(X, blocknumber)
    nconstant = 0  # Running count of constant predictors
    nj = size(X, 2)
    for j = 1:nj
        x = view(X, :, j)
        if length(unique(x)) == 1  # Predictor is constant
            nconstant += 1
            nconstant >= 2 && @warn "$(now()) Predictor $(j) in block $(blocknumber) has only 1 unique non-missing value. Remove from the model."
        else  # Predictor is not constant
            m  = mean(x)
            s  = std(x)
            cv = s / m
            if abs(cv) < 0.1
                msg  = "$(now()) Predictor $(j) has a small coefficient of variation: abs(std/mean) = $(cv)"
                msg *= "\n         Consider recentering the predictor around its mean ($(m)) or thereabouts."
                @warn msg
            end
            if s > 5
                msg  = "$(now()) Predictor $(j) has a high std (> 5): std = $(s)"
                msg *= "\n         Consider rescaling the predictor, especially if your model involves exponentiation."
                @warn msg
            end
        end
    end
    nothing
end

function warn_high_bivariate_correlations(X, blocknumber)
    size(X, 2) == 1 && return nothing
    M  = cor(X)
    nj = size(M, 2)
    highcor = Tuple{Int, Int, typeof(0.0)}[]  # (i,j,cor(Xi,Xj))
    for j = 1:nj
        for i = (j+1):nj
            c = M[i, j]
            abs(c) > 0.95 && push!(highcor, (i, j, c))
        end
    end
    if !isempty(highcor)
        for t in highcor
            @warn "$(now()) High correlation in block $(blocknumber): cor(x$(t[1]), x$(t[2])) = $(t[3])"
        end
    end
    nothing
end

function warn_multivariate_correlations(X, blocknumber)
    nj = size(X, 2)
    nj == 1 && return nothing
    r  = rank(X)
    c  = cond(X)
    r < size(X, 2) && @warn "$(now()) The predictor matrix for block $(blocknumber) has $(nj) columns but rank $(r)"
    c > 1000 && @warn "$(now()) The predictor matrix for block $(blocknumber) has a high condition number, $(c), indicating multicollinearlity"
    nothing
end
