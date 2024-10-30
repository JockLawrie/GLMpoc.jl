# GLMpoc

This repo contains a proof-of-concept for fitting GLMs with multiple parameters that each depend on covariates.

For a given model, each parameter depends on its own set of covariates.
The sets of covariates may or may not overlap; they can be the same too.

The test cases include:
1. Multinomial regression (each category can have a different set of predictors)
2. Beta regression, with both the mean and precision depending on covariates

For now, this repo contains only 1 `fit` method:

```julia
loss, coefs = GLMpoc.fit(distribution, linkfuncs, w, y, Xs, opts)
```

- `distribution`: A distribution from Distributions.jl
- `linkfuncs`:    A tuple of link functions, one for each parameter that depends on covariates
- `w`: A weight vector.
- `y`: The response vector.
- `Xs`: A vector of predictor matrices (`AbstractMatrix`), 1 for each parameter that depends on covariates. Can be a mix of matrices and views.
- `opts`: Solver options. For now a subset of `Optim.Options` from the `Optim` package.

The solver is a block-wise cyclic coordinate descent with a basic line search.
