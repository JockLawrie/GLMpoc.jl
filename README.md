# GLMpoc

This repo contains a proof-of-concept for fitting GLMs with multiple parameters that each depend on covariates.

For a given model, each parameter of the response distribution depends on its own set of covariates.
The sets of covariates may or may not overlap; they can be the same too.

The test cases include:
1. Multinomial regression (each category can have a different set of predictors)
2. Beta regression, with both the mean and precision depending on covariates

For now, this repo contains only 1 `fit` method:

```julia
loss, coefs = GLMpoc.fit(distribution, linkfuncs, w, y, Xs, opts)
```

- `distribution`: The distribution of the response variable. Defined in Distributions.jl.
- `linkfuncs`:    A tuple of link functions, one for each parameter of the reponse distribution.
- `w`: A weight vector.
- `y`: The response vector.
- `Xs`: A vector of predictor matrices (`AbstractMatrix`), with 1 matrix for each parameter of the response distribution. Can be a mix of matrices and views.
- `opts`: Solver options. For now a subset of `Optim.Options` from the `Optim` package.

The solver is a block-wise cyclic coordinate descent with a basic line search.
There is one block for each parameter of the response distribution,
with a vector of coefficients estimated for each block.

Note that common GLMs can be estimated this way too.
The block for the mean has the usual list of coefficients.
If the response distribution has dispersion parameter not equal to 1,
then a second block is included containing a single coefficient for the constant term.

For example, in OLS the second block is used to estimate the standard deviation.
Its predictor matrix contains a single column of 1s and it uses the `LogLink` link function.