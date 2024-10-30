# GLMpoc

This repo contains a proof-of-concept for fitting GLMs with multiple parameters that each depend on covariates.

For a given model, each parameter depends on its own set of covariates.
The sets of covariates may or may not overlap; they can be the same too.

The test cases include:
1. Multinomial regression (each category can have a different set of predictors)
2. Beta regression, with both the mean and precision depending on covariates
