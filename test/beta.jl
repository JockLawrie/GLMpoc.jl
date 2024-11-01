@testset "Beta" begin

# From https://github.com/ararslan/BetaRegression.jl/blob/main/test/runtests.jl

expenditure = [15.998 62.476 1
16.652 82.304 5
21.741 74.679 3
 7.431 39.151 3
10.481 64.724 5
13.548 36.786 3
23.256 83.052 4
17.976 86.935 1
14.161 88.233 2
 8.825 38.695 2
14.184 73.831 7
19.604 77.122 3
13.728 45.519 2
21.141 82.251 2
17.446 59.862 3
 9.629 26.563 3
14.005 61.818 2
 9.160 29.682 1
18.831 50.825 5
 7.641 71.062 4
13.882 41.990 4
 9.670 37.324 3
21.604 86.352 5
10.866 45.506 2
28.980 69.929 6
10.882 61.041 2
18.561 82.469 1
11.629 44.208 2
18.067 49.467 5
14.539 25.905 5
19.192 79.178 5
25.918 75.811 3
28.833 82.718 6
15.869 48.311 4
14.910 42.494 5
 9.550 40.573 4
23.066 44.872 6
14.751 27.167 7]

n  = size(expenditure, 1)
y  = expenditure[:, 1] ./ expenditure[:, 2];
X  = hcat(fill(1.0, n), expenditure[:, 2:3])
Xs = [X, reshape(view(X, :, 1), n, 1)]
w  = nothing

# Config
yname  = "foodincome"
xnames = [["intercept", "income", "persons"], ["intercept"]]
links  = (LogitLink(), LogLink())  # link_mean, link_precision
cfg    = GLMconfig(yname, xnames, Beta(), links)

# Fit
opts   = (iterations=1000, g_abstol=1e-8)
fitted = fit(cfg, y, Xs, w, opts)
println(loglikelihood(fitted))
println(coef(fitted)[1])
println(coef(fitted)[2])
println(vcov(fitted))

end