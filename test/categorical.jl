@testset "Categorical" begin

# Data
iris = dataset("datasets", "iris")
iris[!, "intercept"] = fill(1.0, nrow(iris))
y  = [Int(x.ref) for x in iris.Species]
X  = Matrix(iris[:, ["intercept", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]])
Xs = [view(X, :, :), view(X, :, :)]  # Same predictors for both blocks
w  = nothing

# Config
yname   = "Species"
ylevels = ["setosa", "versicolor", "virginica"]
xnames  = ["intercept", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
xnames  = [xnames, xnames]  # Same predictors for both blocks
cfg     = GLMconfig(yname, ylevels, xnames, Categorical(3), (LogitLink(), LogitLink()))

# Fit
opts    = (iterations=250, g_abstol=1e-8)
fitted  = GLMpoc.fit(cfg, y, Xs, w, opts)
println(loglikelihood(fitted))
println(coef(fitted)[1])
println(coef(fitted)[2])

end