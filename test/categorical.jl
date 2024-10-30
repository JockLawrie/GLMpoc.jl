@testset "Categorical" begin

iris = dataset("datasets", "iris")
iris[!, "intercept"] = fill(1.0, nrow(iris))

w  = nothing
y  = [Int(x.ref) for x in iris.Species]
X  = Matrix(iris[:, ["intercept", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]])
Xs = [view(X, :, :), view(X, :, :)]
links   = (LogitLink(), LogitLink())
opts    = (iterations=250, g_abstol=1e-8)
loss, B = GLMpoc.fit(Categorical(3), links, w, y, Xs, opts)
println(loss)
println(B[1])
println(B[2])

end