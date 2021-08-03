using SparseArrays: sprand
import SparseArrays
using BenchmarkTools
using BenchmarkPlots, StatsPlots
using NiSparseArrays:idotxA, idotAx

m, n = 1000, 500;
x = sprand(m, 0.2)
A = sprand(m, n, 0.2)
y = sprand(n, 0.3)

println("Benchmark on SparseArrays dot function: ")
sdf = @benchmarkable SparseArrays.dot($x, $A, $y)
run(sdf)

r = zero(eltype(A))
println("Benchmark on NiSparseArrays idotxA function: ")
xAf = @benchmarkable idotxA($copy(r), $x, $A, $y)
run(xAf)

r = zero(eltype(A))
println("Benchmark on NiSparseArrays idotAx function: ")
Axf = @benchmarkable idotAx($copy(r), $x, $A, $y)
run(Axf)





