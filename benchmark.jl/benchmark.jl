using SparseArrays

using NiSparseArrays:imul_v1!, imul_v2!

A = sprand(50,100,0.2);

b = rand(50,50);

C = zeros(100,50);

using BenchmarkTools

@btime A'*b
@btime imul_v1!(C, A', b, 1.0, 1.0)
@btime imul_v2!(C, A', b, 1.0, 1.0)