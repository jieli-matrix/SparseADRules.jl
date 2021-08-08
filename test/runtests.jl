using NiSparseArrays
using Test, Random, LinearAlgebra, NiLang, SparseArrays
using NiLang.AD, ForwardDiff

@testset "NiSparseArrays.jl" begin
    include("linalg.jl")
    include("utils.jl") 
    include("jacobian.jl")
end