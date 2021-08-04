using NiSparseArrays
using Test, Random, LinearAlgebra, NiLang, ForwardDiff, SparseArrays
using NiLang.AD
@testset "NiSparseArrays.jl" begin
    include("utils.jl")
    include("linalg.jl") #add sparse multiplication 
    include("jacobian.jl")
end