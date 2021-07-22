using NiSparseArrays
using Test, Random, LinearAlgebra, NiLang, ForwardDiff, SparseArrays
using NiLang.AD
@testset "NiSparseArrays.jl" begin
    include("linalg.jl") #add sparse multiplication 
end