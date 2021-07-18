using NiSparseArrays
using Test
using LinearAlgebra

@testset "NiSparseArrays.jl" begin
    include("linalg.jl") #add sparse multiplication 
    #include("sparsegrad.jl") #add sparse gradient test
end