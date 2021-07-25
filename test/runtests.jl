using SparseArrays: spmatmul
using Base: Float64
using LinearAlgebra: include, eltype
using NiSparseArrays
using Test, Random, LinearAlgebra, NiLang, ForwardDiff, SparseArrays
using NiLang.AD
@testset "NiSparseArrays.jl" begin
    include("utils.jl")
    include("linalg.jl") #add sparse multiplication 
end