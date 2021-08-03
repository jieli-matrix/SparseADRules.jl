using NiSparseArrays
using Test, Random, LinearAlgebra, NiLang, ForwardDiff, SparseArrays
using NiLang.AD
using BenchmarkTools
using BenchmarkPlots, StatsPlots
@testset "NiSparseArrays.jl" begin
    include("utils.jl")
    include("linalg.jl") #add sparse multiplication 
end