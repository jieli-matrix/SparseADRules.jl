using NiSparseArrays
using Test, Random, LinearAlgebra, NiLang, SparseArrays
using NiLang.AD, ForwardDiff
using ChainRulesCore, ChainRulesTestUtils
import FiniteDifferences

include("testutils.jl")

@testset "NiSparseArrays.jl" begin
    #include("linalg.jl")
    include("chainrules.jl")
end
