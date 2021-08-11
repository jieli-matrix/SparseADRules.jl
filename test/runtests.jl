using NiSparseArrays
using Test, Random, LinearAlgebra, NiLang, SparseArrays
using NiLang.AD, ForwardDiff
using ChainRulesCore, ChainRulesTestUtils
import FiniteDifferences

include("testutils.jl")

@testset "NiSparseArrays.jl" begin
    include("linalg.jl")
    include("utils.jl") 
    include("jacobian.jl")
    include("chainrules.jl")
end
