using ChainRulesCore: include
using SparseArraysAD
using Test, Random, LinearAlgebra, NiLang, SparseArrays
using NiLang.AD, ForwardDiff
using ChainRulesCore, ChainRulesTestUtils
import FiniteDifferences

using SparseArraysAD: Normal_QR
function ChainRulesTestUtils.test_approx(actual::Normal_QR, expected::Normal_QR, msg=""; kwargs...)
    ChainRulesTestUtils.test_approx(actual.Q, expected.Q, msg; kwargs...)
    ChainRulesTestUtils.test_approx(actual.R, expected.R, msg; kwargs...)
end

include("testutils.jl")

@testset "SparseArraysAD.jl" begin
    include("linalg.jl")
    include("chainrules.jl")
    include("lowranksvd.jl")
end
