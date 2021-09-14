module NiSparseArrays

using ChainRulesCore: include
using Base: promote_eltype
using LinearAlgebra
using SparseArrays
using NiLang
using NiLang.AD 
using ChainRulesCore

include("compat.jl")
include("linalg.jl")
include("chainrules.jl")
include("lowranksvd.jl")
end
