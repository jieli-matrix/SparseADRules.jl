module NiSparseArrays

using Base: promote_eltype
using LinearAlgebra
using SparseArrays
using NiLang
using NiLang.AD 
using ChainRulesCore

include("compat.jl")
include("linalg.jl")
include("lowranksvd.jl")
include("srcutils.jl")
include("chainrules.jl")
end
