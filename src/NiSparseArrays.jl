module NiSparseArrays

using LinearAlgebra, SparseArrays
using NiLang
using NiLang.AD 
using ChainRulesCore

include("compat.jl")
include("linalg.jl")
include("chainrules.jl")

end
