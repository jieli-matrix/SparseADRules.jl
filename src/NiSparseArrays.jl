module NiSparseArrays

using Base: promote_eltype
using LinearAlgebra
using SparseArrays
using NiLang
using NiLang.AD 

include("compat.jl")
include("linalg.jl")
end
