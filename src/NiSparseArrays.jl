module NiSparseArrays

using LinearAlgebra, SparseArrays
using NiLang, ForwardDiff
using NiLang.AD 

include("linalg.jl")
include("sparsegrad.jl")
end
