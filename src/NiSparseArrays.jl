module NiSparseArrays

using LinearAlgebra, SparseArrays
using NiLang
using NiLang.AD 

include("linalg.jl")
include("sparsegrad.jl")
end
