module NiSparseArrays

using LinearAlgebra: eltype, length
using LinearAlgebra, SparseArrays
using NiLang
using NiLang.AD 

include("linalg.jl")
include("sparsegrad.jl")
end
