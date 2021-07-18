module NiSparseArrays

using LinearAlgebra: eltype
using LinearAlgebra
using NiLang
using SparseArrays: SparseMatrixCSC, AbstractSparseMatrix, nonzeros, rowvals, nzrange

include("linalg.jl")

end
