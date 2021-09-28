# NiSparseArrays

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jieli-matrix.github.io/NiSparseArrays.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jieli-matrix.github.io/NiSparseArrays.jl/dev)
[![Build Status](https://github.com/jieli-matrix/NiSparseArrays.jl/workflows/CI/badge.svg)](https://github.com/jieli-matrix/NiSparseArrays.jl/actions)
[![Coverage](https://codecov.io/gh/jieli-matrix/NiSparseArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jieli-matrix/NiSparseArrays.jl)

[中文版本](README_CN.md)

`NiSparseArrays` is a part of the [Summer 2021 of Open Source Promotion Plan](https://summer.iscas.ac.cn/#/?lang=en). It implements the backward rules for sparse matrix operations using [`NiLang`](https://giggleliu.github.io/NiLang.jl/dev/) and ports these rules to [`ChainRules`](https://github.com/JuliaDiff/ChainRules.jl).

## Background 

Sparse matrices are extensively used in scientific computing, however there is no automatic differentiation package in Julia yet to handle sparse matrix operations. This project utilizes the reversible embedded domain-specific language `NiLang.jl` to differentiate sparse matrix operations by writing the sparse matrix operations in a reversible style. The generated backward rules are ported to `ChainRules.jl` as an extension, so that one can access these features in an automatic differentiation package like [`Zygote`](https://github.com/FluxML/Zygote.jl), [`Flux`](https://github.com/FluxML/Flux.jl) and [`Diffractor`](https://github.com/JuliaDiff/Diffractor.jl) directly.

## Install 

``` shell
git clone https://github.com/jieli-matrix/NiSparseArrays.jl.git
# git clone https://gitlab.summer-ospp.ac.cn/summer2021/210370152.git
```

To install, type ] in a julia (>=1.6) REPL and then input

``` julia
pkg> add NiSparseArrays 
```

## API References  

### Low-Level Operators

| API             | description        |
| ---------------- | --------------- |
| `function imul!(C::StridedVecOrMat, A::AbstractSparseMatrix{T}, B::DenseInputVecOrMat, α::Number, β::Number) where T`   | sparse matrix to dense matrix multiplication |
|`function imul!(C::StridedVecOrMat, xA::Adjoint{T, <:AbstractSparseMatrix}, B::DenseInputVecOrMat, α::Number, β::Number) where T` |  adjoint sparse matrix to dense matrix multiplication |
|`function imul!(C::StridedVecOrMat, X::DenseMatrixUnion, A::AbstractSparseMatrix{T}, α::Number, β::Number) where T`| dense matrix to sparse matrix multiplication |
|`function imul!(C::StridedVecOrMat, X::Adjoint{T1, <:DenseMatrixUnion}, A::AbstractSparseMatrix{T2}, α::Number, β::Number) where {T1, T2}`| adjoint dense matrix to sparse matrix multiplication |
|`imul!(C::StridedVecOrMat, X::DenseMatrixUnion, xA::Adjoint{T, <:AbstractSparseMatrix}, α::Number, β::Number) where T`|dense matrix to sparse matrix multiplication |
|`function idot(r, A::SparseMatrixCSC{T},B::SparseMatrixCSC{T}) where {T}` | dot operation between sparsematrix and sparsematrix|
|`function idot(r, x::AbstractVector, A::AbstractSparseMatrix{T1}, y::AbstractVector{T2}) where {T1, T2}` | dot operation between sparsematrix and densevector|
|`function idot(r, x::SparseVector, A::AbstractSparseMatrix{T1}, y::SparseVector{T2}) where {T1, T2}`| dot operation between sparsematrix and sparsevector|

### High-Level Operators

| API             | description        |
| ---------------- | --------------- |
| `low_rank_svd(A::AbstractSparseMatrix{T}, l::Int, niter::Int = 2, M::Union{AbstractMatrix{T}, Nothing} = nothing) where T` | Return the singular value decomposition of a sparse matrix `A` with estimated rank `l` such that `A ≈ U diag(S) Vt`. In case row vector `M` is given, then SVD is computed for the matrix `A - M`.|

## A Simple Use Case

Here we present a minimal use case to illustrate how to use `NiSparseArrays` to speed up `Zygote`'s gradient computation. To access more examples, please navigate to the `examples` directory.

``` julia 
julia> using SparseArrays, LinearAlgebra, Random, BenchmarkTools

julia> A = sprand(1000, 1000, 0.1);

julia> x = rand(1000);

julia> using Zygote

julia> @btime Zygote.gradient((A, x) -> sum(A*x), $A, $x)
  15.065 ms (27 allocations: 8.42 MiB)

julia> using NiSparseArrays

julia> @btime Zygote.gradient((A, x) -> sum(A*x), $A, $x)
  644.035 μs (32 allocations: 3.86 MiB)
```

You will see that using `NiSparseArrays` would not only speed up the computation process but also save much memory since our implementation does not convert a sparse matrix to a dense arrays in gradient computation. 

## Contribute 

Suggestions and Comments in the Issues are welcome.

## License

MIT License
