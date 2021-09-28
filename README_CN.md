# NiSparseArrays

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jieli-matrix.github.io/NiSparseArrays.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jieli-matrix.github.io/NiSparseArrays.jl/dev)
[![Build Status](https://github.com/jieli-matrix/NiSparseArrays.jl/workflows/CI/badge.svg)](https://github.com/jieli-matrix/NiSparseArrays.jl/actions)
[![Coverage](https://codecov.io/gh/jieli-matrix/NiSparseArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jieli-matrix/NiSparseArrays.jl)

[英文版本](README.md)

`NiSparseArrays`是[开源软件供应链点亮计划-暑期2021仓库](https://summer.iscas.ac.cn/#/?lang=chi)之一。`NiSparseArrays` 使用[`NiLang`](https://giggleliu.github.io/NiLang.jl/dev/)对稀疏矩阵操作进行实现从而得到其微分规则，并将这些规则导入至[`ChainRules`](https://github.com/JuliaDiff/ChainRules.jl)。

## 背景

稀疏矩阵在科学计算中应用广泛，但是在Julia语言里面却没有很好的软件包实现对稀疏矩阵的自动微分，这个项目使用可逆嵌入式语言 `NiLang.jl`对稀疏矩阵操作进行实现从而得到其微分规则。生成的微分规则将以扩展的形式导入到`ChainRules.jl`中，使用者可以直接通过使用自动微分包比如[`Zygote`](https://github.com/FluxML/Zygote.jl), [`Flux`](https://github.com/FluxML/Flux.jl)和[`Diffractor`](https://github.com/JuliaDiff/Diffractor.jl)来获取这些特性。

## 安装 

``` shell
git clone https://github.com/jieli-matrix/NiSparseArrays.jl.git
# git clone https://gitlab.summer-ospp.ac.cn/summer2021/210370152.git
```

在julia (>=1.6) REPL 中键入 ] 然后输入

``` julia
git clone 
pkg> add NiSparseArrays 
```

## API一览  

| API             | 描述        |
| ---------------- | --------------- |
| `function imul!(C::StridedVecOrMat, A::AbstractSparseMatrix{T}, B::DenseInputVecOrMat, α::Number, β::Number) where T`   | 稀疏矩阵与稠密矩阵乘法 |
|`function imul!(C::StridedVecOrMat, xA::Adjoint{T, <:AbstractSparseMatrix}, B::DenseInputVecOrMat, α::Number, β::Number) where T` |  共轭稀疏矩阵与稠密矩阵乘法|
|`function imul!(C::StridedVecOrMat, X::DenseMatrixUnion, A::AbstractSparseMatrix{T}, α::Number, β::Number) where T`| 稠密矩阵与稀疏矩阵乘法 |
|`function imul!(C::StridedVecOrMat, X::Adjoint{T1, <:DenseMatrixUnion}, A::AbstractSparseMatrix{T2}, α::Number, β::Number) where {T1, T2}`| 共轭稠密矩阵与稀疏矩阵乘法 |
|`imul!(C::StridedVecOrMat, X::DenseMatrixUnion, xA::Adjoint{T, <:AbstractSparseMatrix}, α::Number, β::Number) where T`|稠密矩阵与共轭稀疏矩阵乘法 |
|`function idot(r, A::SparseMatrixCSC{T},B::SparseMatrixCSC{T}) where {T}` | 稀疏矩阵与稀疏矩阵的点积 |
|`function idot(r, x::AbstractVector, A::AbstractSparseMatrix{T1}, y::AbstractVector{T2}) where {T1, T2}` | 稀疏矩阵与稠密向量的点积 |
|`function idot(r, x::SparseVector, A::AbstractSparseMatrix{T1}, y::SparseVector{T2}) where {T1, T2}`| 稀疏矩阵与稀疏向量的点积 |

### 高阶算子

| API             | description        |
| ---------------- | --------------- |
| `low_rank_svd(A::AbstractSparseMatrix{T}, l::Int, niter::Int = 2, M::Union{AbstractMatrix{T}, Nothing} = nothing) where T` | 返回秩约为`l`的稀疏矩阵`A`的奇异值分解 `A ≈ U diag(S) Vt`。在行向量`M`给定的情形下，对矩阵`A - M`进行奇异值分解。|

## 一个简单的用例

这里我们用一个最小的用例去展示如何使用`NiSparseArrays`去加速`Zygote`梯度。更多测试用例，请前往`examples`文件夹查看。

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

你会发现使用`NiSparseArrays`不仅能够加速计算过程，还能够节省内存分配——这是因为我们的实现在梯度计算的过程中并不会将一个稀疏矩阵转换为稠密矩阵。

## 贡献

欢迎提出Issue和PR👏

## 许可证

MIT许可证