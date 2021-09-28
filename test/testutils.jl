# ChainRulesTestUtils requires some method override to support sparse array
# BEGIN ChainRulesTestUtils

# we use this to generate Tangent and pass to test_rrule
# for example: A ⊢ sprand_tangent(A)
using SparseArraysAD:Normal_QR

function sprand_tangent(A::AT) where AT<:SparseMatrixCSC
    return ChainRulesTestUtils.rand_tangent(A)
end

function sprand_tangent(A::Adjoint{T, <:AbstractSparseMatrix}) where T
    # when ProjectTo is supported for Adjoint, copy is not necessary
    # https://github.com/JuliaDiff/ChainRulesCore.jl/issues/447
    return ChainRulesTestUtils.rand_tangent(copy(A))
end

function sprand_tangent(x::SparseVector)
    return ChainRulesTestUtils.rand_tangent(x)
end

function Base.:(+)(a::P, b::Tangent{P}) where P<:SparseMatrixCSC
    b[1] .+ a
    return b
end

function FiniteDifferences.to_vec(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    v = reinterpret(real(Tv), A.nzval)
    function SparseMatrixCSC_from_vec(v)
        m = A.m
        n = A.n
        colptr = A.colptr
        rowval = A.rowval
        cv = Vector(reinterpret(Tv, v))
        SparseMatrixCSC(m, n, colptr, rowval, cv)
    end
    return v, SparseMatrixCSC_from_vec
end

function FiniteDifferences.to_vec(x::SparseVector{Tv, Ti}) where {Tv, Ti}
    v= reinterpret(real(Tv), x.nzval)
    function SparseVector_from_vec(v)
        n = x.n
        nzind = x.nzind
        cv = Vector(reinterpret(Tv, v))
        SparseVector(n, nzind, cv)
    end
    return v, SparseVector_from_vec
end

function FiniteDifferences.to_vec(A::Adjoint{T, <:AbstractSparseMatrix}) where T
    Ā = copy(A)
    return FiniteDifferences.to_vec(Ā)
end

function FiniteDifferences.to_vec(x::S) where {S <: Normal_QR }
    q_vec, q_back = FiniteDifferences.to_vec(x.Q)
    r_vec, r_back = FiniteDifferences.to_vec(x.R)
    function Normal_QR_from_vec(v)
        Q_new = q_back(v.Q)
        R_new = r_back(v.R)
        return S(Q_new, R_new)
    end
    return [q_vec; r_vec], Normal_QR_from_vec
end

function FiniteDifferences._j′vp(fdm, f, ȳ::Vector{<:Number}, x::Vector{<:Complex})
    isempty(x) && return eltype(ȳ)[] # if x is empty, then so is the jacobian and x̄
    r_jcb = transpose(first(FiniteDifferences.jacobian(fdm, f, real(x)))) * real(ȳ)
    i_jcb = transpose(first(FiniteDifferences.jacobian(fdm, f, imag(x)))) * imag(ȳ)
    return Complex(r_jcb, i_jcb)
end

# END ChainRulesTestUtils
