# ChainRulesTestUtils requires some method override to support sparse array
# BEGIN ChainRulesTestUtils

# we use this to generate Tangent and pass to test_rrule
# for example: A ⊢ sprand_tangent(A)
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

function FiniteDifferences.to_vec(A::Adjoint{T, <:AbstractSparseMatrix}) where T
    Ā = copy(A)
    return FiniteDifferences.to_vec(Ā)
end

function FiniteDifferences._j′vp(fdm, f, ȳ::Vector{<:Number}, x::Vector{<:Complex})
    isempty(x) && return eltype(ȳ)[] # if x is empty, then so is the jacobian and x̄
    r_jcb = transpose(first(FiniteDifferences.jacobian(fdm, f, real(x)))) * real(ȳ)
    i_jcb = transpose(first(FiniteDifferences.jacobian(fdm, f, imag(x)))) * imag(ȳ)
    return Complex(r_jcb, i_jcb)
end

# END ChainRulesTestUtils
