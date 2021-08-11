# ChainRulesTestUtils requires some method override to support sparse array
# BEGIN ChainRulesTestUtils

# we use this to generate Tangent and pass to test_rrule
# for example: A ⊢ sprand_tangent(A)
function sprand_tangent(A::AT) where AT<:SparseMatrixCSC
    Ā = copy(A)
    Ā.nzval .= rand(eltype(A), length(A.nzval))
    return Tangent{AT}(Ā)
end

function Base.:(+)(a::P, b::Tangent{P}) where P<:SparseMatrixCSC
    b[1] .+ a
    return b
end
function FiniteDifferences.to_vec(A::AT) where AT<:SparseMatrixCSC
    v = A.nzval
    function SparseMatrixCSC_from_vec(v)
        m = A.m
        n = A.n
        colptr = A.colptr
        rowval = A.rowval
        AT(m, n, colptr, rowval, v)
    end
    return v, SparseMatrixCSC_from_vec
end
# END ChainRulesTestUtils
