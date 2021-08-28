function ChainRulesCore.rrule(
    ::typeof(*), A::AbstractSparseMatrix{T}, B::DenseInputVecOrMat
) where T 
    C = A * B
    function mul_pullback(C̄)
        _, i_A, i_B, _, _ = (~imul!)(AD.GVar(C, unthunk(C̄)), AD.GVar(A), AD.GVar(B), AD.GVar(1.), AD.GVar(1.))
        return ChainRulesCore.NoTangent(), AD.grad(i_A), AD.grad(i_B)
    end
    return C, mul_pullback
end

function ChainRulesCore.rrule(
    ::typeof(*), xA::Adjoint{T, <:AbstractSparseMatrix}, B::DenseInputVecOrMat
) where T
    C = xA * B
    function mul_pullback(C̄)
        _, i_xA, i_B, _, _ = (~imul!)(AD.GVar(C, C̄), AD.GVar(xA), AD.GVar(B), AD.GVar(1.), AD.GVar(1.))
        return ChainRulesCore.NoTangent(), AD.grad(i_xA), AD.grad(i_B)
    end
    return C, mul_pullback
end

function ChainRulesCore.rrule(
    ::typeof(*), X::DenseMatrixUnion,A::AbstractSparseMatrix{T}
) where T 
    C = X * A
    function mul_pullback(C̄)
        _, i_X, i_A, _, _ = (~imul!)(AD.GVar(C, C̄), AD.GVar(X), AD.GVar(A), AD.GVar(1.), AD.GVar(1.))
        return ChainRulesCore.NoTangent(), AD.grad(i_X), AD.grad(i_A)
    end
    return C, mul_pullback
end

function ChainRulesCore.rrule(
    ::typeof(*), X::Adjoint{T1, <:DenseMatrixUnion}, A::AbstractSparseMatrix{T2}
) where {T1, T2} 
    C = X * A
    function mul_pullback(C̄)
        _, i_X, i_A, _, _ = (~imul!)(AD.GVar(C, C̄), AD.GVar(X), AD.GVar(A), AD.GVar(1.), AD.GVar(1.))
        return ChainRulesCore.NoTangent(), AD.grad(i_X), AD.grad(i_A)
    end
    return C, mul_pullback
end

function ChainRulesCore.rrule(
    ::typeof(dot), A::AbstractSparseMatrix{T},B::AbstractSparseMatrix{T}
) where T
    r = dot(A, B)
    function dot_pullback(r̄)
        _, i_A, i_B = (~idot)(AD.GVar(r, r̄), AD.GVar(A), AD.GVar(B))
        return ChainRulesCore.NoTangent(), AD.grad(i_A), AD.grad(i_B)
    end
    return r, dot_pullback
end

