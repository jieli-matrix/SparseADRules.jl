function ChainRulesCore.rrule(::typeof(*), A::AbstractSparseMatrix, B::DenseInputVecOrMat)
    C = A * B
    function mul_pullback(C̄)
        _, i_A, i_B, _, _ = (~imul!)(AD.GVar(C, C̄), AD.GVar(A), AD.GVar(B), AD.GVar(1.), AD.GVar(1.))
        return ChainRulesCore.NoTangent(), AD.grad(i_A), AD.grad(i_B)
    end
    return C, mul_pullback
end
