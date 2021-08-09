function ChainRulesCore.rrule(::typeof(*), A::AbstractSparseMatrix, B::DenseInputVecOrMat)
    C = A*B
    function pullback(C̄)
        _, gA, gB, _, _ = grad((~imul!)(GVar(C, C̄), GVar(A), GVar(B), GVar(1.0), GVar(1.0)))
        return ChainRulesCore.NoTangent(), gA, gB
    end
    C, pullback
end
