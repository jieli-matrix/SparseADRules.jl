# declare struct 
if VERSION >= v"1.6"
    # this union alias is added in https://github.com/JuliaLang/julia/pull/39557
    using SparseArrays: DenseMatrixUnion, AdjOrTransDenseMatrix, DenseInputVector, DenseInputVecOrMat 
else
    const DenseMatrixUnion = Union{StridedMatrix, LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular, BitMatrix}
    const AdjOrTransDenseMatrix = Union{DenseMatrixUnion,Adjoint{<:Any,<:DenseMatrixUnion},Transpose{<:Any,<:DenseMatrixUnion}}
    const DenseInputVector = Union{StridedVector, BitVector}
    const DenseInputVecOrMat = Union{AdjOrTransDenseMatrix, DenseInputVector}
end


@i function imul!(C::StridedVecOrMat, A::AbstractSparseMatrix, B::DenseInputVecOrMat, α::Number, β::Number) 
    @safe size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    @safe size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    @safe size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    if (β != 1, ~)
        @safe error("only β = 1 is supported, got β = $(β).")
    end
    # Here, we close the reversibility check inside the loop to increase performance
    @invcheckoff for k = 1:size(C, 2)
        @inbounds for col = 1:size(A, 2)
            @routine begin
                αxj ← zero(eltype(B))
                αxj += B[col,k] * α
            end
            for j = nzrange(A, col)
                C[A.rowval[j], k] += A.nzval[j] * αxj
            end
            ~@routine
        end
    end
end


@i function imul!(C::StridedVecOrMat, xA::Adjoint{<:Any,<:AbstractSparseMatrix}, B::DenseInputVecOrMat, α::Number, β::Number)
    @safe size(xA.parent, 2) == size(C, 1) || throw(DimensionMismatch())
    @safe size(xA.parent, 1) == size(B, 1) || throw(DimensionMismatch())
    @safe size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    if (β != 1, ~)
        @safe error("only β = 1 is supported, got β = $(β).")
    end
    @invcheckoff for k in 1:size(C, 2)
        @inbounds for col in 1:size(xA.parent, 2)
                for j in nzrange(xA.parent, col)
                    @routine begin    
                        anc1 ← zero(eltype(xA))
                        anc1 += (xA.parent.nzval[j])'
                    end
                C[col,k] += anc1 * B[xA.parent.rowval[j], k]
                ~@routine
            end
        end
    end
end
