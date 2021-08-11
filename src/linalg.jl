@i function imul!(C::StridedVecOrMat, A::AbstractSparseMatrix{T}, B::DenseInputVecOrMat, α::Number, β::Number) where T
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


@i function imul!(C::StridedVecOrMat, xA::Adjoint{T, <:AbstractSparseMatrix}, B::DenseInputVecOrMat, α::Number, β::Number) where T
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
                        @zeros T anc1 anc2
                        anc1 += (xA.parent.nzval[j])'* α
                        anc2 += B[xA.parent.rowval[j], k]
                    end
                C[col,k] += anc1 * anc2
                ~@routine
            end
        end
    end
end

@i function imul!(C::StridedVecOrMat, X::DenseMatrixUnion, A::AbstractSparseMatrix{T}, α::Number, β::Number) where T
    @safe size(X, 2) == size(A, 1) || throw(DimensionMismatch())
    @safe size(X, 1) == size(C, 1) || throw(DimensionMismatch())
    @safe size(A, 2) == size(C, 2) || throw(DimensionMismatch())
    if (β != 1, ~)
        @safe error("only β = 1 is supported, got β = $(β).")
    end
    @invcheckoff for col in 1:size(A, 2)
        @inbounds for k in nzrange(A, col)
            @simd for multivec_row in 1:size(X,1)
                @routine begin
                    anc1 ← zero(T)
                    anc2 ← zero(promote_eltype(X, A))
                    anc1 += X[multivec_row, A.rowval[k]]
                    anc2 += anc1 * A.nzval[k]
                end
                C[multivec_row, col] += anc2 * α
                ~@routine
            end
        end
    end
end


@i function imul!(C::StridedVecOrMat, X::Adjoint{T1, <:DenseMatrixUnion}, A::AbstractSparseMatrix{T2}, α::Number, β::Number) where {T1, T2}
    @safe size(X.parent, 1) == size(A, 1) || throw(DimensionMismatch())
    @safe size(X.parent, 2) == size(C, 1) || throw(DimensionMismatch())
    @safe size(A, 2) == size(C, 2) || throw(DimensionMismatch())
    if (β != 1, ~)
        @safe error("only β = 1 is supported, got β = $(β).")
    end
    @invcheckoff for multivec_row in 1:size(X.parent, 2)
        for col in 1:size(A, 2)
            @inbounds for k in nzrange(A, col)
                @routine begin
                    anc1 ← zero(T1)
                    anc2 ← zero(promote_type(T1, T2))
                    anc1 += X.parent[A.rowval[k], multivec_row]'
                    anc2 += anc1 * A.nzval[k]
                end
                C[multivec_row, col] += anc2 * α
                ~@routine
            end
        end
    end
end

@i function imul!(C::StridedVecOrMat, X::DenseMatrixUnion, xA::Adjoint{T, <:AbstractSparseMatrix}, α::Number, β::Number) where T
    @safe size(X, 2) == size(xA.parent, 2) || throw(DimensionMismatch())
    @safe size(X, 1) == size(C, 1) || throw(DimensionMismatch())
    @safe size(xA.parent, 1) == size(C, 2) || throw(DimensionMismatch())
    if (β != 1, ~)
        @safe error("only β = 1 is supported, got β = $(β).")
    end
    @invcheckoff for col in 1:size(xA.parent, 2)
        @inbounds for k in nzrange(xA.parent, col)
                        @routine begin
                            anc1 ← zero(T)
                            anc1 += (xA.parent.nzval[k])'*α
                        end
            @simd for multivec_row in 1:size(X,1)
                C[multivec_row, xA.parent.rowval[k]] += X[multivec_row, col] * anc1
            end
            ~@routine
        end
    end
end