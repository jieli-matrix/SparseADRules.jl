# declare struct 
const DenseMatrixUnion = Union{StridedMatrix, LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular, BitMatrix}
const AdjOrTransDenseMatrix = Union{DenseMatrixUnion,Adjoint{<:Any,<:DenseMatrixUnion},Transpose{<:Any,<:DenseMatrixUnion}}
const DenseInputVector = Union{StridedVector, BitVector}
const DenseInputVecOrMat = Union{AdjOrTransDenseMatrix, StridedVector}


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

@i function imul!(C::StridedVecOrMat, X::DenseMatrixUnion, A::AbstractSparseMatrix, α::Number, β::Number)
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
                    anc ← zero(eltype(A))
                    anc += X[multivec_row, A.rowval[k]]*A.nzval[k]
                end
                C[multivec_row, col] += anc*α
                ~@routine
            end
        end
    end
end


@i function imul!(C::StridedVecOrMat, X::Adjoint{<:Any,<:DenseMatrixUnion}, A::AbstractSparseMatrix, α::Number, β::Number)
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
                    anc ← zero(eltype(A))
                    anc += X.parent[A.rowval[k], multivec_row]' * A.nzval[k]
                end
                C[multivec_row, col] += anc * α
                ~@routine
            end
        end
    end
end

@i function imul!(C::StridedVecOrMat, X::DenseMatrixUnion, xA::Adjoint{<:Any,<:AbstractSparseMatrix}, α::Number, β::Number)
    @safe size(X, 2) == size(xA.parent, 2) || throw(DimensionMismatch())
    @safe size(X, 1) == size(C, 1) || throw(DimensionMismatch())
    @safe size(xA.parent, 1) == size(C, 2) || throw(DimensionMismatch())
    if (β != 1, ~)
        @safe error("only β = 1 is supported, got β = $(β).")
    end
    @invcheckoff for col in 1:size(xA.parent, 2)
        @inbounds for k in nzrange(xA.parent, col)
                        @routine begin
                            anc1 ← zero(eltype(xA.parent))
                            anc1 += (xA.parent.nzval[k])'*α
                        end
            @simd for multivec_row in 1:size(X,1)
                C[multivec_row, xA.parent.rowval[k]] += X[multivec_row, col] * anc1
            end
            ~@routine
        end
    end
end

@i function idot(r::T, A::SparseMatrixCSC{T},B::SparseMatrixCSC{T}) where {T}
    @routine @invcheckoff begin
        (m, n) ← size(A)
        branch_keeper ← zeros(Bool, 2*m)
    end
    @safe size(B) == (m,n) || throw(DimensionMismatch("matrices must have the same dimensions"))
    @invcheckoff @inbounds for j = 1:n
        @routine begin
            ia1 ← A.colptr[j]
            ib1 ← B.colptr[j]
            ia2 ← A.colptr[j+1]
            ib2 ← B.colptr[j+1]
            ia ← ia1
            ib ← ib1
        end
        @inbounds for i=1:ia2-ia1+ib2-ib1-1
            ra ← A.rowval[ia]
            rb ← B.rowval[ib]
            if (ra == rb, ~)
                r += A.nzval[ia]' * B.nzval[ib]
            end
            ## b move -> true, a move -> false
            branch_keeper[i] ⊻= @const ia == ia2-1 || (ib != ib2-1 && ra > rb)
            ra → A.rowval[ia]
            rb → B.rowval[ib]
            if (branch_keeper[i], ~)
                INC(ib)
            else
                INC(ia)
            end
        end
        ~@inbounds for i=1:ia2-ia1+ib2-ib1-1
            ## b move -> true, a move -> false
            branch_keeper[i] ⊻= @const ia == ia2-1 || (ib != ib2-1 && A.rowval[ia] > B.rowval[ib])
            if (branch_keeper[i], ~)
                INC(ib)
            else
                INC(ia)
            end
        end
        ~@routine
    end
    ~@routine
end

@i function idot(r, x::AbstractVector, A::AbstractSparseMatrix{T1}, y::AbstractVector{T2}) where {T1, T2} 
    @safe length(x) == size(A, 1) || throw(DimensionMismatch())
    @safe length(y) == size(A, 2) || throw(DimensionMismatch())

    @invcheckoff @inbounds for col in 1:size(A, 2)
        for k in nzrange(A, col)
            @routine begin
                anc ← zero(promote_type(T1, T2))
                anc += A.nzval[k] * y[col]
            end
            r += (x[A.rowval[k]])' * anc
            ~@routine 
        end
    end
end 

@i function idot(r, x::SparseVector, A::AbstractSparseMatrix{T1}, y::SparseVector{T2}) where {T1, T2}
    @safe length(x) == size(A, 1) || throw(DimensionMismatch())
    @safe length(y) == size(A, 2) || throw(DimensionMismatch())

    @invcheckoff for (yi, yv) in zip(y.nzind, y.nzval)
        for (xi, xv) in  zip(x.nzind, x.nzval)
            for k in nzrange(A, yi)
                if A.rowval[k] == xi
                    @routine begin
                        anc ← zero(promote_type(T1, T2))
                        anc += A.nzval[k] * yv
                    end
                    r += xv' * anc
                    ~@routine
                end
            end
        end
    end
end 