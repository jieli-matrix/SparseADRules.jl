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
            αxj ← zero(eltype(B))
            αxj += B[col,k] * α
            for j = nzrange(A, col)
                C[A.rowval[j], k] += A.nzval[j]*αxj
            end
            αxj -= B[col,k] * α
            αxj → zero(eltype(B))
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
                    anc1 ← zero(eltype(xA))
                    anc1 += (xA.parent.nzval[j])'
                    C[col,k] += anc1*B[xA.parent.rowval[j], k]
                    anc1 -= (xA.parent.nzval[j])'
                    anc1 → zero(eltype(xA))
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
                C[multivec_row, col] += X[multivec_row, A.rowval[k]]*A.nzval[k]*α
            end
        end
    end
end

# @i function imul!(C::StridedVecOrMat, X::Adjoint{<:Any,<:DenseMatrixUnion}, A::AbstractSparseMatrix, α::Number, β::Number)
#     @safe size(X, 2) == size(A, 1) || throw(DimensionMismatch())
#     @safe size(X, 1) == size(C, 1) || throw(DimensionMismatch())
#     @safe size(A, 2) == size(C, 2) || throw(DimensionMismatch())
#     if (β != 1, ~)
#         @safe error("only β = 1 is supported, got β = $(β).")
#     end
#     @invcheckoff for multivec_row in 1:size(X, 1)
#         for col in 1:size(A, 2)
#             @inbounds for k in nzrange(A, col)
#                 C[multivec_row, col] += X[multivec_row, A.rowval[k]]*A.nzval[k]*α
#             end
#         end
#     end
# end

# this is a better version 
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
                C[multivec_row, col] += X.parent[A.rowval[k], multivec_row]*A.nzval[k]*α
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
                        anc1 ← zero(eltype(xA.parent))
                        anc1 += (xA.parent.nzval[k])'
            @simd for multivec_row in 1:size(X,1)
                C[multivec_row, xA.parent.rowval[k]] += X[multivec_row, col]*anc1*α
            end
            anc1 -= (xA.parent.nzval[k])'
            anc1 → zero(eltype(xA.parent))
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

# function dot(A::AbstractSparseMatrixCSC{T1,S1},B::AbstractSparseMatrixCSC{T2,S2}) where {T1,T2,S1,S2}
#     # A and B should be the same size
#     m, n = size(A)
#     size(B) == (m,n) || throw(DimensionMismatch("matrices must have the same dimensions"))
#     r = dot(zero(T1), zero(T2))
#     # sparse vector dot iterating by columns in A
#     @inbounds for j = 1:n
#         # get nonzero element in A[:,j]/B[:,j] with row idx ra/rb 
#         ia = getcolptr(A)[j]; ia_nxt = getcolptr(A)[j+1]
#         ib = getcolptr(B)[j]; ib_nxt = getcolptr(B)[j+1]
#         if ia < ia_nxt && ib < ib_nxt
#             ra = rowvals(A)[ia]; rb = rowvals(B)[ib]
#             # iteration in A[:,j]/B[:,j]
#             while true
#                 if ra < rb # match false samller ra
#                     ia += oneunit(S1)
#                     ia < ia_nxt || break
#                     ra = rowvals(A)[ia] # update ra bigger
#                 elseif ra > rb # match false sammler rb
#                     ib += oneunit(S2)
#                     ib < ib_nxt || break
#                     rb = rowvals(B)[ib] # update rb bigger
#                 else # ra == rb
#                     r += dot(nonzeros(A)[ia], nonzeros(B)[ib]) # equals to A.nzval[ia]' * B.nzval[ib]
#                     ia += oneunit(S1); ib += oneunit(S2)
#                     ia < ia_nxt && ib < ib_nxt || break
#                     ra = rowvals(A)[ia]; rb = rowvals(B)[ib] # update next ra and rb
#                 end
#             end
#         end
#     end
#     return r
# end

# strange result but I couldn't correct
@i function idot!(r, x::AbstractVector, A::AbstractSparseMatrix, y::AbstractVector) 
    @safe length(x) == size(A, 1) || throw(DimensionMismatch())
    @safe length(y) == size(A, 2) || throw(DimensionMismatch())

    @inbounds for col in size(A, 2)
        for k in nzrange(A, col)
            r += (x[A.rowval[k]])' * A.nzval[k] * y[col]
        end
    end
end 

