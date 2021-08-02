# declare struct 
const DenseMatrixUnion = Union{StridedMatrix, LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular, BitMatrix}
const AdjOrTransDenseMatrix = Union{DenseMatrixUnion,Adjoint{<:Any,<:DenseMatrixUnion},Transpose{<:Any,<:DenseMatrixUnion}}
const DenseInputVector = Union{StridedVector, BitVector}
const DenseInputVecOrMat = Union{AdjOrTransDenseMatrix, StridedVector}

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

# a simple version of Gustavson matrix algorithms
function i_spmatmul(A::SparseMatrixCSC{Tv, Ti}, B::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    mA, nA = size(A)
    nB = size(B, 2)
    nA == size(B, 1) || throw(DimensionMismatch())

    # use mA*nB as nonzeros size of C would be ok, upper bound
    # we may consider estimate a better bound later
    nnzC = mA*nB
    colptrC = Vector{Ti}(undef, nB+1)
    rowvalC = Vector{Ti}(undef, nnzC)
    nzvalC = Vector{Tv}(undef, nnzC)

    @inbounds begin
        ip = 1 #colptr index for C 
        xb = fill(false, mA) #bool column of C in iteration
        for i in 1:nB
            colptrC[i] = ip
            # update ip for next column in C
            ip = i_spcolmul!(rowvalC, nzvalC, xb, i, ip, A, B)
        end
        colptrC[nB+1] = ip
    end
    # delete surplus zeros in C
    resize!(rowvalC, ip - 1)
    resize!(nzvalC, ip - 1)

    # construct C as SparseMatrixCSC format
    C = SparseMatrixCSC(mA, nB, colptrC, rowvalC, nzvalC)
    C
end

function i_spcolmul!(rowvalC, nzvalC, xb, i, ip, A, B)
    rowvalA = rowvals(A); nzvalA = nonzeros(A)
    rowvalB = rowvals(B); nzvalB = nonzeros(B)
    mA = size(A, 1)
    ip0 = ip
    k0 = ip - 1 
    @inbounds begin
        for jp in nzrange(B, i) # nz address of B[:,i]
            nzB = nzvalB[jp]
            # row index for nzB -> select A[:,j] do nzB*A[:,j]
            j = rowvalB[jp] 
            for kp in nzrange(A, j)
                nzC = nzvalA[kp] * nzB
                k = rowvalA[kp] # nz row index
                if xb[k]
                    # nzvalC saved not continous
                    # xv used in original Gustavson algorithm
                    nzvalC[k+k0] += nzC # update C's column
                else
                    # nzvalC saved not continous
                    nzvalC[k+k0] = nzC # initial C's column
                    xb[k] = true
                    rowvalC[ip] = k
                    ip += 1
                end
            end
        end
        if ip > ip0 # add new nzvals or not
            for k = 1:mA
                if xb[k]
                    xb[k] = false
                    rowvalC[ip0] = k
                    nzvalC[ip0] = nzvalC[k+k0] # save it in continous memory
                    ip0 += 1
                end
            end
        end
    end
    ip
end

function i_estimate_mulsize(m::Integer, nnzA::Integer, n::Integer, nnzB::Integer, k::Integer)
    # may be seperate into serveral steps in NiLang
    p = (nnzA / (m * n)) * (nnzB / (n * k)) 
    # branch_keeper need here?
    p >= 1 ? m*k : p > 0 ? Int(ceil(-expm1(log1p(-p) * n)*m*k)) : 0 
end
