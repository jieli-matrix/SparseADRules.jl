# declare struct 
const DenseMatrixUnion = Union{StridedMatrix, LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular, BitMatrix}
const AdjOrTransDenseMatrix = Union{DenseMatrixUnion,Adjoint{<:Any,<:DenseMatrixUnion},Transpose{<:Any,<:DenseMatrixUnion}}
const DenseInputVector = Union{StridedVector, BitVector}
const DenseInputVecOrMat = Union{AdjOrTransDenseMatrix, StridedVector}

@i function idotxA(r, x::SparseVector, A::AbstractSparseMatrix{T1}, y::SparseVector{T2}) where {T1, T2}
    @safe length(x) == size(A, 1) || throw(DimensionMismatch())
    @safe length(y) == size(A, 2) || throw(DimensionMismatch())

    @invcheckoff @inbounds for (yi, yv) in zip(y.nzind, y.nzval)
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

@i function idotAx(r, x::SparseVector, A::AbstractSparseMatrix{T1}, y::SparseVector{T2}) where {T1, T2}
    @safe length(x) == size(A, 1) || throw(DimensionMismatch())
    @safe length(y) == size(A, 2) || throw(DimensionMismatch())

    @invcheckoff @inbounds for (yi, yv) in zip(y.nzind, y.nzval)
        for k in nzrange(A, yi)
            for (xi, xv) in  zip(x.nzind, x.nzval)
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