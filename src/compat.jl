if VERSION >= v"1.6"
    # this union alias is added in https://github.com/JuliaLang/julia/pull/39557
    using SparseArrays: DenseMatrixUnion, AdjOrTransDenseMatrix, DenseInputVector, DenseInputVecOrMat 
else
    const DenseMatrixUnion = Union{StridedMatrix, LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular, BitMatrix}
    const AdjOrTransDenseMatrix = Union{DenseMatrixUnion,Adjoint{<:Any,<:DenseMatrixUnion},Transpose{<:Any,<:DenseMatrixUnion}}
    const DenseInputVector = Union{StridedVector, BitVector}
    const DenseInputVecOrMat = Union{AdjOrTransDenseMatrix, DenseInputVector}
end
