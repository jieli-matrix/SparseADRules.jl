trtrs!(c1::Char, c2::Char, c3::Char, r::AbstractMatrix, b::AbstractVecOrMat) = LinearAlgebra.LAPACK.trtrs!(c1, c2, c3, r, b)

"""
    copyltu!(A::AbstractMatrix) -> AbstractMatrix
copy the lower triangular to upper triangular.
"""
function copyltu!(A::AbstractMatrix)
    m, n = size(A)
    for i=1:m
        A[i,i] = real(A[i,i])
        for j=i+1:n
            @inbounds A[i,j] = conj(A[j,i])
        end
    end
    A
end

do_adjoint(A::Matrix) = Matrix(A')