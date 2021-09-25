# Low Rank Singular Value Decomposition
"""
    get_approximate_basis(A, l::Int; niter::Int = 2, M = nothing) -> Q
Return Matrix ``Q`` with ``l`` orthonormal columns such that ``Q Q^H A`` approximates ``A``. If ``M`` is specified, then ``Q`` is such that ``Q Q^H (A - M)`` approximates ``A - M``.
"""

struct Normal_QR{QT, RT}
    Q::QT
    R::RT
    function Normal_QR(Q::QT, R::RT) where {QT<:AbstractMatrix, RT<:AbstractMatrix}
        size(Q, 2) == size(R, 1) || throw(DimensionMismatch("$(size(Q)), $(length(R)) not compatible"))
        new{QT, RT}(Q, R)
    end
end

function private_qr(A::AbstractMatrix{T}) where T
	res = LinearAlgebra.qr(A)
	Normal_QR(Matrix(res.Q), res.R)
end

function get_approximate_basis(
    A::AbstractSparseMatrix{T}, l::Int, niter::Int = 2, M::Union{AbstractSparseMatrix{T}, Nothing} = nothing) where T
    m, n = size(A)
    Ω = rand(T, (n, l))
    if M === nothing 
        F_j_Q = private_qr(A * Ω).Q
        for j = 1:niter
            F_H_j_Q = private_qr(A' * F_j_Q).Q
            F_j_Q = private_qr(A * F_H_j_Q).Q
        end
    else
        F_j_Q = private_qr(A * Ω - M * Ω).Q
        for j = 1:niter
            F_H_j_Q = private_qr(A' * F_j_Q - M' * F_j_Q).Q
            F_j_Q = private_qr(A * F_H_j_Q - M * F_H_j_Q).Q
        end
    end
    F_j_Q
end

"""
    low_rank_svd(A, l::Int; niter::Int = 2, M = nothing) -> U, S, Vt
Return the singular value decomposition LowRankSVD(U, S, Vt) of a matrix or a sparse matrix ``A`` such that ``A ≈ U diag(S) Vt``. In case ``M`` is given, then SVD is computed for the matrix ``A - M``.
"""

function low_rank_svd(A::AbstractSparseMatrix{T}, l::Int, niter::Int = 2, M::Union{AbstractSparseMatrix{T}, Nothing} = nothing) where T
    Q = get_approximate_basis(A, l, niter, M)
    if M === nothing
        B = Q' * A
    else
        B = Q' * (A - M)
    end

    dense_svd = svd(B)
    U = Q * dense_svd.U
    return U, dense_svd.S, dense_svd.Vt
end
