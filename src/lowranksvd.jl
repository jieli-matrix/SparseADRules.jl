# Low Rank Singular Value Decomposition
"""
    get_approximate_basis(A, l::Int; niter::Int = 2, M = nothing) -> Q
Return Matrix ``Q`` with ``l`` orthonormal columns such that ``Q Q^H A`` approximates ``A``. If ``M`` is specified, then ``Q`` is such that ``Q Q^H (A - M)`` approximates ``A - M``.
"""

function get_approximate_basis(
    A::AbstractSparseMatrix{T}, l::Int, niter::Int = 2, M::Union{AbstractSparseMatrix{T}, Nothing} = nothing) where T
    m, n = size(A)
    Ω = rand(T, (n, l))
    if M === nothing 
        F_j = qr(A * Ω)
        for j = 1:niter
            F_H_j = qr(A' * Matrix(F_j.Q))
            F_j = qr(A * Matrix(F_H_j.Q))
        end
    else
        F_j = qr(A * Ω - M * Ω)
        for j = 1:niter
            F_H_j = qr(A' * Matrix(F_j.Q) - M' * Matrix(F_j.Q))
            F_j = qr(A * Matrix(F_H_j.Q) - M * Matrix(F_H_j.Q))
        end
    end
    Matrix(F_j.Q)
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
