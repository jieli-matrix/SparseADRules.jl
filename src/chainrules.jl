function ChainRulesCore.rrule(
    ::typeof(*), A::AbstractSparseMatrix{T}, B::DenseInputVecOrMat
) where T 
    C = A * B
    function mul_pullback(C̄)
        _, i_A, i_B, _, _ = (~imul!)(AD.GVar(C, unthunk(C̄)), AD.GVar(A), AD.GVar(B), AD.GVar(1.), AD.GVar(1.))
        return ChainRulesCore.NoTangent(), AD.grad(i_A), AD.grad(i_B)
    end
    return C, mul_pullback
end

function ChainRulesCore.rrule(
    ::typeof(*), xA::Adjoint{T, <:AbstractSparseMatrix}, B::DenseInputVecOrMat
) where T
    C = xA * B
    function mul_pullback(C̄)
        _, i_xA, i_B, _, _ = (~imul!)(AD.GVar(C, C̄), AD.GVar(xA), AD.GVar(B), AD.GVar(1.), AD.GVar(1.))
        return ChainRulesCore.NoTangent(), AD.grad(i_xA), AD.grad(i_B)
    end
    return C, mul_pullback
end

function ChainRulesCore.rrule(
    ::typeof(*), X::DenseMatrixUnion,A::AbstractSparseMatrix{T}
) where T 
    C = X * A
    function mul_pullback(C̄)
        _, i_X, i_A, _, _ = (~imul!)(AD.GVar(C, C̄), AD.GVar(X), AD.GVar(A), AD.GVar(1.), AD.GVar(1.))
        return ChainRulesCore.NoTangent(), AD.grad(i_X), AD.grad(i_A)
    end
    return C, mul_pullback
end

function ChainRulesCore.rrule(
    ::typeof(*), X::Adjoint{T1, <:DenseMatrixUnion}, A::AbstractSparseMatrix{T2}
) where {T1, T2} 
    C = X * A
    function mul_pullback(C̄)
        _, i_X, i_A, _, _ = (~imul!)(AD.GVar(C, C̄), AD.GVar(X), AD.GVar(A), AD.GVar(1.), AD.GVar(1.))
        return ChainRulesCore.NoTangent(), AD.grad(i_X), AD.grad(i_A)
    end
    return C, mul_pullback
end

function ChainRulesCore.rrule(
    ::typeof(dot), A::AbstractSparseMatrix{T},B::AbstractSparseMatrix{T}
) where T
    r = dot(A, B)
    function dot_pullback(r̄)
        _, i_A, i_B = (~idot)(AD.GVar(r, r̄), AD.GVar(A), AD.GVar(B))
        return ChainRulesCore.NoTangent(), AD.grad(i_A), AD.grad(i_B)
    end
    return r, dot_pullback
end

function ChainRulesCore.rrule(
    ::typeof(dot), x::AbstractVector, A::AbstractSparseMatrix{T1}, y::AbstractVector{T2}
) where {T1, T2}
    r = dot(x, A, y)
    function dot_pullback(r̄)
        _, i_x, i_A, i_y = (~idot)(AD.GVar(r, r̄), AD.GVar(x), AD.GVar(A), AD.GVar(y))
        return ChainRulesCore.NoTangent(), AD.grad(i_x), AD.grad(i_A), AD.grad(i_y)
    end
    return r, dot_pullback
end

function ChainRulesCore.rrule(
    ::typeof(dot), x::SparseVector, A::AbstractSparseMatrix{T1}, y::SparseVector{T2}
) where {T1, T2}
    r = dot(x, A, y)
    function dot_pullback(r̄)
        _, i_x, i_A, i_y = (~idot)(AD.GVar(r, r̄), AD.GVar(x), AD.GVar(A), AD.GVar(y))
        return ChainRulesCore.NoTangent(), AD.grad(i_x), AD.grad(i_A), AD.grad(i_y)
    end
    return r, dot_pullback
end

function _qr_pullback(Ȳ::Tangent, F, A)
    ∂X = qr_rev(F.Q, F.R, Ȳ.Q, Ȳ.R, A)
    return (NoTangent(), ∂X)
end
_qr_pullback(Ȳ::AbstractThunk, F, A) = _qr_pullback(unthunk(Ȳ), F, A)
function ChainRulesCore.rrule(::typeof(private_qr), X::AbstractArray)
    F = private_qr(X)
    qr_pullback(ȳ) = _qr_pullback(ȳ, F, X)
    return F, qr_pullback
end

function ChainRulesCore.rrule(::typeof(getproperty), F::T, x::Symbol) where T <: Normal_QR
    function getproperty_qr_pullback(Ȳ)
        C = Tangent{T}
        ∂F = if x === :Q
            C(Q=Ȳ,)
        elseif x === :R
            C(R=Ȳ,)
        end
        return NoTangent(), ∂F, NoTangent()
    end
    return getproperty(F, x), getproperty_qr_pullback
end

function qr_rev_fullrank(q, r, dq, dr)
    dqnot0 = !(dq isa NoTangent)
    drnot0 = !(dr isa NoTangent)
    if (!dqnot0 && !drnot0)
        return NoTangent()
    end
    ex = drnot0 && dqnot0 ? r*dr' - dq'*q : (dqnot0 ? -dq'*q : r*dr')
    b = dqnot0 ? q*copyltu!(ex)+dq : q*copyltu!(ex)
    return Matrix(trtrs!('U', 'N', 'N', r, do_adjoint(b))')
end

function qr_rev(q, r, dq, dr, A)
    dqnot0 = !(dq isa NoTangent)
    drnot0 = !(dr isa NoTangent)
    (!dqnot0 && !drnot0) && return NoTangent()
    size(r, 1) == size(r, 2) && return qr_rev_fullrank(q, r, dq ,dr)
    M, N = size(r)
    B = view(A,:,M+1:N)
    U = view(r,:,1:M)
    D = view(r,:,M+1:N)
    if drnot0
        dD = view(dr,:,M+1:N)
        da = qr_rev_fullrank(q, U, dqnot0 ? dq+B*dD' : B*dD', view(dr,:,1:M))
        db = q*dD
    else
        da = qr_rev_fullrank(q, U, dq, nothing)
        db = zero(B)
    end
    return hcat(da, db)
end