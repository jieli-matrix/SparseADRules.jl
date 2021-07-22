###calculate jacobian by NiLang###
"""
    helper function params and fitparams
        function params wraps parameters in track of gradient when given array/matrix.
            params(v::DenseArray)
            params(A::SparseMatrixCSC)
        function fitparams fits the original datastructure with parameters in track of gradient.
            fitparams(v::DenseArray)
            fitparams(A::SparseMatrixCSC)
"""
params(v::DenseArray{<:Real}) = v
fitparams(v::DenseArray{<:Real}, p) = p
params(v::DenseArray{<:Complex{T}}) where T = collect(reinterpret(T, v))
fitparams(v::DenseArray{<:T}, p::DenseArray{T2}) where {T<:Complex,T2} = collect(reinterpret(Complex{T2}, p))
params(A::SparseMatrixCSC) = params(A.nzval)
fitparams(A::SparseMatrixCSC, p) = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, fitparams(A.nzval, p))

"""
    mvloss(l, j, y::AbstractArray{<:Real}, A, x) -> scalar l 
mvloss defines loss on sparse-matrix & dense vector/matrix multiplication(Real Type)   
"""
@i function mmloss(l, j, y::AbstractArray{<:Real}, A, x)
    imul!(y, A, x, 1.0, 1.0)
    l += y[j]
end

"""
    mvloss(l, j, y::AbstractArray{<:Complex}, A, x) -> scalar l 
mvloss defines loss on sparse-matrix & dense vector/matrix multiplication(Complex Type)   
"""
@i function mmloss(l, j, y::AbstractArray{<:Complex}, A, x)
    imul!(y, A, x, 1.0, 1.0)
    if j%2 == 1
        l += y[div(j-1,2)+1].re
    else
        l += y[div(j-1,2)+1].im
    end
end

"""
    nilang_mv_jacobian(A, x::AbstractArray) -> AbstractArray J
nilang_mv_jacobian calculates the jacobian with respect to parameter A and x based on mvloss in NiLang
"""
function nilang_mm_jacobian(A, x::AbstractArray)
    px, pA = params(x), params(A)
    J = zeros(eltype(px), length(px), length(px) + length(pA))
    for j=1:length(px)
        _, _, _, _, gA, gx = NiLang.AD.Grad(mmloss)(Val(1), 0.0, j, zero(x), A, x)
        J[j,:] = vcat(grad(params(gA)), grad(params(gx)))
    end
    return J
end


