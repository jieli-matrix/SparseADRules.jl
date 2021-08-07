using NiSparseArrays:imul!

"""
    forwarddiff_mv_jacobian(A, B) -> AbstractArray J 
forwarddiff_mv_jacobian calculates the jacobian with respect to parameter A and B based on multiplication in ForwardDiff
"""
function forwarddiff_mm_jacobian(A, B)
    pA = params(A)
    pB = params(B)
    pC = vcat(pA, pB)
    ForwardDiff.jacobian(pC) do pC
        pA = pC[1:length(pA)]
        pB = pC[length(pA)+1:end]
        A2 = fitparams(A, pA)
        B2 = fitparams(B, pB)
        pC = params(A2 * B2)
    end
end

"""
    nilang_mv_jacobian(A, x::AbstractArray) -> AbstractArray J
nilang_mv_jacobian calculates the jacobian with respect to parameter A and x based on mvloss in NiLang
"""
function nilang_mm_jacobian(A, B)
    pB, pA = params(B), params(A)
    C = zeros(eltype(B), size(A, 1), size(B, 2))
    pC = params(C)
    J = zeros(eltype(pB), length(pC), length(pB) + length(pA))
    for j=1:length(pC)
        _, _, _, _, gA, gB = NiLang.AD.Grad(mmloss)(Val(1), 0.0, j, C, A, B)
        J[j,:] = vcat(grad(params(gA)), grad(params(gB)))
    end
    return J
end

"""
    mvloss(l, j, y::AbstractArray{<:Real}, A, x) -> scalar l 
mvloss defines loss on sparse-matrix & dense vector/matrix multiplication(Real Type)   
"""
@i function mmloss(l, j, C::AbstractArray{<:Real}, A, B)
    imul!(C, A, B, 1.0, 1.0)
    l += C[j]
end

"""
    mvloss(l, j, y::AbstractArray{<:Complex}, A, x) -> scalar l 
mvloss defines loss on sparse-matrix & dense vector/matrix multiplication(Complex Type)   
"""
@i function mmloss(l, j, C::AbstractArray{<:Complex}, A, B)
    imul!(C, A, B, 1.0, 1.0)
    if j%2 == 1
        l += C[div(j-1,2)+1].re
    else
        l += C[div(j-1,2)+1].im
    end
end

"""
    helper function params and fitparams
        function params wraps parameters in track of gradient when given array/matrix.
            params(v::DenseArray)
            params(A::SparseMatrixCSC)
        function fitparams fits the original datastructure with parameters in track of gradient.
            fitparams(v::DenseArray)
            fitparams(A::SparseMatrixCSC)
"""
params(B::DenseArray{<:Real}) = B[:]
fitparams(B::DenseArray{<:Real}, p) = reshape(p, size(B))
params(B::DenseArray{<:Complex{T}}) where T = collect(reinterpret(T, B[:]))
fitparams(B::DenseArray{<:T}, p::DenseArray{T2}) where {T<:Complex,T2} = reshape(collect(reinterpret(Complex{T2}, p)), size(B))
params(A::AbstractSparseMatrix) = params(A.nzval)
fitparams(A::AbstractSparseMatrix, p) = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, fitparams(A.nzval, p))
function params(xA::Adjoint{<:Any,<:AbstractSparseMatrix})
    nA = copy(xA)
    params(nA.nzval)
end
function fitparams(xA::Adjoint{<:Any,<:AbstractSparseMatrix}, p) 
    nA = copy(xA)
    fitparams(nA, p)
end

