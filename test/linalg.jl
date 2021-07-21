using NiSparseArrays:imul!
using SparseArrays, Random, Test
Random.seed!(1234)
using NiLang, ForwardDiff
using NiLang.AD

# Is declaring approx_rtol here safe? But I don't want to declare approx_rtol in each testset...
approx_rtol = 100*eps()
# Would it be reasonable to test accuracy like this? Is there any better readable code?
# ≈(ioutv, outv, rtol=approx_rtol) 
# keep the same style with SparseArrays 
# https://github.com/JuliaLang/julia/blob/b773bebcdb1eccaf3efee0bfe564ad552c0bcea7/stdlib/SparseArrays/test/sparse.jl#L254 
# real value case
@testset "matrix multiplication" begin
    for i = 1:5
        A = sprand(10, 5, 0.5)
        b = rand(5)
        outv = A*b
        c = zero(outv)
        ioutv = imul!(copy(c), A, b, 1 ,1)[1] # replace with imul!(similar(outv), A, b, 1, 1)[1]?
        @test ≈(ioutv, outv, rtol=approx_rtol)

        B = rand(5, 3)
        outm = A*B
        C = zero(outm)
        ioutm = imul!(copy(C), A, B, 1, 1)[1]
        @test ≈(ioutm, outm, rtol=approx_rtol)
    end
end

# complex value case
@testset "complex matrix multiplication" begin
    for i = 1:5
        A = sprand(ComplexF64, 10, 5, 0.2)
        b = rand(ComplexF64, 5)
        outv = A*b
        c = zero(outv)
        ioutv = imul!(copy(c), A, b, 1, 1)[1]
        @test ≈(ioutv, outv, rtol=approx_rtol)

        B = rand(ComplexF64, 5, 3)
        outm = A*B
        C = zero(outm)
        ioutm = imul!(copy(C), A, B, 1, 1)[1]
        @test ≈(ioutm, outm, rtol=approx_rtol)
    end
end

@testset "adjoint/transpose matrix multiplication" begin
    for t in (adjoint, transpose)
        @eval for i = 1:5
            A = sprand(ComplexF64, 5, 5, 0.2)
            b = rand(ComplexF64, 5)
            outv = $t(A)*b
            c= zero(outv)
            ioutv = imul!(copy(c), $t(A), b, 1, 1)[1]
            @test ≈(ioutv, outv, rtol=approx_rtol)

            B = rand(ComplexF64, 5, 3)
            outm = $t(A)*B
            C = zero(outm)
            ioutm = imul!(copy(C), $t(A), B, 1, 1)[1]
            @test ≈(ioutm, outm, rtol=approx_rtol)
        end
    end
end

###### Test jacobian matrix ######
params(A::SparseMatrixCSC) = params(A.nzval)
fitparams(A::SparseMatrixCSC, p) = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, fitparams(A.nzval, p))
params(v::DenseArray{<:Real}) = v
fitparams(v::DenseArray{<:Real}, p) = p
params(v::DenseArray{<:Complex{T}}) where T = collect(reinterpret(T, v))
fitparams(v::DenseArray{<:T}, p::DenseArray{T2}) where {T<:Complex,T2} = collect(reinterpret(Complex{T2}, p))

@i function loss(l, j, y::AbstractArray{<:Real}, A, x)
    imul!(y, A, x, 1.0, 1.0)
    l += y[j]
end
@i function loss(l, j, y::AbstractArray{<:Complex}, A, x)
    imul!(y, A, x, 1.0, 1.0)
    if j%2 == 1
        l += y[div(j-1,2)+1].re
    else
        l += y[div(j-1,2)+1].im
    end
end

function nilang_mul_jacobian(A, x::AbstractArray)
    px, pA = params(x), params(A)
    J = zeros(T, length(px), length(px) + length(pA))
    for j=1:length(px)
        _, _, _, _, gA, gx = NiLang.AD.Grad(loss)(Val(1), 0.0, j, zero(x), A, x)
        J[j,:] = vcat(grad(params(gA)), grad(params(gx)))
    end
    return J
end

function forwarddiff_mul_jacobian(A, x)
    pA = params(A)
    px = params(x)
    pAx = vcat(pA, px)
    ForwardDiff.jacobian(pAx) do pAx
        pA = pAx[1:length(pA)]
        px = pAx[length(pA)+1:end]
        A2 = fitparams(A, pA)
        x2 = fitparams(x, px)
        y = params(A2 * x2)
    end
end

@testset "jacobian" begin
    for T in [Float64, ComplexF64]
        A = sprand(T, 10, 10, 0.2)
        x = randn(T, 10)
        JF = forwarddiff_mul_jacobian(A, x)
        JN = nilang_mul_jacobian(A, x)
        @test isapprox(JF, JN)
    end
end