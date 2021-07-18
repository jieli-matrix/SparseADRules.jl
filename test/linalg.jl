using NiSparseArrays:imul!
using SparseArrays, Random, Test
Random.seed!(1234)

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
