using NiSparseArrays:imul!, forwarddiff_mv_jacobian, nilang_mv_jacobian
const approx_rtol = 100*eps()

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


@testset "jacobian" begin
    for T in [Float64, ComplexF64]
        A = sprand(T, 10, 10, 0.2)
        x = randn(T, 10)
        JF = forwarddiff_mv_jacobian(A, x)
        JN = nilang_mv_jacobian(A, x)
        @test isapprox(JF, JN)
    end
end