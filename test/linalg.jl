using NiSparseArrays:imul!, nilang_mm_jacobian, idot!
const approx_rtol = 100*eps()

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

@testset "adjoint matrix multiplication" begin
        for i = 1:5
            A = sprand(ComplexF64, 5, 5, 0.2)
            b = rand(ComplexF64, 5)
            outv = A'*b
            c= zero(outv)
            ioutv = imul!(copy(c), A', b, 1, 1)[1]
            @test ≈(ioutv, outv, rtol=approx_rtol)

            B = rand(ComplexF64, 5, 3)
            outm = A'*B
            C = zero(outm)
            ioutm = imul!(copy(C), A', B, 1, 1)[1]
            @test ≈(ioutm, outm, rtol=approx_rtol)
        end
end



@testset "jacobian" begin
    for T in [Float64, ComplexF64]
        A = sprand(T, 10, 10, 0.2)
        x = randn(T, 10)
        JF = forwarddiff_mm_jacobian(A, x)
        JN = nilang_mm_jacobian(A, x)
        @test isapprox(JF, JN)
    end
end

@testset "dense matrix-sparse matrix multiplication" begin
    for i = 1:5
        B = rand(10, 10)
        A = sprand(10, 5, 0.5)
        outm = B*A
        C = zero(outm)
        ioutm = imul!(copy(C), B, A, 1 ,1)[1] # replace with imul!(similar(outv), A, b, 1, 1)[1]?
        @test ≈(ioutm, outm, rtol=approx_rtol)
    end
end

@testset "adjoint dense matrix-sparse matrix multiplication" begin
    for i = 1:5
        B = rand(10, 5)
        A = sprand(10, 5, 0.5)
        outm = B'*A
        C = zero(outm)
        ioutm = imul!(copy(C), B', A, 1 ,1)[1] # replace with imul!(similar(outv), A, b, 1, 1)[1]?
        @test ≈(ioutm, outm, rtol=approx_rtol)
    end
end

@testset "dense matrix - adjoint sparse matrix multiplication" begin
    for i = 1:5
        B = rand(10, 10)
        A = sprand(5, 10, 0.5)
        outm = B*A'
        C = zero(outm)
        ioutm = imul!(copy(C), B, A', 1 ,1)[1] # replace with imul!(similar(outv), A, b, 1, 1)[1]?
        @test ≈(ioutm, outm, rtol=approx_rtol)
    end
end

@testset "dense vector - sparse matrix dot" begin
    for i = 1:5
        x = rand(Float64, 10)
        A = sprand(10, 5, 0.2)
        y = rand(Float64, 5)
        outd = dot(x, A, y)
        r = zero(Float64)
        ioutd = idot!(copy(r), x, A, y)[1]
        @test ≈(ioutd, outd, rtol=approx_rtol)
    end
end
