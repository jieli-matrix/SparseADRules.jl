using NiSparseArrays:imul!, idot, nilang_mm_jacobian
const approx_rtol = 100*eps()

@testset "matrix multiplication" begin
    for T in (Float64, ComplexF64)
        for i = 1:5
            A = sprand(T, 10, 5, 0.5)
            b = rand(T, 5)
            outv = A*b
            c = zero(outv)
            ioutv = imul!(copy(c), A, b, 1 ,1)[1] # replace with imul!(similar(outv), A, b, 1, 1)[1]?
            @test ≈(ioutv, outv, rtol=approx_rtol)

            B = rand(T, 5, 3)
            outm = A*B
            C = zero(outm)
            ioutm = imul!(copy(C), A, B, 1, 1)[1]
            @test ≈(ioutm, outm, rtol=approx_rtol)
        end
    end
end

@testset "adjoint matrix multiplication" begin
    for T in (Float64, ComplexF64)
        for i = 1:5
            A = sprand(T, 5, 5, 0.2)
            b = rand(T, 5)
            outv = A'*b
            c= zero(outv)
            ioutv = imul!(copy(c), A', b, 1, 1)[1]
            @test ≈(ioutv, outv, rtol=approx_rtol)

            B = rand(T, 5, 3)
            outm = A'*B
            C = zero(outm)
            ioutm = imul!(copy(C), A', B, 1, 1)[1]
            @test ≈(ioutm, outm, rtol=approx_rtol)
        end
    end    
end



@testset "jacobian" begin
    for T in (Float64, ComplexF64)
        A = sprand(T, 10, 10, 0.2)
        x = randn(T, 10)
        JF = forwarddiff_mm_jacobian(A, x)
        JN = nilang_mm_jacobian(A, x)
        @test isapprox(JF, JN)
    end
end

@testset "dense matrix-sparse matrix multiplication" begin
    for T in (Float64, ComplexF64)    
        for i = 1:5
            B = rand(T, 10, 10)
            A = sprand(T, 10, 5, 0.5)
            outm = B*A
            C = zero(outm)
            ioutm = imul!(copy(C), B, A, 1 ,1)[1] # replace with imul!(similar(outv), A, b, 1, 1)[1]?
            @test ≈(ioutm, outm, rtol=approx_rtol)
        end
    end    
end

@testset "adjoint dense matrix-sparse matrix multiplication" begin
    for T in (Float64, ComplexF64)    
        for i = 1:5
            B = rand(T, 10, 5)
            A = sprand(T, 10, 5, 0.5)
            outm = B'*A
            C = zero(outm)
            ioutm = imul!(copy(C), B', A, 1 ,1)[1] # replace with imul!(similar(outv), A, b, 1, 1)[1]?
            @test ≈(ioutm, outm, rtol=approx_rtol)
        end
    end
end

@testset "dense matrix - adjoint sparse matrix multiplication" begin
    for T in (Float64, ComplexF64)    
        for i = 1:5
            B = rand(T, 10, 10)
            A = sprand(T, 5, 10, 0.5)
            outm = B*A'
            C = zero(outm)
            ioutm = imul!(copy(C), B, A', 1 ,1)[1] # replace with imul!(similar(outv), A, b, 1, 1)[1]?
            @test ≈(ioutm, outm, rtol=approx_rtol)
        end
    end
end

@testset "dense vector - sparse matrix dot" begin
    for T in (Float64, ComplexF64)    
        for i = 1:5
            x = rand(T, 10)
            A = sprand(T, 10, 5, 0.2)
            y = rand(T, 5)
            outd = dot(x, A, y)
            r = zero(T)
            ioutd = idot(copy(r), x, A, y)[1]
            @test ≈(ioutd, outd, rtol=approx_rtol)
        end
    end
end

@testset "sparse vector - sparse matrix dot" begin
    for T in (Float64, ComplexF64)    
        for i = 1:5
            x = sprand(T, 10, 0.2)
            A = sprand(T, 10, 5, 0.2)
            y = sprand(T, 5, 0.3)
            outd = dot(x, A, y)
            r = zero(T)
            ioutd = idot(copy(r), x, A, y)[1]
            @test ≈(ioutd, outd, rtol=approx_rtol)
        end
    end
end
