using NiSparseArrays:imul!
const approx_rtol = 100*eps()

@testset "matrix multiplication" begin
    for T in (Float64, ComplexF64)
        for i = 1:5
            A = sprand(T, 10, 5, 0.5)
            b = rand(T, 5)
            outv = A*b
            c = zero(outv)
            ioutv = imul!(copy(c), A, b, 1 ,1)[1] 
            @test check_inv(imul!, (copy(c), A, b, 1, 1))
            @test ≈(ioutv, outv, rtol=approx_rtol)

            B = rand(T, 5, 3)
            outm = A*B
            C = zero(outm)
            ioutm = imul!(copy(C), A, B, 1, 1)[1]
            @test check_inv(imul!, (copy(C), A, B, 1, 1))
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
            @test check_inv(imul!, (copy(c), A', b, 1, 1))
            @test ≈(ioutv, outv, rtol=approx_rtol)

            B = rand(T, 5, 3)
            outm = A'*B
            C = zero(outm)
            ioutm = imul!(copy(C), A', B, 1, 1)[1]
            @test check_inv(imul!, (copy(C), A', B, 1, 1))
            @test ≈(ioutm, outm, rtol=approx_rtol)
        end
    end    
end

@testset "dense matrix-sparse matrix multiplication" begin
    for T in (Float64, ComplexF64)    
        for i = 1:5
            B = rand(T, 10, 10)
            A = sprand(T, 10, 5, 0.5)
            outm = B*A
            C = zero(outm)
            ioutm = imul!(copy(C), B, A, 1 ,1)[1]
            @test check_inv(imul!, (copy(C), B, A, 1, 1)) 
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
            ioutm = imul!(copy(C), B', A, 1 ,1)[1]
            @test check_inv(imul!, (copy(C), B', A, 1, 1))  
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
            ioutm = imul!(copy(C), B, A', 1 ,1)[1] 
            @test check_inv(imul!, (copy(C), B, A', 1, 1))
            @test ≈(ioutm, outm, rtol=approx_rtol)
        end
    end
end