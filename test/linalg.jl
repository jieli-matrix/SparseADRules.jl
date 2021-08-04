using NiSparseArrays:imul!
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

