using NiSparseArrays: idot, i_spmatmul
const approx_rtol = 100*eps()

@testset "simple spatmul implementaation" begin
    for T in (Float64, ComplexF64)
        for i = 1:5
            A = sprand(T, 10,10,0.2);
            B = sprand(T, 10,10,0.3);
            C = spmatmul(A, B);
            iC = i_spmatmul(A, B);
            @test ≈(C, iC, rtol=approx_rtol)
        end
    end
end

@testset "sparse matrix - sparse matrix dot" begin
    for T in (Float64, ComplexF64)
        for i = 1:5
            A = sprand(T, 10, 5, 0.2)
            B = sprand(T, 10, 5, 0.3)
            outd = dot(A, B)
            r = zero(T)
            ioutd = idot(copy(r), A, B)[1]
            @test ≈(ioutd, outd, rtol=approx_rtol)
        end
    end
end