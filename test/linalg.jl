using NiSparseArrays:idotxA, idotAx
const approx_rtol = 100*eps()

@testset "sparse dot xA" begin
    for T in (Float64, ComplexF64)    
        for i = 1:5
            x = sprand(T, 10, 0.2)
            A = sprand(T, 10, 5, 0.2)
            y = sprand(T, 5, 0.3)
            outd = dot(x, A, y)
            r = zero(T)
            ioutd = idotxA(copy(r), x, A, y)[1]
            @test ≈(ioutd, outd, rtol=approx_rtol)
        end
    end
end

@testset "sparse dot Ax" begin
    for T in (Float64, ComplexF64)    
        for i = 1:5
            x = sprand(T, 10, 0.2)
            A = sprand(T, 10, 5, 0.2)
            y = sprand(T, 5, 0.3)
            outd = dot(x, A, y)
            r = zero(T)
            ioutd = idotAx(copy(r), x, A, y)[1]
            @test ≈(ioutd, outd, rtol=approx_rtol)
        end
    end
end

# m, n = 1000, 500;
# x = sprand(m, 0.2)
# A = sprand(m, n, 0.2)
# y = sprand(n, 0.3)
# r = zero(eltype(A))

# @testset "SparseArrays dot benchmark" begin
#     sdb = @benchmarkable SparseArrays.dot($x, $A, $y)
#     run(sdb)
#     plot(sdb)
# end

# @testset "idotxA benchmark" begin    
#     xAb = @benchmarkable idotxA($copy(r), $x, $A, $y)
#     run(xAb)
#     plot(xAb)
# end

# @testset "idotAx benchmark" begin
#     Axb = @benchmarkable idotAx($copy(r), $x, $A, $y)
#     run(Axb)
#     plot(Axb)
# end