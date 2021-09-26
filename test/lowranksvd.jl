using NiSparseArrays:get_approximate_basis, low_rank_svd
@testset "low rank svd" begin
    for T in (Float64, ComplexF64)
        approx_rtol = 100*eps(real(T))
        for (n, m) in ((64,32), (32,32))
            for i = 1:5
                A = sprand(n, m, 0.1)
                r = rank(A) # sparse matrix not always low rank 
                @testset "Q basis for A" begin
                    Q = get_approximate_basis(A, r)
                    @test norm(A - Q*Q'*A) < approx_rtol*norm(A)
                end
                
                @testset "svd test for A" begin
                    U, S, Vt = low_rank_svd(A, r)
                    A_approx = U * Diagonal(S) * Vt
                    @test norm(A - A_approx) < approx_rtol*norm(A)
                end

                M = rand(1, m)
                A_m = A .- M 
                r_m = rank(A_m)
                @testset "Q basis for A - M" begin
                    Q = get_approximate_basis(A, r_m, 2, M)
                    @test norm(A_m - Q*Q'*A_m) < approx_rtol*norm(A_m)
                end
                
                @testset "svd test for A - M" begin
                    U, S, Vt = low_rank_svd(A, r_m, 2, M)
                    A_m_approx = U * Diagonal(S) * Vt
                    @test norm(A_m - A_m_approx) < approx_rtol*norm(A_m)
                end
            end    
        end
    end
end