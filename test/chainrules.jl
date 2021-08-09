@testset "test rrule" begin
    @testset "mul" begin
        # sparse * vector
        A = sprand(5, 3, 0.5)
        B = rand(3)
        C, C_pullback = rrule(*, A, B)
        
        C̄ = rand(5)
        _, Ā, B̄ = C_pullback(C̄)
        
        # Here I want to use FiniteDifferences.jl to check j'vp
        # j′vp(central_fdm(5, 1), f, x, C̄)[1]
        # but how to define f and x?
    end
end

