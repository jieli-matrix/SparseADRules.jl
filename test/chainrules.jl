@testset "test rrule" begin
    @testset "mul" begin
        # sparse * vector
        A = sprand(5, 3, 0.5)
        B = rand(3)
        test_rrule(*, A ⊢ sprand_tangent(A), B; check_thunked_output_tangent=false)
        # test_rrule(*, A ⊢ sprand_tangent(A), B) # FIXME: Thunk doesn't yet supported.

        # sparse * dense matrix
        A = sprand(5, 3, 0.5)
        B = rand(3, 2)
        test_rrule(*, A ⊢ sprand_tangent(A), B; check_thunked_output_tangent=false)
        # test_rrule(*, A ⊢ sprand_tangent(A), B) # FIXME: Thunk doesn't yet supported.
    end
end
