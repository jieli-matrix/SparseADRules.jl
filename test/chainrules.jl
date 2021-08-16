@testset "test rrule" begin
    @testset "matrix multiplication" begin
        # sparse * vector
        #for T in (Float64)
            A = sprand(5, 3, 0.5)
            B = rand(3)
            test_rrule(*, A ⊢ sprand_tangent(A), B; check_thunked_output_tangent=false)
            # test_rrule(*, A ⊢ sprand_tangent(A), B) # FIXME: Thunk doesn't yet supported.

            # sparse * dense matrix
            A = sprand(5, 3, 0.5)
            B = rand(3, 2)
            test_rrule(*, A ⊢ sprand_tangent(A), B; check_thunked_output_tangent=false)
            # test_rrule(*, A ⊢ sprand_tangent(A), B) # FIXME: Thunk doesn't yet supported.
        #end 
    end

    @testset "dense matrix-sparse matrix multiplication" begin
        # dense matrix * sparse matrix
        #for T in (Float64)
            B = rand(10, 10)
            A = sprand(10, 5, 0.5)
            test_rrule(*, B, A ⊢ sprand_tangent(A); check_thunked_output_tangent=false)
        #end 
    end

    @testset "adjoint dense matrix-sparse matrix multiplication" begin
    #     # adjoint dense matrix * sparse matrix
    #     for T in (Float64)
            B = rand(10, 5)
            A = sprand(10, 5, 0.5)
            test_rrule(*, B', A ⊢ sprand_tangent(A); check_thunked_output_tangent=false)
    #     end 
    end
end
