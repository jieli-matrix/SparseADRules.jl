using SparseArraysAD:private_qr
@testset "test rrule" begin
    @testset "matrix multiplication" begin
        # sparse * vector
        for T in (Float64, ComplexF64)
            A = sprand(T, 5, 3, 0.5)
            B = rand(T, 3)
            test_rrule(*, A ⊢ sprand_tangent(A), B; check_thunked_output_tangent=false)
            # test_rrule(*, A ⊢ sprand_tangent(A), B) # FIXME: Thunk doesn't yet supported.

            # sparse * dense matrix
            A = sprand(T, 5, 3, 0.5)
            B = rand(T, 3, 2)
            test_rrule(*, A ⊢ sprand_tangent(A), B; check_thunked_output_tangent=false)
            # test_rrule(*, A ⊢ sprand_tangent(A), B) # FIXME: Thunk doesn't yet supported.
        end 
    end

    @testset "adjoint matrix multiplication" begin
        for T in (Float64, ComplexF64)
            A = sprand(T, 5, 5, 0.2)
            b = rand(T, 5)
            test_rrule(*, A' ⊢ sprand_tangent(A'), b; check_thunked_output_tangent=false)

            A = sprand(T, 5, 5, 0.2)
            B = rand(T, 5, 3)
            test_rrule(*, A' ⊢ sprand_tangent(A'), B; check_thunked_output_tangent=false)
        end
    end


    @testset "dense matrix-sparse matrix multiplication" begin
        # dense matrix * sparse matrix
        for T in (Float64, ComplexF64)
            B = rand(T, 10, 10)
            A = sprand(T, 10, 5, 0.5)
            test_rrule(*, B, A ⊢ sprand_tangent(A); check_thunked_output_tangent=false)
        end 
    end

    @testset "adjoint dense matrix-sparse matrix multiplication" begin
    #     # adjoint dense matrix * sparse matrix
        for T in (Float64, ComplexF64)
            B = rand(T, 10, 5)
            A = sprand(T, 10, 5, 0.5)
            test_rrule(*, B', A ⊢ sprand_tangent(A); check_thunked_output_tangent=false)
        end 
    end

    @testset "dense vector - sparse matrix dot" begin
        for T in (Float64, ComplexF64)
            x = rand(T, 10)
            A = sprand(T, 10, 5, 0.2)
            y = rand(T, 5)
            test_rrule(dot, x, A ⊢ sprand_tangent(A), y; check_thunked_output_tangent=false)
        end
    end

    @testset "idot - sparse sparse" begin
        for T in (Float64, ComplexF64)
            A = sprand(T, 10, 5, 0.2)
            B = sprand(T, 10, 5, 0.2)
            test_rrule(dot, A ⊢ sprand_tangent(A), B ⊢ sprand_tangent(B); check_thunked_output_tangent=false)
        end
    end

    @testset "sparse vector - sparse matrix dot" begin
        for T in (Float64, ComplexF64)
            x = sprand(T, 10, 0.2)
            A = sprand(T, 10, 5, 0.2)
            y = sprand(T, 5, 0.3)
            test_rrule(dot, x ⊢ sprand_tangent(x), A ⊢ sprand_tangent(A), y ⊢ sprand_tangent(y); check_thunked_output_tangent=false)
        end
    end

    @testset "qr decomposition" begin
        for size in [(4, 4), (7, 4), (19, 5), (44, 17)]
            m,n = size
            X = randn(n,m)
            test_rrule(private_qr, X; check_thunked_output_tangent=false)
        end
    end
end

