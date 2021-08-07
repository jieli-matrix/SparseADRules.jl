include("utils.jl")

@testset "jacobian on matrix multiplication" begin
    for T in (Float64, ComplexF64)
        A = sprand(T, 10, 5, 0.2)
        b = randn(T, 5)
        JFb = forwarddiff_mm_jacobian(A, b)
        JNb = nilang_mm_jacobian(A, b)
        @test isapprox(JFb, JNb)

        B = randn(T, 5, 3)
        JFB = forwarddiff_mm_jacobian(A, B)
        JNB = nilang_mm_jacobian(A, B)
        @test isapprox(JFB, JNB)
    end
end

@testset "jacobian on adjoint matrix multiplication" begin
    for T in (Float64, ComplexF64)
        A = sprand(T, 5, 10, 0.2)
        b = randn(T, 5)
        JFb = forwarddiff_mm_jacobian(A', b)
        JNb = nilang_mm_jacobian(A', b)
        @test isapprox(JFb, JNb)

        B = randn(T, 5, 3)
        JFB = forwarddiff_mm_jacobian(A', B)
        JNB = nilang_mm_jacobian(A', B)
        @test isapprox(JFB, JNB)
    end
end
