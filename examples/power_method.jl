using Zygote
using SparseArrays, LinearAlgebra
using BenchmarkTools
function power_max_eigen(A, x, target; niter=100)
    for i=1:niter
        x = A * x
        x /= norm(x)
    end
    return abs(x' * target)
end

function power_min_eigen(A, x, target, λ; niter=100)
    A = λ*I - A 
    for i=1:niter
        x = A * x
        x /= norm(x)
    end
    return abs(x' * target)
end


A = sprand(100, 100, 0.1)
x = randn(100)
target = randn(100)
λ = 2
@btime max_ga_z, max_gx_z, max_gt_z = Zygote.gradient(power_max_eigen, A, x, target) # 3.694 ms (3198 allocations: 12.26 MiB)
@btime min_ga_z, min_gx_z, min_gt_z = Zygote.gradient(power_min_eigen, A, x, target, λ) # 4.031 ms (3438 allocations: 12.62 MiB)

using NiSparseArrays
@btime max_ga, max_gx, max_gt = Zygote.gradient(power_max_eigen, A, x, target) # 1.413 ms (3398 allocations: 7.99 MiB)
@btime min_ga, min_gx, min_gt = Zygote.gradient(power_min_eigen, A, x, target, λ) #  1.610 ms (3738 allocations: 8.62 MiB)

using Test

max_ga_z, max_gx_z, max_gt_z = Zygote.gradient(power_max_eigen, A, x, target)
min_ga_z, min_gx_z, min_gt_z = Zygote.gradient(power_min_eigen, A, x, target, λ)
max_ga, max_gx, max_gt = Zygote.gradient(power_max_eigen, A, x, target)
min_ga, min_gx, min_gt = Zygote.gradient(power_min_eigen, A, x, target, λ)

@testset begin
    @testset "power method for max eigenvalue" begin
        @test max_ga_z ≈ max_ga
        @test max_gx_z ≈ max_gx
        @test max_gt_z ≈ max_gt
    end
    @testset "power method for min eigenvalue" begin
        @test min_ga_z ≈ min_ga
        @test min_gx_z ≈ min_gx
        @test min_gt_z ≈ min_gt
    end
end