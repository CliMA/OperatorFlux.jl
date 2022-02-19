using OperatorFlux
using Test
using LinearAlgebra

@testset "ChebyshevTransform" begin
    # auxiliary data
    M = 32

    # test constructors
    trafo = ChebyshevTransform(modes = (3, 7))
    @test trafo.modes == (3, 7)

    # test forward & inverse
    for mode in [(4,), (4, 6), (4, 7, 9), (5,), (5, 6), (5, 7, 9)]
        z = rand(3, mode..., 5)
        trafo = ChebyshevTransform(modes = mode)
        @test norm(
            OperatorFlux.inverse(trafo, OperatorFlux.forward(trafo, z)) - z,
        ) ≤ 20 * eps(norm(z))
    end

    # test truncate_modes & pad_modes
    M = 32
    trafo = ChebyshevTransform(modes = (3, 7, 6))
    c = rand(7, M, M, M, 3)
    inds = [[1, 2, 3], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6]]
    @test all(OperatorFlux.truncate_modes(trafo, c) .== c[:, inds..., :])

    trafo = ChebyshevTransform(modes = (3, 6))
    c = rand(7, M, M, 3)
    inds = [[1, 2, 3], [1, 2, 3, 4, 5, 6]]
    @test all(OperatorFlux.truncate_modes(trafo, c) .== c[:, inds..., :])

    trafo = ChebyshevTransform(modes = (3,))
    c = rand(7, M, 3)
    inds = [[1, 2, 3]]
    @test all(OperatorFlux.truncate_modes(trafo, c) .== c[:, inds..., :])

    trafo = ChebyshevTransform(modes = (12, 5))
    c = rand(27, 16, 14, 1)
    inds = [collect(1:m) for m in size(c)[2:(end - 1)]]
    @test all(OperatorFlux.pad_modes(trafo, c, (32, 14))[:, inds..., :] .== c)

    # rountrip integration test
    for M in [32, 33]
        a, b = -π, π
        x = @. (cos(pi * (0:M) / M) + 1) / 2
        x = (b - a) .* x .+ a
        y = copy(x)
        z = x * y'
        z = z[:, :, :, :] # emulate channels and batches
        z = permutedims(z, (3, 1, 2, 4))
        trafo = ChebyshevTransform(modes = (12, 17))
        a = OperatorFlux.forward(trafo, z)
        b = OperatorFlux.truncate_modes(trafo, a)
        c = OperatorFlux.pad_modes(trafo, b, size(a)[2:(end - 1)])
        d = OperatorFlux.inverse(trafo, c)
        @test norm(d - z) ≤ eps(10 * norm(z))
    end

    # test Base extensions
    trafo = ChebyshevTransform(modes = (5, 11))
    @test ndims(trafo) == 2
    @test eltype(trafo) == Float32
    @test size(trafo) == [5, 11]
end
