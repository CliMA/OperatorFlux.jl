using OperatorFlux
using Test
using FFTW
using Flux
using LinearAlgebra
using ChainRulesCore
using ChainRulesTestUtils

@testset "FourierTransform" begin
    # auxiliary data
    M = 32

    # test constructors
    trafo = FourierTransform(modes = (3, 7))
    @test trafo.modes == (3, 7)

    # test forward & inverse
    trafo = FourierTransform(modes = (16, 16))
    z = rand(5, M, M, 4)
    @test size(OperatorFlux.forward(trafo, z)) == (5, M, M, 4)
    @test norm(OperatorFlux.inverse(trafo, forward(trafo, z)) - z) ≤
          2 * eps(norm(z))
    @test norm(OperatorFlux.inverse(trafo, forward(trafo, z, 2:2), 2:2) - z) ≤
          2 * eps(norm(z))
    @test norm(OperatorFlux.inverse(trafo, forward(trafo, z, (3,)), (3,)) - z) ≤
          2 * eps(norm(z))

    # test truncate_modes
    trafo = FourierTransform(modes = (3, 7, 6))
    c = rand(7, M, M, M, 3)
    inds = [
        [1, 2, 3, 4, 31, 32],
        [1, 2, 3, 4, 5, 6, 7, 8, 27, 28, 29, 30, 31, 32],
        [1, 2, 3, 4, 5, 6, 7, 28, 29, 30, 31, 32],
    ]
    @test all(OperatorFlux.truncate_modes(trafo, c) .== c[:, inds..., :])

    trafo = FourierTransform(modes = (3, 6))
    dims = (2, 4)
    inds =
        [[1, 2, 3, 4, 31, 32], 1:32, [1, 2, 3, 4, 5, 6, 7, 28, 29, 30, 31, 32]]
    @test all(OperatorFlux.truncate_modes(trafo, c, dims) .== c[:, inds..., :])

    trafo = FourierTransform((3,), (2,))
    dims = trafo.dims
    inds = [[1, 2, 3, 4, 31, 32], 1:32, 1:32]
    @test all(OperatorFlux.truncate_modes(trafo, c, dims) .== c[:, inds..., :])

    # test pad_modes
    trafo = FourierTransform(modes = (12, 5))
    c = rand(27, 16, 14, 1)
    inds = [
        [(1:(div(m, 2) + 1))..., ((s - div(m, 2) + 2):s)...]
        for (s, m) in zip((M, 14), size(c)[2:(end - 1)])
    ]
    @test all(OperatorFlux.pad_modes(trafo, c, (M, 14))[:, inds..., :] .== c)

    # rountrip integration test
    a, b = -π, π
    x = (b - a) .* collect(0:(M - 1)) / M .+ a
    y = copy(x)
    z = sin.(x) * sin.(y)'
    z = z[:, :, :, :] # emulate channels and batches
    z = permutedims(z, (3, 1, 2, 4))
    trafo = FourierTransform(modes = (12, 17))
    a = OperatorFlux.forward(trafo, z)
    b = OperatorFlux.truncate_modes(trafo, a)
    c = OperatorFlux.pad_modes(trafo, b, size(a)[2:(end - 1)])
    d = OperatorFlux.inverse(trafo, c)
    @test norm(d - z) ≤ eps(norm(z))

    trafo = FourierTransform((12,), (2,))
    a = OperatorFlux.forward(trafo, z, (2,))
    b = OperatorFlux.truncate_modes(trafo, a, (2,))
    c = OperatorFlux.pad_modes(trafo, b, size(a)[2:(end - 1)])
    d = OperatorFlux.inverse(trafo, c, (2,))
    @test norm(d - z) ≤ eps(norm(z))

    # test Base extensions
    trafo = FourierTransform(modes = (5, 11))
    @test ndims(trafo) == 2
    @test eltype(trafo) == ComplexF32
    @test size(trafo) == [10, 22]
end

@testset "FourierTransform - ChainRulesCore" begin
    # test ChainRulesCore extensions
    M = 4
    trafo = FourierTransform(modes = (2, 2))
    c = rand(3, M, M, 5)
    dims = 2:3
    # check_thunked_output_tangent checks that thunked objects pass through
    # in order to make this work we need to pass through the dims 
    # and not use size(c), for example, in the code
    test_rrule(
        OperatorFlux.truncate_modes,
        trafo,
        c,
        dims,
        check_thunked_output_tangent = false,
    )
    test_rrule(
        OperatorFlux.pad_modes,
        trafo,
        c,
        (2 * M, 2 * M),
        check_thunked_output_tangent = false,
    )
end
