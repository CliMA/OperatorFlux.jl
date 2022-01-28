using OperatorFlux
using Test
using FFTW
using Flux
using LinearAlgebra

@testset "FourierTransform" begin
    # auxiliary data
    M = 32

    # test constructors
    trafo = FourierTransform(modes = (3, 7))
    @test trafo.modes == (3, 7)
    @test trafo.dims == 2:3

    # test forward & inverse
    trafo = FourierTransform(modes = (16, 16))
    z = rand(5, M, 18, 4)
    @test size(OperatorFlux.forward(trafo, z)) == (5, div(M, 2) + 1, 18, 4)
    @test norm(OperatorFlux.inverse(trafo, forward(trafo, z)) - z) ≤
          3 * eps(norm(z))
    @test norm(OperatorFlux.inverse(trafo, forward(trafo, z, 2:2), 2:2) - z) ≤
          3 * eps(norm(z))
    @test norm(OperatorFlux.inverse(trafo, forward(trafo, z, (3,)), (3,)) - z) ≤
          3 * eps(norm(z))

    # test truncate_modes & pad_modes
    trafo = FourierTransform(modes = (3, 7, 6))
    c = rand(7, M, M, M, 3)
    inds = [
        [1, 2, 3],
        [1, 2, 3, 4, 5, 6, 7, 26, 27, 28, 29, 30, 31, 32],
        [1, 2, 3, 4, 5, 6, 27, 28, 29, 30, 31, 32],
    ]
    @test all(OperatorFlux.truncate_modes(trafo, c) .== c[:, inds..., :])

    trafo = FourierTransform(modes = (3, 6))
    dims = (2, 4)
    inds = [[1, 2, 3], 1:32, [1, 2, 3, 4, 5, 6, 27, 28, 29, 30, 31, 32]]
    @test all(OperatorFlux.truncate_modes(trafo, c, dims) .== c[:, inds..., :])

    trafo = FourierTransform((3,), (2,))
    dims = trafo.dims
    inds = [[1, 2, 3], 1:32, 1:32]
    @test all(OperatorFlux.truncate_modes(trafo, c, dims) .== c[:, inds..., :])

    trafo = FourierTransform(modes = (12, 5))
    c = rand(27, 16, 14, 1)
    inds = [
        [(1:(div(m, 2) + 1))..., ((s - div(m, 2) + 2):s)...]
        for (s, m) in zip((M, 14), size(c)[2:(end - 1)])
    ]
    inds[1] = collect(1:16)
    @test all(OperatorFlux.pad_modes(trafo, c, (M, 14))[:, inds..., :] .== c)

    # rountrip integration tests
    a, b = -π, π
    x = (b - a) .* collect(0:(M - 1)) / M .+ a
    y = copy(x)
    z = sin.(2x) * cos.(2y)' + sin.(2x) * cos.(3y)'
    z = z[:, :, :, :] # emulate channels and batches
    z = permutedims(z, (3, 1, 2, 4))
    trafo = FourierTransform(modes = (12, 17))
    a = OperatorFlux.forward(trafo, z)
    b = OperatorFlux.truncate_modes(trafo, a)
    c = OperatorFlux.pad_modes(trafo, b, size(a)[2:(end - 1)])
    d = OperatorFlux.inverse(trafo, c)
    @test norm(d - z) ≤ 3 * eps(norm(z))

    trafo = FourierTransform((12,), (2,))
    a = OperatorFlux.forward(trafo, z, (2,))
    b = OperatorFlux.truncate_modes(trafo, a, (2,))
    c = OperatorFlux.pad_modes(trafo, b, size(a)[2:(end - 1)])
    d = OperatorFlux.inverse(trafo, c, (2,))
    @test norm(d - z) ≤ 2 * eps(norm(z))

    # test Base extensions
    trafo = FourierTransform(modes = (5, 11))
    @test ndims(trafo) == 2
    @test eltype(trafo) == ComplexF32
    @test size(trafo) == [5, 22]
end
