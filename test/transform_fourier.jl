using OperatorFlux
using Test
using LinearAlgebra

@testset "FourierTransform" begin
    # test constructors
    trafo = FourierTransform(modes = (3, 7))
    @test trafo.modes == (3, 7)
    @test trafo.even == true

    # test forward & inverse
    for mode in [(4,), (4, 6), (4, 7, 9), (5,), (5, 6), (5, 7, 9)]
        z = rand(3, mode..., 5)
        even = Bool(mod(mode[1], 2)) ? false : true
        trafo = FourierTransform(modes = mode, even = even)
        @test norm(
            OperatorFlux.inverse(trafo, OperatorFlux.forward(trafo, z)) - z,
        ) ≤ 20 * eps(norm(z))
    end

    # test truncate_modes & pad_modes
    M = 32
    trafo = FourierTransform(modes = (3, 7, 6))
    c = rand(7, M, M, M, 3)
    inds = [
        [1, 2, 3],
        [1, 2, 3, 4, 5, 6, 7, 26, 27, 28, 29, 30, 31, 32],
        [1, 2, 3, 4, 5, 6, 27, 28, 29, 30, 31, 32],
    ]
    @test all(OperatorFlux.truncate_modes(trafo, c) .== c[:, inds..., :])

    trafo = FourierTransform(modes = (3, 6))
    c = rand(7, M, M, 3)
    inds = [[1, 2, 3], [1, 2, 3, 4, 5, 6, 27, 28, 29, 30, 31, 32]]
    @test all(OperatorFlux.truncate_modes(trafo, c) .== c[:, inds..., :])

    trafo = FourierTransform(modes = (3,))
    c = rand(7, M, 3)
    inds = [[1, 2, 3]]
    @test all(OperatorFlux.truncate_modes(trafo, c) .== c[:, inds..., :])

    trafo = FourierTransform(modes = (12, 5))
    c = rand(27, 16, 14, 1)
    inds = [
        [(1:(div(m, 2) + 1))..., ((s - div(m, 2) + 2):s)...] for
        (s, m) in zip((M, 14), size(c)[2:(end - 1)])
    ]
    inds[1] = collect(1:16)
    @test all(OperatorFlux.pad_modes(trafo, c, (32, 14))[:, inds..., :] .== c)

    # rountrip integration tests
    for M in [32, 33]
        a, b = -π, π
        x = (b - a) .* collect(0:(M - 1)) / M .+ a
        y = copy(x)
        z = sin.(2x) * cos.(2y)' + sin.(2x) * cos.(3y)'
        z = z[:, :, :, :] # emulate channels and batches
        z = permutedims(z, (3, 1, 2, 4))
        even = Bool(mod(M, 2)) ? false : true
        trafo = FourierTransform(modes = (12, 17), even = even)

        a = OperatorFlux.forward(trafo, z)
        b = OperatorFlux.truncate_modes(trafo, a)
        c = OperatorFlux.pad_modes(trafo, b, size(a)[2:(end - 1)])
        d = OperatorFlux.inverse(trafo, c)
        @test norm(d - z) ≤ 3 * eps(norm(z))
    end

    # test Base extensions
    trafo = FourierTransform(modes = (5, 11))
    @test ndims(trafo) == 2
    @test eltype(trafo) == ComplexF32
    @test size(trafo) == [5, 22]
end
