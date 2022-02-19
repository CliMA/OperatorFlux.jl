using OperatorFlux
using Test
using Flux
using LinearAlgebra

@testset "SpectralConv - FourierTransform" begin
    # auxiliary data
    M = 32

    # test constructor
    trafo = FourierTransform(modes = (10, 23))
    ch = 3 => 13
    sc = SpectralConv(trafo, ch)
    @test ndims(sc.weights) == 4

    # test convolution
    a, b = -π, π
    x = (b - a) .* collect(0:(M - 1)) / M .+ a
    y = copy(x)
    z = sin.(x) * sin.(y)'
    z = z[:, :, :, :] # emulate channels and batches
    z = permutedims(z, (3, 1, 2, 4))
    modes = (12, 16)
    trafo = FourierTransform(modes = modes)
    ch = 1 => 1
    sc = SpectralConv(trafo, ch)
    sc.weights .= ones(eltype(z), 1, size(trafo)..., 1)
    @test norm(sc(z) - z) ≤ eps(norm(z))
    @test ndims(sc.weights) == 4

    for M in [31, 32]
        even = Bool(mod(M, 2)) ? false : true

        # test more than one out_channel
        ch = 1 => 2
        trafo = FourierTransform(modes = (16, 16), even = even)
        sc = SpectralConv(trafo, ch)
        x = rand(Float32, 1, M, M, 2)
        @test size(sc(x)) == (2, M, M, 2)

        # test different dimensionalities
        # 1d
        ch = 1 => 2
        trafo = FourierTransform(modes = (16,), even = even)
        sc = SpectralConv(trafo, ch)
        x = rand(Float32, 1, M, 2)
        @test size(sc(x)) == (2, M, 2)

        # 3d
        ch = 13 => 3
        trafo = FourierTransform(modes = (16, 16, 14), even = even)
        sc = SpectralConv(trafo, ch)
        x = rand(Float32, 13, M, M, M, 2)
        @test size(sc(x)) == (3, M, M, M, 2)

        # 4d
        ch = 13 => 7
        trafo = FourierTransform(modes = (16, 16, 14, 3), even = even)
        sc = SpectralConv(trafo, ch)
        x = rand(Float32, 13, M, M, M, 63, 2)
        @test size(sc(x)) == (7, M, M, M, 63, 2)
    end

    # test Base extensions
    trafo = FourierTransform(modes = (10,))
    ch = 3 => 13
    sc = SpectralConv(trafo, ch)
    @test ndims(sc) == 1
end

@testset "SpectralConv - ChebyshevTransformTransform" begin
    # auxiliary data
    M = 32

    # test constructor
    trafo = ChebyshevTransform(modes = (10, 23))
    ch = 3 => 13
    sc = SpectralConv(trafo, ch)
    @test ndims(sc.weights) == 4

    # test convolution
    a, b = -π, π
    x = @. (cos(pi * (0:M) / M) + 1) / 2
    x = (b - a) .* x .+ a
    y = copy(x)
    z = x * y'
    z = z[:, :, :, :] # emulate channels and batches
    z = permutedims(z, (3, 1, 2, 4))
    modes = (12, 17)
    trafo = ChebyshevTransform(modes = modes)
    ch = 1 => 1
    sc = SpectralConv(trafo, ch)
    sc.weights .= ones(eltype(z), 1, collect(modes)..., 1)
    @test norm(sc(z) - z) ≤ 2 * eps(norm(z))
    @test ndims(sc.weights) == 4

    for M in [31, 32]
        # test more than one out_channel
        ch = 1 => 2
        trafo = ChebyshevTransform(modes = (16, 16))
        sc = SpectralConv(trafo, ch)
        x = rand(Float32, 1, M, M, 2)
        @test size(sc(x)) == (2, M, M, 2)

        # test different dimensionalities
        # 1d
        ch = 1 => 2
        trafo = ChebyshevTransform(modes = (16,))
        sc = SpectralConv(trafo, ch)
        x = rand(Float32, 1, M, 2)
        @test size(sc(x)) == (2, M, 2)

        # 3d
        ch = 13 => 3
        trafo = ChebyshevTransform(modes = (16, 16, 14))
        sc = SpectralConv(trafo, ch)
        x = rand(Float32, 13, M, M, M, 2)
        @test size(sc(x)) == (3, M, M, M, 2)

        # 4d
        ch = 13 => 7
        trafo = ChebyshevTransform(modes = (16, 16, 14, 3))
        sc = SpectralConv(trafo, ch)
        x = rand(Float32, 13, M, M, M, 63, 2)
        @test size(sc(x)) == (7, M, M, M, 63, 2)
    end

    # test Base extensions
    trafo = ChebyshevTransform(modes = (10,))
    ch = 3 => 13
    sc = SpectralConv(trafo, ch)
    @test ndims(sc) == 1
end


@testset "SpectralConv - LegendreTransform" begin
    # auxiliary data / legendre polynomials
    P₀(x) = 1
    P₁(x) = x
    P₂(x) = 0.5 * (3 * x^2 - 1)
    P₃(x) = 0.5 * (5 * x^3 - 3 * x)
    P₄(x) = 0.125 * (35 * x^4 - 30 * x^2 + 3)
    P₅(x) = 0.125 * (63 * x^5 - 70 * x^3 + 15 * x)
    polynomials = [P₀, P₁, P₂, P₃, P₄, P₅]

    # test constructors
    for mode in [(4,), (4, 5), (4, 5, 6)]
        trafo = LegendreTransform(modes = mode)
        ch = 3 => 13
        sc = SpectralConv(trafo, ch)
        @test ndims(sc.weights) == length(mode) + 2
    end

    # rountrip integration test
    for mode in [(5,), (5, 6), (5, 7, 9)]
        z = rand(1, mode..., 5)
        trafo = LegendreTransform(modes = mode)
        ch = 1 => 1
        sc = SpectralConv(trafo, ch)
        sc.weights .= ones(eltype(z), 1, collect(mode)..., 1)
        @test norm(sc(z) - z) ≤ 10 * eps(norm(z))
    end

    # test more than one out_channel
    for mode in [(5,), (5, 6), (5, 7, 9)]
        for c in [2, 3]
            z = rand(1, mode..., 5)
            trafo = LegendreTransform(modes = mode)
            ch = 1 => c
            sc = SpectralConv(trafo, ch)
            sc.weights .= ones(eltype(z), 1, collect(mode)..., 1)
            @test size(sc(z)) == (c, mode..., 5)
        end
    end

    # test Base extensions
    trafo = LegendreTransform(modes = (10,))
    ch = 3 => 13
    sc = SpectralConv(trafo, ch)
    @test ndims(sc) == 1
end
