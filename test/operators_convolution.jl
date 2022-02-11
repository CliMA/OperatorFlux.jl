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
    x = (b - a) .* collect(0:(M-1)) / M .+ a
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

    # test more than one out_channel
    ch = 1 => 2
    trafo = FourierTransform(modes = (16, 16))
    sc = SpectralConv(trafo, ch)
    x = rand(Float32, 1, 32, 32, 2)
    @test size(sc(x)) == (2, 32, 32, 2)

    # test different dimensionalities
    # 1d
    ch = 1 => 2
    trafo = FourierTransform(modes = (16,))
    sc = SpectralConv(trafo, ch)
    x = rand(Float32, 1, 32, 2)
    @test size(sc(x)) == (2, 32, 2)

    # 3d
    ch = 13 => 3
    trafo = FourierTransform(modes = (16, 16, 14))
    sc = SpectralConv(trafo, ch)
    x = rand(Float32, 13, 32, 32, 32, 2)
    @test size(sc(x)) == (3, 32, 32, 32, 2)

    # 4d
    ch = 13 => 7
    trafo = FourierTransform(modes = (16, 16, 14, 3))
    sc = SpectralConv(trafo, ch)
    x = rand(Float32, 13, 32, 32, 32, 63, 2)
    @test size(sc(x)) == (7, 32, 32, 32, 63, 2)

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

    # test more than one out_channel
    ch = 1 => 2
    trafo = ChebyshevTransform(modes = (16, 16))
    sc = SpectralConv(trafo, ch)
    x = rand(Float32, 1, 32, 32, 2)
    @test size(sc(x)) == (2, 32, 32, 2)

    # test different dimensionalities
    # 1d
    ch = 1 => 2
    trafo = ChebyshevTransform(modes = (16,))
    sc = SpectralConv(trafo, ch)
    x = rand(Float32, 1, 32, 2)
    @test size(sc(x)) == (2, 32, 2)

    # 3d
    ch = 13 => 3
    trafo = ChebyshevTransform(modes = (16, 16, 14))
    sc = SpectralConv(trafo, ch)
    x = rand(Float32, 13, 32, 32, 32, 2)
    @test size(sc(x)) == (3, 32, 32, 32, 2)

    # 4d
    ch = 13 => 7
    trafo = ChebyshevTransform(modes = (16, 16, 14, 3))
    sc = SpectralConv(trafo, ch)
    x = rand(Float32, 13, 32, 32, 32, 63, 2)
    @test size(sc(x)) == (7, 32, 32, 32, 63, 2)

    # test Base extensions
    trafo = ChebyshevTransform(modes = (10,))
    ch = 3 => 13
    sc = SpectralConv(trafo, ch)
    @test ndims(sc) == 1
end

@testset "SpectralConv - SpectralElementTransform" begin
    # auxiliary data / legendre polynomials
    P₀(x) = 1
    P₁(x) = x
    P₂(x) = 0.5 * (3 * x^2 - 1)
    P₃(x) = 0.5 * (5 * x^3 - 3 * x)
    P₄(x) = 0.125 * (35 * x^4 - 30 * x^2 + 3)
    P₅(x) = 0.125 * (63 * x^5 - 70 * x^3 + 15 * x)
    polynomials = [P₀, P₁, P₂, P₃, P₄, P₅]

    # test constructor
    trafo = SpectralElementTransform(modes = (10, 23))
    ch = 3 => 13
    sc = SpectralConv(trafo, ch)
    @test ndims(sc.weights) == 4

    # # test convolution
    # a, b = -π, π
    # x = @. (cos(pi * (0:M) / M) + 1) / 2
    # x = (b - a) .* x .+ a
    # y = copy(x)
    # z = x * y'
    # z = z[:, :, :, :] # emulate channels and batches
    # z = permutedims(z, (3, 1, 2, 4))
    # modes = (12, 17)
    # trafo = ChebyshevTransform(modes = modes)
    # ch = 1 => 1
    # sc = SpectralConv(trafo, ch)
    # sc.weights .= ones(eltype(z), 1, collect(modes)..., 1)
    # @test norm(sc(z) - z) ≤ 2 * eps(norm(z))
    # @test ndims(sc.weights) == 4

    # # test more than one out_channel
    # ch = 1 => 2
    # trafo = ChebyshevTransform(modes = (16, 16))
    # sc = SpectralConv(trafo, ch)
    # x = rand(Float32, 1, 32, 32, 2)
    # @test size(sc(x)) == (2, 32, 32, 2)

    # # test different dimensionalities
    # # 1d
    # ch = 1 => 2
    # trafo = ChebyshevTransform(modes = (16,))
    # sc = SpectralConv(trafo, ch)
    # x = rand(Float32, 1, 32, 2)
    # @test size(sc(x)) == (2, 32, 2)

    # # 3d
    # ch = 13 => 3
    # trafo = ChebyshevTransform(modes = (16, 16, 14))
    # sc = SpectralConv(trafo, ch)
    # x = rand(Float32, 13, 32, 32, 32, 2)
    # @test size(sc(x)) == (3, 32, 32, 32, 2)

    # # 4d
    # ch = 13 => 7
    # trafo = ChebyshevTransform(modes = (16, 16, 14, 3))
    # sc = SpectralConv(trafo, ch)
    # x = rand(Float32, 13, 32, 32, 32, 63, 2)
    # @test size(sc(x)) == (7, 32, 32, 32, 63, 2)

    # # test Base extensions
    # trafo = ChebyshevTransform(modes = (10,))
    # ch = 3 => 13
    # sc = SpectralConv(trafo, ch)
    # @test ndims(sc) == 1
end
