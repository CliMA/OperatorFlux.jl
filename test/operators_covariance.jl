using OperatorFlux
using Test
using Flux
using LinearAlgebra

@testset "SpectralCovariance - FourierTransform" begin
    # auxiliary data
    M = 32

    # test constructor
    trafo = FourierTransform(modes = (3, 3))
    ch = 3 => 1
    rank = 2
    sc = SpectralCovariance(trafo, rank, ch)
    @test sc.transform == trafo
    @test sc.in_channels == 3
    @test sc.out_channels == 1
    @test ndims(sc.weights_μ) == 2
    @test ndims(sc.weights_d) == 2
    @test ndims(sc.weights_w) == 3
    @test size(sc.weights_μ) == (3, 1)
    @test size(sc.weights_d) == (3, 1)
    @test size(sc.weights_w) == (3, 2, 1)

    # test covariance
    trafo = FourierTransform(modes = (3, 3))
    ch = 3 => 1
    rank = 2
    sc = SpectralCovariance(trafo, rank, ch)
    M = 32
    z = rand(3, M, M, 12)
    μ, D, V = sc(z)
    @test size(μ) == (1, M, M, 12)
    @test size(D) == (1, M, M, 12)
    @test size(V) == (1, M, M, rank, 12)
    @test all(D .> 0)
end

@testset "SpectralCovariance - ChebyshevTransform" begin
    # auxiliary data
    M = 32

    # test constructor
    trafo = ChebyshevTransform(modes = (3, 3))
    ch = 3 => 1
    rank = 2
    sc = SpectralCovariance(trafo, rank, ch)
    @test sc.transform == trafo
    @test sc.in_channels == 3
    @test sc.out_channels == 1
    @test ndims(sc.weights_μ) == 2
    @test ndims(sc.weights_d) == 2
    @test ndims(sc.weights_w) == 3
    @test size(sc.weights_μ) == (3, 1)
    @test size(sc.weights_d) == (3, 1)
    @test size(sc.weights_w) == (3, 2, 1)

    # test covariance
    trafo = ChebyshevTransform(modes = (3, 3))
    ch = 3 => 1
    rank = 2
    sc = SpectralCovariance(trafo, rank, ch)
    M = 32
    z = rand(3, M, M, 12)
    μ, D, V = sc(z)
    @test size(μ) == (1, M, M, 12)
    @test size(D) == (1, M, M, 12)
    @test size(V) == (1, M, M, rank, 12)
    @test all(D .> 0)
end

@testset "SpectralCovariance - LegendreTransform" begin
    # auxiliary data
    M = 32

    # test constructor
    trafo = LegendreTransform(modes = (3, 3))
    ch = 3 => 1
    rank = 2
    sc = SpectralCovariance(trafo, rank, ch)
    @test sc.transform == trafo
    @test sc.in_channels == 3
    @test sc.out_channels == 1
    @test ndims(sc.weights_μ) == 2
    @test ndims(sc.weights_d) == 2
    @test ndims(sc.weights_w) == 3
    @test size(sc.weights_μ) == (3, 1)
    @test size(sc.weights_d) == (3, 1)
    @test size(sc.weights_w) == (3, 2, 1)

    # test covariance
    trafo = LegendreTransform(modes = (3, 3))
    ch = 3 => 1
    rank = 2
    sc = SpectralCovariance(trafo, rank, ch)
    M = 32
    z = rand(3, M, M, 12)
    μ, D, V = sc(z)
    @test size(μ) == (1, M, M, 12)
    @test size(D) == (1, M, M, 12)
    @test size(V) == (1, M, M, rank, 12)
    @test all(D .> 0)
end
