using OperatorFlux
using Test
using FFTW
using Flux
using LinearAlgebra

@testset "ChebyshevTransform" begin
    # auxiliary data
    M = 32

    # test constructors
    trafo = ChebyshevTransform(modes = (3, 7))
    @test trafo.modes == (3, 7)
    @test trafo.dims == 2:3

    # test forward & inverse
    trafo = ChebyshevTransform(modes = (16, 16))
    z = rand(5, M, 18, 4)
    @test size(OperatorFlux.forward(trafo, z)) == (5, M, 18, 4)
    @test norm(OperatorFlux.inverse(trafo, forward(trafo, z)) - z) ≤
          4 * eps(norm(z))
    @test norm(OperatorFlux.inverse(trafo, forward(trafo, z, 2:2), 2:2) - z) ≤
          4 * eps(norm(z))
    OperatorFlux.inverse(trafo, forward(trafo, z, (3,)), (3,))
    @test norm(OperatorFlux.inverse(trafo, forward(trafo, z, (3,)), (3,)) - z) ≤
          3 * eps(norm(z))

    # test truncate_modes
    trafo = ChebyshevTransform(modes = (3, 7, 6))
    c = rand(7, M, M, M, 3)
    inds = [collect(1:m) for m in trafo.modes]
    @test all(OperatorFlux.truncate_modes(trafo, c) .== c[:, inds..., :])

    trafo = ChebyshevTransform(modes = (3, 6))
    dims = (2, 4)
    inds = [[1, 2, 3], 1:32, [1, 2, 3, 4, 5, 6]]
    @test all(OperatorFlux.truncate_modes(trafo, c, dims) .== c[:, inds..., :])

    trafo = ChebyshevTransform((3,), (2,))
    dims = trafo.dims
    inds = [[1, 2, 3], 1:32, 1:32]
    @test all(OperatorFlux.truncate_modes(trafo, c, dims) .== c[:, inds..., :])

    # test pad_modes
    trafo = ChebyshevTransform(modes = (12, 5))
    c = rand(27, 16, 14, 1)
    inds = [collect(1:m) for m in size(c)[2:(end - 1)]]
    @test all(OperatorFlux.pad_modes(trafo, c, (M, M))[:, inds..., :] .== c)

    # rountrip integration test
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

    # test Base extensions
    trafo = ChebyshevTransform(modes = (5, 11))
    @test ndims(trafo) == 2
    @test eltype(trafo) == Float32
    @test size(trafo) == [5, 11]
end
