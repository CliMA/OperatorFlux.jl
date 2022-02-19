using OperatorFlux
using Test
using GaussQuadrature
using LinearAlgebra

import OperatorFlux:
    legendre_transform_forward_matrix, legendre_transform_inverse_matrix

@testset "LegendreTransform" begin
    # auxiliary data / legendre polynomials
    P₀(x) = 1
    P₁(x) = x
    P₂(x) = 0.5 * (3 * x^2 - 1)
    P₃(x) = 0.5 * (5 * x^3 - 3 * x)
    P₄(x) = 0.125 * (35 * x^4 - 30 * x^2 + 3)
    P₅(x) = 0.125 * (63 * x^5 - 70 * x^3 + 15 * x)
    polynomials = [P₀, P₁, P₂, P₃, P₄, P₅]

    # test constructors
    trafo = LegendreTransform(modes = (4, 5, 6))
    @test trafo.modes == (4, 5, 6)

    # test forward & inverse
    for mode in [(5,), (5, 6), (5, 7, 9)]
        z = rand(3, mode..., 5)
        trafo = LegendreTransform(modes = mode)
        @test norm(
            OperatorFlux.inverse(trafo, OperatorFlux.forward(trafo, z)) - z,
        ) ≤ 20 * eps(norm(z))
    end

    # test truncate_modes & pad_modes
    M = 32
    trafo = LegendreTransform(modes = (3, 7, 6))
    c = rand(7, M, M, M, 3)
    inds = [[1, 2, 3], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6]]
    @test all(OperatorFlux.truncate_modes(trafo, c) .== c[:, inds..., :])

    trafo = LegendreTransform(modes = (3, 6))
    c = rand(7, M, M, 3)
    inds = [[1, 2, 3], [1, 2, 3, 4, 5, 6]]
    @test all(OperatorFlux.truncate_modes(trafo, c) .== c[:, inds..., :])

    trafo = LegendreTransform(modes = (3,))
    c = rand(7, M, 3)
    inds = [[1, 2, 3]]
    @test all(OperatorFlux.truncate_modes(trafo, c) .== c[:, inds..., :])

    trafo = LegendreTransform(modes = (12, 5))
    c = rand(27, 16, 14, 1)
    inds = [collect(1:m) for m in size(c)[2:(end - 1)]]
    @test all(OperatorFlux.pad_modes(trafo, c, (32, 14))[:, inds..., :] .== c)

    # rountrip integration test
    for mode in [(5,), (5, 6), (5, 7, 9)]
        z = rand(3, mode..., 5)
        trafo = LegendreTransform(modes = mode)
        a = OperatorFlux.forward(trafo, z)
        b = OperatorFlux.truncate_modes(trafo, a)
        c = OperatorFlux.pad_modes(trafo, b, size(a)[2:(end - 1)])
        d = OperatorFlux.inverse(trafo, c)
        @test norm(d - z) ≤ eps(10 * norm(z))
    end

    # test utils
    for n in 6:16
        x, _ = GaussQuadrature.legendre(n, GaussQuadrature.both)

        inv_transform = legendre_transform_inverse_matrix(n)
        transform = inv(inv_transform)
        @test all(transform .== legendre_transform_forward_matrix(n))

        for (i, p) in enumerate(polynomials)
            @test (transform * p.(x))[i] - 1 ≤ eps(10 * norm(p.(x)))
            @test norm(transform * p.(x)) - 1 ≤ eps(10 * norm(p.(x)))
        end
        for (i, p) in enumerate(polynomials)
            @test norm(inv_transform * transform * p.(x) - p.(x)) ≤
                  eps(10 * norm(p.(x)))
        end
    end

    # test Base extensions
    trafo = LegendreTransform(modes = (5, 6, 12))
    @test ndims(trafo) == 3
    @test eltype(trafo) == Float32
    @test size(trafo) == (5, 6, 12)
end
