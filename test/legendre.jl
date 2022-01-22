using OperatorFlux
using Test
using FFTW
using Flux
using GaussQuadrature
using LinearAlgebra
using Revise
using ChainRulesCore
using ChainRulesTestUtils

import OperatorFlux: legendre_transform_forward_matrix, legendre_transform_inverse_matrix

@testset "LegendreTransform" begin
    # auxiliary data
    P₀(x) = 1
    P₁(x) = x
    P₂(x) = 0.5 * (3 * x^2 - 1)
    P₃(x) = 0.5 * (5 * x^3 - 3 * x)
    P₄(x) = 0.125 * (35 * x^4 - 30 * x^2 + 3)
    P₅(x) = 0.125 * (63 * x^5 - 70 * x^3 + 15 * x)
    polynomials = [P₀, P₁, P₂, P₃, P₄, P₅]

    # test constructors

    # test forward & inverse

    # test truncate_modes

    # test pad_modes

    # rountrip integration test

    # test utils
    for n in 6:16
        x, _ = GaussQuadrature.legendre(n, GaussQuadrature.both)

        inv_transform = legendre_transform_inverse_matrix(n)
        transform = inv(inv_transform)
        @test all(transform .== legendre_transform_forward_matrix(n))
    
        for (i, p) in enumerate(polynomials)
            @test (transform*p.(x))[i] - 1 ≤ eps(10*norm(p.(x)))
            @test norm(transform * p.(x)) - 1 ≤ eps(10*norm(p.(x)))
        end
        for (i, p) in enumerate(polynomials)
            @test norm(inv_transform * transform * p.(x) - p.(x)) ≤ eps(10 * norm(p.(x)))
        end
    end

    # test Base extensions
    # trafo = LegendreTransform()
    # @test ndims(trafo) == 
    # @test eltype(trafo) == Float32
    # @test size(trafo) == 
end

#=
LTM = legendretransformmatrix 

@tullio q[s, i, j, k, e, b] := LTM[i,ii] * LTM[j, jj] * LTM[k, kk] * q[s, ii, jj, kk, e, b]
=#