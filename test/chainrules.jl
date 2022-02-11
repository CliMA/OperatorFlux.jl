using OperatorFlux
using Test
using FFTW
using Flux
using LinearAlgebra
using ChainRulesCore
using ChainRulesTestUtils

@testset "ChainRulesCore Extensions" begin
    # FFTW.r2r
    M = 4
    trafo = ChebyshevTransform(modes = (4,))
    c = rand(3, M, 5)
    test_rrule(
        FFTW.r2r,
        c,
        FFTW.REDFT00,
        2:2,
        check_thunked_output_tangent = false,
    )

    # truncate modes
    M = 4
    trafo = FourierTransform(modes = (2, 2))
    c = rand(3, M, M, 5)
    test_rrule(
        OperatorFlux.truncate_modes,
        trafo,
        c,
        check_thunked_output_tangent = false,
    )

    trafo = ChebyshevTransform(modes = (2, 2))
    test_rrule(
        OperatorFlux.truncate_modes,
        trafo,
        c,
        check_thunked_output_tangent = false,
    )

    for mode in [(5,), (5, 6), (5, 7, 9)]
        z = rand(3, mode..., 5)
        trafos = (LegendreTransform(modes = mode),)
        for trafo in trafos
            test_rrule(
                OperatorFlux.truncate_modes,
                trafo,
                z,
                check_thunked_output_tangent = false,
            )
        end
    end

    # pad modes
    c = rand(3, 2, 8, 5)
    trafo = FourierTransform(modes = (2, 4))
    test_rrule(
        OperatorFlux.pad_modes,
        trafo,
        c,
        (4, 8),
        check_thunked_output_tangent = false,
    )

    trafo = ChebyshevTransform(modes = (2, 4))
    c = rand(3, 2, 4, 5)
    test_rrule(
        OperatorFlux.pad_modes,
        trafo,
        c,
        (2, 4),
        check_thunked_output_tangent = false,
    )

    for mode in [(5,), (5, 6), (5, 7, 9)]
        z = rand(3, mode..., 5)
        trafos = (LegendreTransform(modes = mode),)
        for trafo in trafos
            test_rrule(
                OperatorFlux.pad_modes,
                trafo,
                z,
                mode,
                check_thunked_output_tangent = false,
            )
        end
    end
end
