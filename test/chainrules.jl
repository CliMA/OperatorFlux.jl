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
    # check_thunked_output_tangent checks that thunked objects pass through
    # in order to make this work we need to pass through the dims 
    # and not use size(c), for example, in the code
    test_rrule(
        FFTW.r2r,
        c,
        FFTW.REDFT00,
        2:2,
        check_thunked_output_tangent = false,
    )

    # truncate modes
    # NOTE: check_thunked_output_tangent checks that thunked objects pass through
    # in order to make this work we need to pass through the dims 
    # and not use size(c), for example, in the code
    M = 4
    trafo = FourierTransform(modes = (2, 2))
    c = rand(3, M, M, M, 5)
    dims = 2:3
    test_rrule(
        OperatorFlux.truncate_modes,
        trafo,
        c,
        dims,
        check_thunked_output_tangent = false,
    )
    dims = (3, 4)
    test_rrule(
        OperatorFlux.truncate_modes,
        trafo,
        c,
        dims,
        check_thunked_output_tangent = false,
    )

    trafo = ChebyshevTransform(modes = (2, 2))
    dims = 2:3
    test_rrule(
        OperatorFlux.truncate_modes,
        trafo,
        c,
        dims,
        check_thunked_output_tangent = false,
    )
    dims = (3, 4)
    test_rrule(
        OperatorFlux.truncate_modes,
        trafo,
        c,
        dims,
        check_thunked_output_tangent = false,
    )

    # pad modes
    trafo = FourierTransform(modes = (2, 4))
    c = rand(3, 2, 8, 10, 5)
    dims = (2, 3)
    test_rrule(
        OperatorFlux.pad_modes,
        trafo,
        c,
        (4, 8, 10),
        dims,
        check_thunked_output_tangent = false,
    )
    c = rand(3, 10, 2, 8, 5)
    dims = (3, 4)
    test_rrule(
        OperatorFlux.pad_modes,
        trafo,
        c,
        (10, 4, 8),
        dims,
        check_thunked_output_tangent = false,
    )

    trafo = ChebyshevTransform(modes = (2, 4))
    c = rand(3, 2, 4, 10, 5)
    dims = (2, 3)
    test_rrule(
        OperatorFlux.pad_modes,
        trafo,
        c,
        (2, 4, 10),
        dims,
        check_thunked_output_tangent = false,
    )
    c = rand(3, 10, 2, 4, 5)
    dims = (3, 4)
    test_rrule(
        OperatorFlux.pad_modes,
        trafo,
        c,
        (10, 2, 4),
        dims,
        check_thunked_output_tangent = false,
    )
end
