@testset "Chebyshev" begin
    # auxiliary data
    M = 32

    # test constructors
    trafo = Chebyshev(modes = (3, 7))
    @test trafo.modes == (3, 7)
    @test trafo.dims == 2:3

    # test forward & inverse
    trafo = Chebyshev(modes = (16, 16))
    z = rand(5, M, M, 4)
    @test size(OperatorFlux.forward(trafo, z)) == (5, M, M, 4)
    @test norm(OperatorFlux.inverse(trafo, forward(trafo, z)) - z) ≤
          3 * eps(norm(z))

    # test truncate_modes
    trafo = Chebyshev(modes = (3, 7))
    c = rand(7, M, M, 3)
    inds = [collect(1:m) for m in trafo.modes]
    @test all(OperatorFlux.truncate_modes(trafo, c) .== c[:, inds..., :])

    # test pad_modes
    trafo = Chebyshev(modes = (12, 5))
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
    trafo = Chebyshev(modes = (12, 17))
    a = OperatorFlux.forward(trafo, z)
    b = OperatorFlux.truncate_modes(trafo, a)
    c = OperatorFlux.pad_modes(trafo, b, size(a)[2:(end - 1)])
    d = OperatorFlux.inverse(trafo, c)
    @test norm(d - z) ≤ eps(10 * norm(z))

    # test Base extensions
    trafo = Chebyshev(modes = (5, 11))
    @test ndims(trafo) == 2
    @test eltype(trafo) == Float32
    @test size(trafo) == [5, 11]
end

@testset "Chebyshev - ChainRulesCore" begin
    # test ChainRulesCore extensions
    M = 4
    trafo = Chebyshev(modes = (4,))
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
    test_rrule(
        OperatorFlux.truncate_modes,
        trafo,
        c,
        check_thunked_output_tangent = false,
    )
    test_rrule(
        OperatorFlux.pad_modes,
        trafo,
        c,
        (M,),
        check_thunked_output_tangent = false,
    )
end
