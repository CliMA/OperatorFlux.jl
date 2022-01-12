@testset "FourierTransform" begin
    # auxiliary data
    M = 32

    # test constructors
    trafo = FourierTransform(modes = (3, 7))
    @test trafo.modes == (3, 7)

    # test forward & inverse
    trafo = FourierTransform(modes = (16, 16))
    z = rand(5, M, M, 4)
    @test size(OperatorFlux.forward(trafo, z)) == (5, M, M, 4)
    @test norm(OperatorFlux.inverse(trafo, forward(trafo, z)) - z) ≤ 2 * eps(norm(z))

    # test truncate_modes
    trafo = FourierTransform(modes = (3, 7))
    c = rand(7, M, M, 3)
    inds = [[(1:m+1)..., (s-m+2:s)...] for (s, m) in zip(size(c)[2:end-1], trafo.modes)]
    @test all(OperatorFlux.truncate_modes(trafo, c) .== c[:, inds..., :])

    # test pad_modes
    trafo = FourierTransform(modes = (12, 5))
    c = rand(27, 16, 14, 1)
    inds = [[(1:div(m, 2)+1)..., (s-div(m, 2)+2:s)...] for (s, m) in zip((M, M), size(c)[2:end-1])]
    @test all(OperatorFlux.pad_modes(trafo, c, (M, M))[:, inds..., :] .== c)

    # rountrip integration test
    a, b = -π, π
    x = (b - a) .* collect(0:(M-1)) / M .+ a
    y = copy(x)
    z = sin.(x) * sin.(y)'
    z = z[:, :, :, :] # emulate channels and batches
    z = permutedims(z, (3, 1, 2, 4))
    trafo = FourierTransform(modes = (12, 17))
    a = OperatorFlux.forward(trafo, z)
    b = OperatorFlux.truncate_modes(trafo, a)
    c = OperatorFlux.pad_modes(trafo, b, size(a)[2:end-1])
    d = OperatorFlux.inverse(trafo, c)
    @test norm(d - z) ≤ eps(norm(z))

    # test Base extensions
    trafo = FourierTransform(modes = (5, 11))
    @test ndims(trafo) == 2
    @test eltype(trafo) == ComplexF32
    @test size(trafo) == [10, 22]
end

@testset "FourierTransform - ChainRulesCore" begin
    # test ChainRulesCore extensions
    M = 4
    trafo = FourierTransform(modes = (2, 2))
    c = rand(3, M, M, 5)
    # check_thunked_output_tangent checks that thunked objects pass through
    # in order to make this work we need to pass through the dims 
    # and not use size(c), for example, in the code
    test_rrule(OperatorFlux.truncate_modes, trafo, c, check_thunked_output_tangent = false)
    test_rrule(OperatorFlux.pad_modes, trafo, c, (2 * M, 2 * M), check_thunked_output_tangent = false)
end