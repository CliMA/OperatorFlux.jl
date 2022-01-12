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
    modes = (12, 17)
    trafo = FourierTransform(modes = modes)
    ch = 1 => 1
    sc = SpectralConv(trafo, ch)
    sc.weights .= ones(eltype(z), 1, (2 * collect(modes))..., 1)
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

@testset "SpectralConv - ChebyshevTransform" begin
    # auxiliary data
    M = 32

    # test constructor
    trafo = Chebyshev(modes = (10, 23))
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
    trafo = Chebyshev(modes = modes)
    ch = 1 => 1
    sc = SpectralConv(trafo, ch)
    sc.weights .= ones(eltype(z), 1, collect(modes)..., 1)
    @test norm(sc(z) - z) ≤ 2 * eps(norm(z))
    @test ndims(sc.weights) == 4

    # test more than one out_channel
    ch = 1 => 2
    trafo = Chebyshev(modes = (16, 16))
    sc = SpectralConv(trafo, ch)
    x = rand(Float32, 1, 32, 32, 2)
    @test size(sc(x)) == (2, 32, 32, 2)

    # test different dimensionalities
    # 1d
    ch = 1 => 2
    trafo = Chebyshev(modes = (16,))
    sc = SpectralConv(trafo, ch)
    x = rand(Float32, 1, 32, 2)
    @test size(sc(x)) == (2, 32, 2)

    # 3d
    ch = 13 => 3
    trafo = Chebyshev(modes = (16, 16, 14))
    sc = SpectralConv(trafo, ch)
    x = rand(Float32, 13, 32, 32, 32, 2)
    @test size(sc(x)) == (3, 32, 32, 32, 2)

    # 4d
    ch = 13 => 7
    trafo = Chebyshev(modes = (16, 16, 14, 3))
    sc = SpectralConv(trafo, ch)
    x = rand(Float32, 13, 32, 32, 32, 63, 2)
    @test size(sc(x)) == (7, 32, 32, 32, 63, 2)

    # test Base extensions
    trafo = Chebyshev(modes = (10,))
    ch = 3 => 13
    sc = SpectralConv(trafo, ch)
    @test ndims(sc) == 1
end

@testset "SpectralKernelOperator - FourierTransform" begin
    # test spectral operator forward evaluation
    trafo = FourierTransform(modes = (8, 7, 5))
    model = Chain(
        Dense(3, 32), # lifting
        SpectralKernelOperator(trafo, 32 => 32),
        SpectralKernelOperator(trafo, 32 => 32),
        SpectralKernelOperator(trafo, 32 => 32),
        SpectralKernelOperator(trafo, 32 => 32),
        Dense(32, 4), # unlifting
    )
    x = rand(Float32, 3, 32, 16, 27, 17)
    @test size(model(x)) == (4, 32, 16, 27, 17)

    # integration test using Flux.jl
    trafo = FourierTransform(modes = (8,))
    model = Chain(
        Dense(1, 32),
        SpectralKernelOperator(trafo, 32 => 32),
        Dense(32, 128),
    )
    params_before = deepcopy(params(model))
    loss(x, y) = Flux.mse(model(x), y)
    x = rand(Float32, 1, 1024, 5)
    data = [(x, rand(Float32, 128, 1024, 5))]
    Flux.train!(loss, params(model), data, Flux.ADAM())
    params_after = deepcopy(params(model))
    # test that something actually happened during 
    # optimization with Flux.
    for i in 1:length(params_before)
        @test !all(params_after[i] .== params_before[i])
    end
end


@testset "SpectralKernelOperator - ChebyshevTransform" begin
    # test spectral operator forward evaluation
    trafo = Chebyshev(modes = (8,))
    model = Chain(
        Dense(3, 32), # lifting
        SpectralKernelOperator(trafo, 32 => 32),
        SpectralKernelOperator(trafo, 32 => 32),
        SpectralKernelOperator(trafo, 32 => 32),
        SpectralKernelOperator(trafo, 32 => 32),
        Dense(32, 4), # unlifting
    )
    x = rand(Float32, 3, 32, 17)
    @test size(model(x)) == (4, 32, 17)

    # integration test using Flux.jl
    trafo = Chebyshev(modes = (8,))
    model = Chain(
        Dense(1, 32),
        SpectralKernelOperator(trafo, 32 => 32),
        Dense(32, 128),
    )
    params_before = deepcopy(params(model))
    loss(x, y) = Flux.mse(model(x), y)
    x = rand(Float32, 1, 1024, 5)
    data = [(x, rand(Float32, 128, 1024, 5))]
    Flux.train!(loss, params(model), data, Flux.ADAM())
    params_after = deepcopy(params(model))
    # test that something actually happened during 
    # optimization with Flux.
    for i in 1:length(params_before)
        @test !all(params_after[i] .== params_before[i])
    end
end