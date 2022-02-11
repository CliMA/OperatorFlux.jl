using OperatorFlux
using Test
using Flux
using LinearAlgebra

@testset "SpectralKernelOperator - FourierTransform" begin
    for M in [32, 33]
        even = Bool(mod(M, 2)) ? false : true
        # test spectral operator forward evaluation
        trafo = FourierTransform(modes = (8, 7, 5), even = even)
        model = Chain(
            Dense(3, 32), # lifting
            SpectralKernelOperator(trafo, 32 => 32),
            SpectralKernelOperator(trafo, 32 => 32),
            SpectralKernelOperator(trafo, 32 => 32),
            SpectralKernelOperator(trafo, 32 => 32),
            Dense(32, 4), # unlifting
        )
        x = rand(Float32, 3, M, 16, 27, 17)
        @test size(model(x)) == (4, M, 16, 27, 17)
    end

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
    for M in [32, 33]
        # test spectral operator forward evaluation
        trafo = ChebyshevTransform(modes = (8,))
        model = Chain(
            Dense(3, 32), # lifting
            SpectralKernelOperator(trafo, 32 => 32),
            SpectralKernelOperator(trafo, 32 => 32),
            SpectralKernelOperator(trafo, 32 => 32),
            SpectralKernelOperator(trafo, 32 => 32),
            Dense(32, 4), # unlifting
        )
        x = rand(Float32, 3, M, 17)
        @test size(model(x)) == (4, M, 17)
    end

    # integration test using Flux.jl
    trafo = ChebyshevTransform(modes = (8,))
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

@testset "SpectralKernelOperator - LegendreTransform" begin
    for M in [32, 33]
        # test spectral operator forward evaluation
        trafo = LegendreTransform(modes = (8, 8))
        model = Chain(
            Dense(3, 32), # lifting
            SpectralKernelOperator(trafo, 32 => 32),
            SpectralKernelOperator(trafo, 32 => 32),
            SpectralKernelOperator(trafo, 32 => 32),
            SpectralKernelOperator(trafo, 32 => 32),
            Dense(32, 4), # unlifting
        )
        x = rand(Float32, 3, M, 10, 17)
        @test size(model(x)) == (4, M, 10, 17)
    end

    # TODO!: Broken due to mutating array
    # # integration test using Flux.jl
    # trafo = LegendreTransform(modes = (8, 8))
    # model = Chain(
    #     Dense(1, 32),
    #     SpectralKernelOperator(trafo, 32 => 32),
    #     Dense(32, 128),
    # )
    # params_before = deepcopy(params(model))
    # loss(x, y) = Flux.mse(model(x), y)
    # x = rand(Float32, 1, 8, 10, 5)
    # data = [(x, rand(Float32, 128, 8, 10, 5))]
    # Flux.train!(loss, params(model), data, Flux.ADAM())
    # params_after = deepcopy(params(model))
    # # test that something actually happened during 
    # # optimization with Flux.
    # for i in 1:length(params_before)
    #     @test !all(params_after[i] .== params_before[i]) broken = true
    # end
end
