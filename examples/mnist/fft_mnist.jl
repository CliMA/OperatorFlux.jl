push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

# Classification of MNIST dataset 
## with the convolutional neural network known as LeNet5.
## This script also combines various
## packages from the Julia ecosystem with Flux.
using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using CUDA
using MLDatasets
using OperatorFlux

function getdata(args, device)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset	
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    # Reshape Data in order to flatten each image into a linear array
    xtrain = reshape(xtrain, 1, 28, 28, :)
    xtest = reshape(xtest, 1, 28, 28, :)

    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Create DataLoaders (mini-batch iterators)
    train_loader =
        DataLoader((xtrain, ytrain), batchsize = args.batchsize, shuffle = true)
    test_loader = DataLoader((xtest, ytest), batchsize = args.batchsize)

    return train_loader, test_loader
end

function build_model(; nclasses = 10)
    trafo = FourierTransform(modes = (8, 8))
    return Chain(
        SpectralKernelOperator(trafo, 1 => 1, gelu),
        SpectralKernelOperator(trafo, 1 => 1, gelu),
        flatten,
        Dense(784, nclasses),
    )
end

function loss_and_accuracy(data_loader, model, device)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x, y = device(x), device(y)
        ŷ = model(x)
        ls += logitcrossentropy(ŷ, y, agg = sum)
        acc += sum(onecold(ŷ) .== onecold(y))
        num += size(x)[end]
    end
    return ls / num, acc / num
end

@kwdef mutable struct Args
    η::Float64 = 3e-4       # learning rate
    batchsize::Int = 256    # batch size
    epochs::Int = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
end

function train(; kws...)
    args = Args(; kws...) # collect options in a struct for convenience

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    # Create test and train dataloaders
    train_loader, test_loader = getdata(args, device)

    # Construct model
    model = build_model() |> device
    ps = Flux.params(model) # model's trainable parameters

    ## Optimizer
    opt = ADAM(args.η)

    ## Training
    for epoch in 1:(args.epochs)
        for (x, y) in train_loader
            x, y = device(x), device(y) # transfer data to device
            gs = gradient(() -> logitcrossentropy(model(x), y), ps) # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end

        # Report on train and test
        train_loss, train_acc = loss_and_accuracy(train_loader, model, device)
        test_loss, test_acc = loss_and_accuracy(test_loader, model, device)
        println("Epoch=$epoch")
        println("  train_loss = $train_loss, train_accuracy = $train_acc")
        println("  test_loss = $test_loss, test_accuracy = $test_acc")
    end
end

### Run training 
if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
# train(η=0.01) # can change hyperparameters
