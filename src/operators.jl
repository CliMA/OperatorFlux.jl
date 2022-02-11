struct SpectralConv{T, W, I, O} <: AbstractSpectralOperator
    transform::T
    weights::W
    in_channels::I
    out_channels::O
end

"""
    SpectralConv(transform, channels; init = glorot_uniform)

Constructs a spectral convolution operator with spectral transform `trafo`, channels `channels`, 
and initialization function `init`.

# Example
```
julia> trafo = FourierTransform(modes = (12, 3, 4, 12))
julia> channels = 1 => 4
julia> conv = SpectralConv(trafo, channels)
```
"""
function SpectralConv(
    transform::T,
    ch::Pair{S, S};
    init = Flux.glorot_uniform,
) where {T, S <: Int}
    # initialize optimizable weights
    FT = eltype(transform)
    in, out = ch
    scale = one(FT) / (in * out)

    # depending on the transform, the size of the coefficient
    # matrix may be different
    space_dims = size(transform)

    # depending on the transform, we may need to initialize
    # with real or complex numbers
    if FT <: Complex
        weights =
            init(in, space_dims..., out) + init(in, space_dims..., out) * im
    else
        weights = init(in, space_dims..., out)
    end
    weights = scale * weights

    return SpectralConv(transform, weights, in, out)
end

Flux.@functor SpectralConv

function (conv::SpectralConv)(x::AbstractArray)
    # TODO: (1) perhaps dispatch directly instead of calling ndims(conv)
    #       (2) perhaps something to infer size(x) at compile time
    c = OperatorFlux.forward(conv.transform, x)
    ct = OperatorFlux.truncate_modes(conv.transform, c)
    wc = OperatorFlux.tensor_contraction(conv.weights, ct, Val(ndims(conv)))
    wcp = OperatorFlux.pad_modes(conv.transform, wc, size(c)[2:(end - 1)])
    y = OperatorFlux.inverse(conv.transform, wcp)

    return y
end

struct SpectralCovariance{T, U, V, W, I, O} <: AbstractSpectralOperator
    transform::T
    weights_μ::U
    weights_d::V
    weights_w::W
    in_channels::I
    out_channels::O
end

"""
    SpectralCovariance(transform, rank, channels; init = glorot_uniform)

Constructs a spectral covariance operator with spectral transform `trafo`, low-rank matrix rank `rank`, channels `channels`, 
and initialization function `init`.

# Example
```
julia> trafo = FourierTransform(modes = (12, 3, 4, 12))
julia> channels = 1 => 4
julia> rank = 3
julia> conv = SpectralCovariance(trafo, rank, channels)
```
"""
function SpectralCovariance(
    transform::T,
    rank::R,
    ch::Pair{S, S};
    init = Flux.glorot_uniform,
) where {T, R, S <: Int}
    FT = eltype(transform)
    in, out = ch
    scale = one(FT) / (in * out)

    # depending on the transform, we may need to initialize
    # with real or complex numbers
    if FT <: Complex
        weights_μ = init(in, out) + init(in, out) * im
        weights_d = init(in, out) + init(in, out) * im
        weights_w = init(in, rank, out) + init(in, rank, out) * im
    else
        weights_μ = init(in, out)
        weights_d = init(in, out)
        weights_w = init(in, rank, out)
    end
    weights_μ = scale * weights_μ
    weights_d = scale * weights_d
    weights_w = scale * weights_w

    return SpectralCovariance(
        transform,
        weights_μ,
        weights_d,
        weights_w,
        in,
        out,
    )
end

Flux.@functor SpectralCovariance

function (so::SpectralCovariance)(x::AbstractArray)
    N_space = length(size(x)[2:(end - 1)])

    # Bring input into truncated spectral representation
    c = OperatorFlux.forward(so.transform, x)
    ct = OperatorFlux.truncate_modes(so.transform, c)

    # paramterize the mean for the low-rank approximation
    μ = OperatorFlux.sparse_mean(so.weights_μ, ct, Val(N_space))
    μ = OperatorFlux.pad_modes(so.transform, μ, size(c)[2:(end - 1)])
    μ = OperatorFlux.inverse(so.transform, μ)

    # parameterize the diagonal matrix for the low-rank approximation
    D = OperatorFlux.sparse_mean(so.weights_d, ct, Val(N_space))
    D = OperatorFlux.pad_modes(so.transform, D, size(c)[2:(end - 1)])
    D = OperatorFlux.inverse(so.transform, D)
    D = @. log(1 + exp(D)) # keeping a stable positive attitude

    # parameterize the low-rank matrix for the low-rank approximation
    rank = size(so.weights_w)[end - 1]
    V = OperatorFlux.sparse_covariance(so.weights_w, ct, Val(N_space))
    V = OperatorFlux.pad_modes(so.transform, V, (size(c)[2:(end - 1)]..., rank))
    # we need to bring the array into a shape that is compatible with the 
    # transform API
    store_v_size = size(V)
    V = reshape(
        V,
        (store_v_size[1:(end - 2)]..., prod(store_v_size[(end - 1):end]))...,
    )
    V = OperatorFlux.inverse(so.transform, V)
    V = reshape(V, (size(V)[1:(end - 1)]..., store_v_size[(end - 1):end]...)...)

    return μ, D, V
end

struct SpectralKernelOperator{L, C, A} <: AbstractSpectralOperator
    linear::L
    conv::C
    σ::A
end

"""
    SpectralKernelOperator(transform, channels, σ = identity; init = glorot_uniform)

Constructs a spectral kernel operator with spectral transform `trafo`, channels `channels`, 
activation function `σ`, and initialization function `init`.

References:
Fourier Neural Operator for Parametric Partial Differential Equations - https://arxiv.org/abs/2010.08895

# Example
```
julia> trafo = FourierTransform(modes = (12, 3, 4, 12))
julia> channels = 1 => 4
julia> σ = relu
julia> conv = SpectralKernelOperator(trafo, channels, σ)
```
"""
function SpectralKernelOperator(
    transform::AbstractTransform,
    channels::Pair{S, S},
    σ = identity;
    init = Flux.glorot_uniform,
) where {S <: Int, N}
    linear = Dense(channels.first, channels.second)
    conv = SpectralConv(transform, channels, init = init)

    return SpectralKernelOperator(linear, conv, σ)
end

Flux.@functor SpectralKernelOperator

function (so::SpectralKernelOperator)(x)
    # The definition of the Spectral Kernel Operators include a linear transform as well.
    return so.σ.(so.linear(x) + so.conv(x))
end

# Base extensions
Base.ndims(so::AbstractSpectralOperator) = ndims(so.transform)

function Base.show(io::IO, so::SpectralConv)
    print(
        io,
        "SpectralConv($(so.in_channels) => $(so.out_channels), $(so.transform))",
    )
end

function Base.show(io::IO, so::SpectralCovariance)
    print(
        io,
        "SpectralConv($(so.in_channels) => $(so.out_channels), $(so.transform)))",
    )
end

function Base.show(io::IO, so::SpectralKernelOperator)
    print(
        io,
        "SpectralKernelOperator(" *
        "$(sco.conv.in_channel) => $(sco.conv.out_channel), " *
        "$(sco.conv.transform), " *
        "σ=$(string(sco.σ)) " *
        ")",
    )
end
