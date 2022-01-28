struct SpectralConv{T, W, I, O} <: AbstractOperator
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
    dims = size(transform)

    # depending on the transform, we may need to initialize
    # with real or complex numbers
    if FT <: Complex
        weights = init(in, dims..., out) + init(in, dims..., out) * im
    else
        weights = init(in, dims..., out)
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

struct SpectralKernelOperator{L, C, A} <: AbstractOperator
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
Base.ndims(sc::SpectralConv) = ndims(sc.transform)

function Base.show(io::IO, sc::SpectralConv)
    print(
        io,
        "SpectralConv($(sc.in_channels) => $(sc.out_channels), $(sc.transform))",
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

# Utils
tensor_contraction(
    A,
    B,
    ::Val{1},
) = @tullio C[o, a, b] := A[i, a, o] * B[i, a, b]
tensor_contraction(A, B, ::Val{2}) =
    @tullio C[o, a₁, a₂, b] := A[i, a₁, a₂, o] * B[i, a₁, a₂, b]
tensor_contraction(A, B, ::Val{3}) =
    @tullio C[o, a₁, a₂, a₃, b] := A[i, a₁, a₂, a₃, o] * B[i, a₁, a₂, a₃, b]
tensor_contraction(A, B, ::Val{4}) = @tullio C[o, a₁, a₂, a₃, a₄, b] :=
    A[i, a₁, a₂, a₃, a₄, o] * B[i, a₁, a₂, a₃, a₄, b]
