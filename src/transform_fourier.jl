"""
    FourierTransform(; modes, even = true)

Constructs a discrete Fourier transform with modes `modes` and style `even` that 
operates on the 2nd, 3rd, ...N-modes+1-st dimension of the data.

# Example
```
julia> FourierTransform(modes = (12, 3, 4, 12))
```
"""
Base.@kwdef struct FourierTransform{N, T} <: AbstractTransform
    modes::NTuple{N, T}
    even::Bool = true
end

function forward(::FourierTransform{N}, x) where {N}
    return rfft(x, 2:(N + 1))
end

function inverse(tr::FourierTransform{N}, x) where {N}
    if tr.even
        d = 2 * size(x)[2] - 2
    else
        d = 2 * size(x)[2] - 1
    end
    return irfft(x, d, 2:(N + 1))
end

function truncate_modes(tr::FourierTransform{N}, coeff) where {N}
    # return a low-pass filtered version of coeff assuming
    # that coeff is a tensor of spectral weights.
    # Ex.: tr.modes = (3,)
    # [0, 1, 2, 3, -2, -1] -> [0, 1, 2, -1]
    # [a, b, c, d,  e,  f] -> [a, b, c,  f]

    # indices for the spectral coefficients that we need to retain
    inds = [
        vcat(collect(1:(m)), collect((s - m + 1):s))
        for (s, m) in zip(size(coeff)[2:(end - 1)], tr.modes)
    ]

    # we need to handle the first dimension of the real Fourier transform
    # separately
    inds[1] = collect(1:tr.modes[1])

    coeff_truncated = OperatorFlux.mview(coeff, inds, Val(N))

    return coeff_truncated
end

function pad_modes(
    ::FourierTransform,
    coeff,
    size_pad::NTuple{N, T},
) where {N, T}
    # return a padded-with-zeros version of coeff assuming
    # that coeff is a tensor of spectral weights, thereby inflating coeff.
    size_space = size(coeff)[2:(end - 1)] # sizes of space-like dimensions of coeff

    # generate a zero array and indices that point to the filled-in
    # locations
    size_padded = (size(coeff)[1], size_pad..., size(coeff)[end])
    coeff_padded = zeros(eltype(coeff), size_padded)
    inds = [
        vcat(collect(1:(div(m, 2) + 1)), collect((s - div(m, 2) + 2):s))
        for (s, m) in zip(size_pad, size(coeff)[2:(end - 1)])
    ]

    # we need to handle the first dimension of the real Fourier transform
    # separately
    inds[1] = collect(1:size_space[1])

    coeff_padded_view = OperatorFlux.mview(coeff_padded, inds, Val(N))
    coeff_padded_view .= coeff

    return coeff_padded
end

# Base extensions
Base.ndims(::FourierTransform{N}) where {N} = N
Base.eltype(::FourierTransform) = ComplexF32
Base.size(tr::FourierTransform) = [tr.modes[1]..., 2 * [tr.modes[2:end]...]...]

function Base.show(io::IO, tr::FourierTransform)
    print(io, "FourierTransform(modes = $(tr.modes), dims = $(tr.dims))")
end
