struct FourierTransform{N, T, D} <: AbstractTransform
    modes::NTuple{N, T}
    dims::D
end

"""
    Fourier(; modes)

Constructs a discrete Fourier transform with modes `modes` that 
operates on the 2nd, 3rd, ...N-modes+1-st dimension of the data.
Input must be have even-numbered size.

# Example
```
julia> Fourier(modes = (12, 3, 4, 12))
```
"""
function FourierTransform(; modes::NTuple{N, T}) where {N, T}
    # assumes transform is over first consecutive space-like
    # dimensions, aka 2:N+1. Input must be even-sized.
    dims = 2:(length(modes) + 1)
    return FourierTransform(modes, dims)
end

function forward(tr::FourierTransform{N}, x, dims = tr.dims) where {N}
    # assume real-valued input of even-numbered grid dimensions
    return rfft(x, dims)
end

function inverse(tr::FourierTransform{N}, x, dims = tr.dims) where {N}
    # need to specify the output dimension for irfft since the output
    # size depends on teh dimensionality of the input.
    d = 2 * size(x)[dims[1]] - 2
    return irfft(x, d, dims)
end

function truncate_modes(
    tr::FourierTransform{N},
    coeff,
    dims = tr.dims,
) where {N}
    # return a low-pass filtered version of coeff assuming
    # that coeff is a tensor of spectral weights.
    # Want to keep 1:M+1 to end-M+2:end using FFTW convention for all but
    # the first dimension
    #
    # Ex.: tr.modes = (3,)
    # [0, 1, 2, 3, -2, -1] -> [0, 1, 2, -1]
    # [a, b, c, d,  e,  f] -> [a, b, c,  f]

    # calculate the retained modes taking into account the dimensions
    # that the spectral transform operates over
    size_space = size(coeff)[2:(end - 1)] # sizes of space-like dimensions of coeff
    inds_space = 2:(length(size_space) + 1) # indices of space-like dimensions of coeff
    inds_map = Dict(zip(dims, tr.modes)) # maps index location to retained modes
    inds_offset = 1

    # we only truncate along dimensions contained in dims and otherwise keep
    # all modes
    modes = [
        i ∈ dims ? inds_map[i] : div(size_space[i - inds_offset], 2)
        for i in inds_space
    ]

    # indices for the spectral coefficients that we need to retain
    inds = [
        vcat(collect(1:(m)), collect((s - m + 1):s))
        for (s, m) in zip(size(coeff)[2:(end - 1)], modes)
    ]

    # we need to handle the first dimension of the real Fourier transform
    # separately
    inds[dims[1] - inds_offset] = collect(1:tr.modes[1])

    coeff_truncated = OperatorFlux.mview(coeff, inds, Val(length(size_space)))

    return coeff_truncated
end

function pad_modes(
    tr::FourierTransform,
    coeff,
    size_pad::NTuple,
    dims = tr.dims,
)
    # return a padded-with-zeros version of coeff assuming
    # that coeff is a tensor of spectral weights, thereby inflating coeff.
    N = length(size_pad) # number of space-like dimensions
    size_space = size(coeff)[2:(end - 1)] # sizes of space-like dimensions of coeff

    # generate a zero array and indices that point to the filled-in
    # locations
    coeff_padded =
        zeros(eltype(coeff), (size(coeff)[1], size_pad..., size(coeff)[end]))
    inds = [
        vcat(collect(1:(div(m, 2) + 1)), collect((s - div(m, 2) + 2):s))
        for (s, m) in zip(size_pad, size(coeff)[2:(end - 1)])
    ]

    # we need to handle the first dimension of the real Fourier transform
    # separately
    inds_offset = 1
    inds[dims[1] - inds_offset] = collect(1:size_space[dims[1] - inds_offset])

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
