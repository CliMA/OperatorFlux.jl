"""
    ChebyshevTransform(; modes)

Constructs a discrete Chebyshev transform transform with modes `modes` that 
operates on the 2nd, 3rd, ...N-modes+1-st dimension of the data.
Input must be have even-numbered size.
# Example
```
julia> ChebyshevTransform(modes = (12, 3, 4, 12))
```
"""
Base.@kwdef struct ChebyshevTransform{N, T} <: AbstractTransform
    modes::NTuple{N, T}
end

function forward(::ChebyshevTransform{N}, x) where {N}
    return FFTW.r2r(x, FFTW.REDFT00, 2:(N + 1))
end

function inverse(::ChebyshevTransform{N}, x) where {N}
    return FFTW.r2r(
        x ./ (prod(2 .* (size(x)[2:(N + 1)] .- 1))),
        FFTW.REDFT00,
        2:(N + 1),
    )
end

function truncate_modes(tr::ChebyshevTransform{N}, coeff) where {N}
    # Returns a low-pass filtered version of coeff assuming
    # that coeff is a tensor of spectral weights.
    # Ex.: tr.modes = (3,)
    # [0, 1, 2, 3, 4, 5] -> [0, 1, 2]
    # [a, b, c, d, e, f] -> [a, b, c]

    # indices for the spectral coefficients that we need to retain
    inds = [collect(1:m) for (s, m) in zip(size(coeff)[2:(end - 1)], tr.modes)]

    coeff_truncated = OperatorFlux.mview(coeff, inds, Val(N))

    return coeff_truncated
end

function pad_modes(
    ::ChebyshevTransform,
    coeff,
    size_pad::NTuple{N, T},
) where {N, T}
    # return a padded-with-zeros version of coeff assuming
    # that coeff is a tensor of spectral weights, thereby inflating coeff.
    # generate a zero array and indices that point to the filled-in
    # locations
    size_padded = (size(coeff)[1], size_pad..., size(coeff)[end])
    coeff_padded = zeros(eltype(coeff), size_padded)
    inds = [collect(1:m) for (s, m) in zip(size_pad, size(coeff)[2:(end - 1)])]

    coeff_padded_view = OperatorFlux.mview(coeff_padded, inds, Val(N))
    coeff_padded_view .= coeff

    return coeff_padded
end

# Base extensions
Base.ndims(::ChebyshevTransform{N}) where {N} = N
Base.eltype(::ChebyshevTransform) = Float32
Base.size(tr::ChebyshevTransform) = [tr.modes...]

function Base.show(io::IO, tr::ChebyshevTransform)
    print(io, "ChebyshevTransform(modes = $(tr.modes), dims = $(tr.dims))")
end
