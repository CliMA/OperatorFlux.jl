struct ChebyshevTransform{N, T, D} <: AbstractTransform
    modes::NTuple{N, T}
    dims::D
end

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
# FFTW.REDFT00 = 2N-1 - Chebyshev
# \int_{-1}^1 T_m(x) / sqrt(1-x^2) f(x) dx for each m
# change of variable x = cos\theta
# \int_0^\pi cos(m \theta) f(cos \theta) d\theta
function ChebyshevTransform(; modes::NTuple{N, T}) where {N, T}
    # assumes transform is over first consecutive space-like
    # dimensions, aka 2:N+1. Input must be even-sized.
    dims = 2:(length(modes) + 1)
    return ChebyshevTransform(modes, dims)
end

function forward(tr::ChebyshevTransform{N}, x, dims = tr.dims) where {N}
    return FFTW.r2r(x, FFTW.REDFT00, dims)
end

function inverse(tr::ChebyshevTransform{N}, x, dims = tr.dims) where {N}
    return FFTW.r2r(
        x ./ (prod(2 .* (size(x)[collect(dims)] .- 1))),
        FFTW.REDFT00,
        dims,
    )
end

function truncate_modes(
    tr::ChebyshevTransform{N},
    coeff,
    dims = tr.dims,
) where {N}
    # Returns a low-pass filtered version of coeff assuming
    # that coeff is a tensor of spectral weights.
    # Want to keep 1:M+1 to end-M+2:end using FFTW convention
    #
    # Ex.: tr.modes = (3,)
    # TODO! Be consistent with truncation conventions.
    # [0, 1, 2, 3, 4, 5] -> [0, 1, 2]
    # [a, b, c, d, e, f] -> [a, b, c]
    # calculate the retained modes taking into account the dimensions
    # that the spectral transform operates over
    size_space = size(coeff)[2:(end - 1)] # sizes of space-like dimensions of coeff
    inds_space = 2:(length(size_space) + 1) # indices of space-like dimensions of coeff
    inds_map = Dict(zip(dims, tr.modes)) # maps index location to retained modes
    inds_offset = 1

    # we only truncate along dimensions contained in dims and otherwise keep
    # all modes
    modes = [
        i ∈ dims ? inds_map[i] : size_space[i - inds_offset]
        for i in inds_space
    ]

    # indices for the spectral coefficients that we need to retain
    inds = [collect(1:m) for (s, m) in zip(size(coeff)[2:(end - 1)], modes)]

    coeff_truncated = OperatorFlux.mview(coeff, inds, Val(length(size_space)))

    return coeff_truncated
end

function pad_modes(
    tr::ChebyshevTransform,
    coeff,
    size_pad::NTuple,
    dims = tr.dims,
)
    # return a padded-with-zeros version of coeff assuming
    # that coeff is a tensor of spectral weights, thereby inflating coeff.
    N = length(size_pad) # number of space-like dimensions

    # generate a zero array and indices that point to the filled-in
    # locations
    coeff_padded =
        zeros(eltype(coeff), (size(coeff)[1], size_pad..., size(coeff)[end]))
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
