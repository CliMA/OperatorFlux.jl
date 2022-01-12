import FFTW: r2r

"""
    Chebyshev(; modes)

Constructs a discrete Chebyshev transform with modes `modes` on the dimensions of the data
`dims` by using .

# Example
```
julia> Chebyshev(modes = (12, 3, 4, 12))
```
"""
# FFTW.REDFT00 = 2N-1 - Chebyshev
# \int_{-1}^1 T_m(x) / sqrt(1-x^2) f(x) dx for each m
# change of variable x = cos\theta
# \int_0^\pi cos(m \theta) f(cos \theta) d\theta
struct Chebyshev{N,T,D} <: AbstractTransform
    modes::NTuple{N,T}
    dims::D
end

function Chebyshev(; modes::NTuple{N,T}) where {N,T}
    return Chebyshev(modes, 2:N+1)
end

function forward(tr::Chebyshev{N}, x) where {N}
    # x -> [in_channels, dims(x), batch_size]
    return FFTW.r2r(x, FFTW.REDFT00, tr.dims)
end

function inverse(tr::Chebyshev{N}, x) where {N}
    # x -> [in_channels, dims(x), batch_size]
    return FFTW.r2r(x ./ (prod(2 .* (size(x)[tr.dims] .- 1))), FFTW.REDFT00, tr.dims)
end

function truncate_modes(tr::Chebyshev{N}, c) where {N}
    # c -> [in_channels, size_c..., batch_size]
    # Returns a low-pass filtered version of c assuming
    # that c is a tensor of spectral weights.
    # Want to keep 1:M+1 to end-M+2:end using FFTW convention
    #
    # Ex.: tr.modes = (3,)
    # TODO! Be consistent with truncation conventions.
    # [0, 1, 2, 3, 4, 5] -> [0, 1, 2]
    # [a, b, c, d, e, f] -> [a, b, c]
    inds = [collect(1:m) for m in tr.modes]
    c_truncated = OperatorFlux.mview(c, inds, Val(N))

    return c_truncated
end

function pad_modes(::Chebyshev{N}, c, size_pad::NTuple) where {N}
    # c -> [in_channels, size_c..., batch_size]
    # c_padded -> [in_channels, size_pad..., batch_size]
    # Returns a padded-with-zeros version of c assuming
    # that c is a tensor of spectral weights, thereby inflating c.
    # Want to keep 1:M+1 to end-M+2:end using FFTW convention, so need to 
    # fill rest with zeros.
    #
    # Ex.: dims = (6,)
    # [0, 1, 2, 3] -> [0, 1, 2, 3, 4, 5]
    # [a, b, c, d] -> [a, b, c, d, 0, 0]
    c_padded = zeros(eltype(c), (size(c)[1], size_pad..., size(c)[end]))
    inds = [collect(1:m) for m in size(c)[2:end-1]]
    c_padded_view = OperatorFlux.mview(c_padded, inds, Val(N))
    c_padded_view .= c

    return c_padded
end

# Base extensions
Base.ndims(::Chebyshev{N}) where {N} = N
Base.eltype(::Chebyshev) = Float32
Base.size(tr::Chebyshev) = [tr.modes...]

function Base.show(io::IO, ft::Chebyshev)
    print(
        io,
        "Chebyshev(modes = $(ft.modes)"
    )
end

function ChainRulesCore.rrule(::typeof(r2r), x::AbstractArray, kind, dims)

    (M,) = size(x)[2:end-1]
    a1 = ones(M)
    a2 = [(-1)^i for i = 1:M]
    a2[1] = a2[end] = 0.0
    a1[1] = a1[end] = 0.0
    e1 = zeros(M)
    e1[1] = 1.0
    eN = zeros(M)
    eN[end] = 1.0

    function r2r_pullback(y)
        # r2r pullback turns out to be r2r + a rank 4 correction
        w = r2r(y, kind, dims)
        @tullio w[s, i, b] += a1[i] * e1[k] * y[s, k, b] - a2[i] * eN[k] * y[s, k, b]
        @tullio w[s, i, b] += eN[i] * a2[k] * y[s, k, b] - e1[i] * a1[k] * y[s, k, b]
        return NoTangent(), w, NoTangent(), NoTangent()
    end

    return r2r(x, kind, dims), r2r_pullback
end
