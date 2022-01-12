"""
    FourierTransform(; modes)

Constructs a discrete Fourier transform with modes `modes`.

# Example
```
julia> FourierTransform(modes = (12, 3, 4, 12))
```
"""
Base.@kwdef struct FourierTransform{N,T} <: AbstractTransform
    modes::NTuple{N,T}
end

function forward(tr::FourierTransform{N}, x) where {N}
    # x -> [in_channels, dims(x), batch_size]
    return fft(x, 2:N+1)
end

function inverse(tr::FourierTransform{N}, x) where {N}
    # x -> [in_channels, dims(x), batch_size]
    return real(ifft(x, 2:N+1))
end

function truncate_modes(tr::FourierTransform{N}, c) where {N}
    # c -> [in_channels, size_c..., batch_size]
    # Returns a low-pass filtered version of c assuming
    # that c is a tensor of spectral weights.
    # Want to keep 1:M+1 to end-M+2:end using FFTW convention
    #
    # Ex.: tr.modes = (2,)
    # [0, 1, 2, 3, -2, -1] -> [0, 1, 2, -1]
    # [a, b, c, d,  e,  f] -> [a, b, c,  f]
    inds = [vcat(collect(1:m+1), collect(s-m+2:s)) for (s, m) in zip(size(c)[2:end-1], tr.modes)]
    c_truncated = OperatorFlux.mview(c, inds, Val(N))

    return c_truncated
end

function pad_modes(::FourierTransform{N}, c, size_pad::NTuple) where {N}
    # c -> [in_channels, size_c..., batch_size]
    # c_padded -> [in_channels, size_pad..., batch_size]
    # Returns a padded-with-zeros version of c assuming
    # that c is a tensor of spectral weights, thereby inflating c.
    # Want to keep 1:M+1 to end-M+2:end using FFTW convention, so need to 
    # fill rest with zeros.
    #
    # Ex.: dims = (6,)
    # [0, 1, 2, -1] -> [0, 1, 2, 3, -2, -1]
    # [a, b, c,  d] -> [a, b, c, 0,  0,  d]
    c_padded = zeros(eltype(c), (size(c)[1], size_pad..., size(c)[end]))
    inds = [vcat(collect(1:div(m, 2)+1), collect(s-div(m, 2)+2:s)) for (s, m) in zip(size_pad, size(c)[2:end-1])]
    c_padded_view = OperatorFlux.mview(c_padded, inds, Val(N))
    c_padded_view .= c

    return c_padded
end

# Base extensions
Base.ndims(::FourierTransform{N}) where {N} = N
Base.eltype(::FourierTransform) = ComplexF32
Base.size(tr::FourierTransform) = 2 * [tr.modes...]

function Base.show(io::IO, ft::FourierTransform)
    print(
        io,
        "FourierTransform(modes = $(ft.modes)"
    )
end
