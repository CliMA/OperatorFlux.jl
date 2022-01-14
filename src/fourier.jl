struct FourierTransform{N,T,D} <: AbstractTransform
    modes::NTuple{N,T}
    dims::D
end

function FourierTransform(; modes::NTuple{N,T}) where {N,T}
    # assumes transform is over first consecutive space-like
    # dimensions, aka 2:N+1
    dims = 2:(length(modes)+1)
    return FourierTransform(modes, dims)
end

function forward(tr::FourierTransform{N}, x, dims = tr.dims) where {N}
    return fft(x, dims)
end

function inverse(tr::FourierTransform{N}, x, dims = tr.dims) where {N}
    return ifft(x, dims)
end

function truncate_modes(tr::FourierTransform{N}, c, dims = tr.dims) where {N}
    # return a low-pass filtered version of c assuming
    # that c is a tensor of spectral weights.
    # Want to keep 1:M+1 to end-M+2:end using FFTW convention
    #
    # Ex.: tr.modes = (2,)
    # [0, 1, 2, 3, -2, -1] -> [0, 1, 2, -1]
    # [a, b, c, d,  e,  f] -> [a, b, c,  f]

    # calculate the retained modes taking into account the dimensions
    # that the spectral transform operates over
    size_space = size(c)[2:end-1] # sizes of space-like dimensions of c
    inds_space = 2:length(size_space)+1 # indices of space-like dimensions of c
    inds_map = Dict(zip(dims, tr.modes)) # maps index location to retained modes
    inds_offset = 1

    # we only truncate along dimensions contained in dims and otherwise keep
    # all modes
    modes = [i ∈ dims ? inds_map[i] : div(size_space[i-inds_offset], 2) for i in inds_space]

    # indices for the spectral coefficients that we need to retain
    inds = [vcat(collect(1:m+1), collect(s-m+2:s)) for (s, m) in zip(size(c)[2:end-1], modes)]
    c_truncated = OperatorFlux.mview(c, inds, Val(length(size_space)))

    return c_truncated
end

function pad_modes(::FourierTransform, c, size_pad::NTuple)
    # return a padded-with-zeros version of c assuming
    # that c is a tensor of spectral weights, thereby inflating c.
    # Want to keep 1:M+1 to end-M+2:end using FFTW convention, so need to 
    # fill rest with zeros.
    #
    # Ex.: dims = (6,)
    # [0, 1, 2, -1] -> [0, 1, 2, 3, -2, -1]
    # [a, b, c,  d] -> [a, b, c, 0,  0,  d]
    N = length(size_pad) # number of space-like dimensions
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
