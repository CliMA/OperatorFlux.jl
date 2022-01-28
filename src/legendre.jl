struct SpectralElementTransform{JD,J,L,LI,D} <: AbstractTransform
    jacobian_det::JD
    jacobian::J
    forward::L
    inverse::LI
    dims::D
end

function SpectralElementTransform(; modes::NTuple{N,T}) where {N,T}
    jacobian_det = nothing
    jacobian = nothing
    forward = legendre_transform_forward_matrix.(modes)
    inverse = legendre_transform_inverse_matrix.(modes)

    dims = 2:(N-2)
    SpectralElementTransform(jacobian_det, jacobian, forward, inverse, dims)
end

function forward(tr::SpectralElementTransform{N}, x, dims = tr.dims) where {N}
    return apply_legendre(tr.forward, x)
end

function inverse(tr::SpectralElementTransform{N}, x, dims = tr.dims) where {N}
    return apply_legendre(tr.inverse, x)
end

function truncate_modes(tr::SpectralElementTransform{N}, c, dims = tr.dims) where {N}
    # return a low-pass filtered version of c assuming
    # that c is a tensor of spectral weights.
    # Want to keep 1:M+1 to end-M+2:end using FFTW convention
    #
    # Ex.: tr.modes = (2,)
    # [0, 1, 2, 3] -> [0, 1]
    # [a, b, c, d] -> [a, b]

    # indices for the spectral coefficients that we need to retain
    inds = [collect(1:m) for m in modes]
    c_truncated = OperatorFlux.mview(c, inds, Val(N))

    return c_truncated
end

# function pad_modes(::SpectralElementTransform, c, size_pad::NTuple)
#     # return a padded-with-zeros version of c assuming
#     # that c is a tensor of spectral weights, thereby inflating c.
#     # Want to keep 1:M+1 to end-M+2:end using FFTW convention, so need to 
#     # fill rest with zeros.
#     #
#     # Ex.: dims = (6,)
#     # [0, 1, 2, -1] -> [0, 1, 2, 3, -2, -1]
#     # [a, b, c,  d] -> [a, b, c, 0,  0,  d]
#     N = length(size_pad) # number of space-like dimensions
#     c_padded = zeros(eltype(c), (size(c)[1], size_pad..., size(c)[end]))
#     inds = [vcat(collect(1:div(m, 2)+1), collect(s-div(m, 2)+2:s)) for (s, m) in zip(size_pad, size(c)[2:end-1])]
#     c_padded_view = OperatorFlux.mview(c_padded, inds, Val(N))
#     c_padded_view .= c

#     return c_padded
# end

# utils
function vandermonde(x, N)
    # create view to assign values
    P = zeros(length(x), N + 1)
    P⁰ = view(P, :, 0 + 1)
    @. P⁰ = 1

    # explicitly compute second coefficient
    if N == 0
        return P
    end

    P¹ = view(P, :, 1 + 1)
    @. P¹ = x

    if N == 1
        return P
    end

    for n in 1:(N-1)
        # get views for ith, i-1th, and i-2th columns
        Pⁿ⁺¹ = view(P, :, n + 1 + 1)
        Pⁿ = view(P, :, n + 0 + 1)
        Pⁿ⁻¹ = view(P, :, n - 1 + 1)

        # compute coefficients for ith column
        @. Pⁿ⁺¹ = ((2n + 1) * x * Pⁿ - n * Pⁿ⁻¹) / (n + 1)
    end

    return P
end

function legendre_transform_inverse_matrix(N)
    # get the legendre points to construct transformation
    # matrix. For small transforms this is performant.
    x, _ = GaussQuadrature.legendre(N, GaussQuadrature.both)
    return vandermonde(x, N - 1)
end

function legendre_transform_forward_matrix(N)
    return legendre_transform_inverse_matrix(N) \ I
end

function apply_legendre(w::Tuple{S}, x) where {S}
    F1, = w
    @tullio c[s, i, e, b] := F1[i, ii] * x[s, ii, e, b]
end

function apply_legendre(w::Tuple{S,T}, x) where {S,T}
    F1, F2 = w
    @tullio c[s, i, j, e, b] := F1[i, ii] * F2[j, jj] * x[s, ii, jj, e, b]
end

function apply_legendre(w::Tuple{S, T, V}, x) where {S, T, V}
    F1, F2, F3 = w
    @tullio c[s, i, j, k, e, b] := F1[i, ii] * F2[j, jj] * F3[k, kk] * x[s, ii, jj, kk, e, b]
end


# Base extensions
Base.ndims(::SpectralElementTransform{N}) where {N} = N
Base.eltype(::SpectralElementTransform) = Float32
Base.size(tr::SpectralElementTransform) = tr.modes

function Base.show(io::IO, tr::SpectralElementTransform)
    print(
        io,
        "SpectralElementTransform(modes = $(tr.modes))"
    )
end
