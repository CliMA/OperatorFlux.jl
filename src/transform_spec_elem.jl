struct SpectralElementTransform{JD,J,L,LI,M,D} <: AbstractTransform
    jacobian_det::JD
    jacobian::J
    forward::L
    inverse::LI
    modes::M
    dims::D
end

function SpectralElementTransform(; modes::NTuple{N,T}) where {N,T}
    jacobian_det = nothing
    jacobian = nothing
    forward = legendre_transform_forward_matrix.(modes)
    inverse = legendre_transform_inverse_matrix.(modes)

    dims = 2:N+1
    SpectralElementTransform(jacobian_det, jacobian, forward, inverse, modes, dims)
end

function forward(tr::SpectralElementTransform{N}, x, dims = tr.dims) where {N}
    return apply_legendre(tr.forward, x)
end

function inverse(tr::SpectralElementTransform{N}, x, dims = tr.dims) where {N}
    return apply_legendre(tr.inverse, x)
end

function truncate_modes(tr::SpectralElementTransform{N}, coeff, dims = tr.dims) where {N}
    # Return a low-pass filtered version of coeff assuming
    # that coeff is a tensor of spectral weights.
    #
    # Ex.: tr.modes = (3,)
    # TODO! Be consistent with truncation conventions.
    # [0, 1, 2, 3, 4, 5] -> [0, 1, 2]
    # [a, b, c, d, e, f] -> [a, b, c]
    # calculate the retained modes taking into account the dimensions
    # that the spectral transform operates over
    size_space = size(coeff)[2:(end - 2)] # sizes of space-like dimensions of coeff
    inds_space = 2:(length(size_space) + 1) # indices of space-like dimensions of coeff
    inds_map = Dict(zip(dims, tr.modes)) # maps index location to retained modes
    inds_offset = 1 # because the first data index is channel dimension

    # we only truncate along dimensions contained in dims and otherwise keep
    # all modes
    modes = [
        i ∈ dims ? inds_map[i] : size_space[i - inds_offset]
        for i in inds_space
    ]

    # indices for the spectral coefficients that we need to retain
    inds = [collect(1:m) for (s, m) in zip(size(coeff)[2:(end - 2)], modes)]

    elem = true
    coeff_truncated = OperatorFlux.mview(coeff, inds, Val(length(size_space)), elem)

    return coeff_truncated    
end

function pad_modes(
    tr::SpectralElementTransform, 
    coeff,
    size_pad::NTuple,
    dims = tr.dims,
)
    # return a padded-with-zeros version of coeff assuming
    # that coeff is a tensor of spectral weights, thereby inflating coeff.
    N = length(size_pad) # number of space-like dimensions
    size_space = size(coeff)[2:(end-1)] # sizes of space-like dimensions of coeff

    # generate a zero array and indices that point to the filled-in
    # locations
    coeff_padded =
        zeros(eltype(coeff), (size(coeff)[1], size_pad..., size(coeff)[end-1:end]...))
    inds = [collect(1:m) for (s, m) in zip(size_pad, size(coeff)[2:(end-2)])]

    elem = true
    coeff_padded_view = OperatorFlux.mview(coeff_padded, inds, Val(N), elem)
    coeff_padded_view .= coeff

    return coeff_padded
end

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
Base.ndims(tr::SpectralElementTransform) = length(tr.modes)
Base.eltype(::SpectralElementTransform) = Float32
Base.size(tr::SpectralElementTransform) = tr.modes

function Base.show(io::IO, tr::SpectralElementTransform)
    print(
        io,
        "SpectralElementTransform(modes = $(tr.modes))"
    )
end
