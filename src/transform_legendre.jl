"""
    LegendreTransform(; modes)

Constructs a legendre transform with modes `modes` that 
operates on the 2nd, 3rd, ...N-modes+1-st dimension of the data.

# Example
```
julia> LegendreTransform(modes = (12, 3, 4, 12))
```
"""
Base.@kwdef struct LegendreTransform{N, T} <: AbstractTransform
    modes::NTuple{N, T}
end

function forward(::LegendreTransform, x)
    L = legendre_transform_forward_matrix.(size(x)[2:(end - 1)])
    return apply_legendre(L, x)
end

function inverse(::LegendreTransform, x)
    L⁻¹ = legendre_transform_inverse_matrix.(size(x)[2:(end - 1)])
    return apply_legendre(L⁻¹, x)
end

function truncate_modes(tr::LegendreTransform{N}, coeff) where {N}
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
    ::LegendreTransform,
    coeff,
    size_pad::NTuple{N, T},
) where {N, T}
    # return a padded-with-zeros version of coeff assuming
    # that coeff is a tensor of spectral weights, thereby inflating coeff.
    size_space = size(coeff)[2:(end - 1)] # sizes of space-like dimensions of coeff

    # generate a zero array and indices that point to the filled-in
    # locations
    coeff_padded =
        zeros(eltype(coeff), (size(coeff)[1], size_pad..., size(coeff)[end]))
    inds = [collect(1:m) for (s, m) in zip(size_pad, size(coeff)[2:(end - 1)])]

    coeff_padded_view = OperatorFlux.mview(coeff_padded, inds, Val(N))
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

    for n in 1:(N - 1)
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
    @tullio c[s, i, b] := F1[i, ii] * x[s, ii, b]
end

function apply_legendre(w::Tuple{S, T}, x) where {S, T}
    F1, F2 = w
    @tullio c[s, i, j, b] := F1[i, ii] * F2[j, jj] * x[s, ii, jj, b]
end

function apply_legendre(w::Tuple{S, T, V}, x) where {S, T, V}
    F1, F2, F3 = w
    @tullio c[s, i, j, k, b] :=
        F1[i, ii] * F2[j, jj] * F3[k, kk] * x[s, ii, jj, kk, b]
end

# Base extensions
Base.ndims(tr::LegendreTransform) = length(tr.modes)
Base.eltype(::LegendreTransform) = Float32
Base.size(tr::LegendreTransform) = tr.modes

function Base.show(io::IO, tr::LegendreTransform)
    print(io, "LegendreTransform(modes = $(tr.modes))")
end
