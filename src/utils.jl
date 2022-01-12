mview(c, inds, ::Val{1}) = view(c, :, inds[1], :)
mview(c, inds, ::Val{2}) = view(c, :, inds[1], inds[2], :)
mview(c, inds, ::Val{3}) = view(c, :, inds[1], inds[2], inds[3], :)
mview(c, inds, ::Val{4}) = view(c, :, inds[1], inds[2], inds[3], inds[4], :)

tensor_contraction(
    A,
    B,
    ::Val{1},
) = @tullio C[o, a, b] := A[i, a, o] * B[i, a, b]
tensor_contraction(A, B, ::Val{2}) =
    @tullio C[o, a₁, a₂, b] := A[i, a₁, a₂, o] * B[i, a₁, a₂, b]
tensor_contraction(A, B, ::Val{3}) =
    @tullio C[o, a₁, a₂, a₃, b] := A[i, a₁, a₂, a₃, o] * B[i, a₁, a₂, a₃, b]
tensor_contraction(A, B, ::Val{4}) = @tullio C[o, a₁, a₂, a₃, a₄, b] :=
    A[i, a₁, a₂, a₃, a₄, o] * B[i, a₁, a₂, a₃, a₄, b]

sparse_mean(w, c, ::Val{1}) = @tullio μ[o, a, b] := w[i, o] * c[i, a, b]
sparse_mean(
    w,
    c,
    ::Val{2},
) = @tullio μ[o, a₁, a₂, b] := w[i, o] * c[i, a₁, a₂, b]
sparse_mean(w, c, ::Val{3}) =
    @tullio μ[o, a₁, a₂, a₃, b] := w[i, o] * c[i, a₁, a₂, a₃, b]
sparse_mean(w, c, ::Val{4}) =
    @tullio μ[o, a₁, a₂, a₃, a₄, b] := w[i, o] * c[i, a₁, a₂, a₃, a₄, b]

sparse_covariance(
    w,
    c,
    ::Val{1},
) = @tullio μ[o, a, r, b] := w[i, r, o] * c[i, a, b]
sparse_covariance(w, c, ::Val{2}) =
    @tullio μ[o, a₁, a₂, r, b] := w[i, r, o] * c[i, a₁, a₂, b]
sparse_covariance(w, c, ::Val{3}) =
    @tullio μ[o, a₁, a₂, a₃, r, b] := w[i, r, o] * c[i, a₁, a₂, a₃, b]
sparse_covariance(w, c, ::Val{4}) =
    @tullio μ[o, a₁, a₂, a₃, a₄, r, b] := w[i, r, o] * c[i, a₁, a₂, a₃, a₄, b]
