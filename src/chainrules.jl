function ChainRulesCore.rrule(::typeof(r2r), x::AbstractArray, kind, dims)
    # only works in 1D for now :O
    (M,) = size(x)[2:(end - 1)]
    a1 = ones(M)
    a2 = [(-1)^i for i in 1:M]
    a2[1] = a2[end] = 0.0
    a1[1] = a1[end] = 0.0
    e1 = zeros(M)
    e1[1] = 1.0
    eN = zeros(M)
    eN[end] = 1.0

    function r2r_pullback(y)
        # r2r pullback turns out to be r2r + a rank 4 correction
        w = r2r(y, kind, dims)
        @tullio w[s, i, b] +=
            a1[i] * e1[k] * y[s, k, b] - a2[i] * eN[k] * y[s, k, b]
        @tullio w[s, i, b] +=
            eN[i] * a2[k] * y[s, k, b] - e1[i] * a1[k] * y[s, k, b]
        return NoTangent(), w, NoTangent(), NoTangent()
    end

    return r2r(x, kind, dims), r2r_pullback
end

function ChainRulesCore.rrule(
    ::typeof(truncate_modes),
    tr::AbstractTransform,
    c::AbstractArray,
    dims,
)
    function truncate_modes_pullback(x)
        # truncate_modes pullback turns out to be pad_modes
        return NoTangent(),
        NoTangent(),
        @thunk(pad_modes(tr, x, size(c)[2:(end - 1)])),
        NoTangent()
    end

    return truncate_modes(tr, c, dims), truncate_modes_pullback
end

function ChainRulesCore.rrule(
    ::typeof(pad_modes),
    tr::AbstractTransform,
    c::AbstractArray,
    size_pad::NTuple,
    dims,
)
    function pad_modes_pullback(x)
        # pad_modes pullback turns out to be truncate_modes

        return NoTangent(),
        NoTangent(),
        @thunk(truncate_modes(tr, x, dims)),
        NoTangent(),
        NoTangent()
    end

    return pad_modes(tr, c, size_pad, dims), pad_modes_pullback
end
