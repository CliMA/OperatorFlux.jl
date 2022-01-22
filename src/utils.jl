mview(c, inds, ::Val{1}) = view(c, :, inds[1], :)
mview(c, inds, ::Val{2}) = view(c, :, inds[1], inds[2], :)
mview(c, inds, ::Val{3}) = view(c, :, inds[1], inds[2], inds[3], :)
mview(c, inds, ::Val{4}) = view(c, :, inds[1], inds[2], inds[3], inds[4], :)

# Chainrules extensions
function ChainRulesCore.rrule(::typeof(truncate_modes), tr::AbstractTransform, c::AbstractArray)
    function truncate_modes_pullback(x)
        # truncate_modes pullback turns out to be pad_modes
        return NoTangent(), NoTangent(), @thunk(pad_modes(tr, x, size(c)[2:end-1]))
    end

    return truncate_modes(tr, c), truncate_modes_pullback
end

function ChainRulesCore.rrule(::typeof(truncate_modes), tr::AbstractTransform, c::AbstractArray, dims)
    function truncate_modes_pullback(x)
        # truncate_modes pullback turns out to be pad_modes
        return NoTangent(), NoTangent(), @thunk(pad_modes(tr, x, size(c)[2:end-1])), NoTangent()
    end

    return truncate_modes(tr, c, dims), truncate_modes_pullback
end

function ChainRulesCore.rrule(::typeof(pad_modes), tr::AbstractTransform, c::AbstractArray, size_pad::NTuple)
    function pad_modes_pullback(x)
        # pad_modes pullback turns out to be truncate_modes
        return NoTangent(), NoTangent(), @thunk(truncate_modes(tr, x)), NoTangent()
    end

    return pad_modes(tr, c, size_pad), pad_modes_pullback
end
