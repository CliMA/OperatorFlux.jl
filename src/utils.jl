mview(c, inds, ::Val{1}) = view(c, :, inds[1], :)
mview(c, inds, ::Val{2}) = view(c, :, inds[1], inds[2], :)
mview(c, inds, ::Val{3}) = view(c, :, inds[1], inds[2], inds[3], :)
mview(c, inds, ::Val{4}) = view(c, :, inds[1], inds[2], inds[3], inds[4], :)
