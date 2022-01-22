module OperatorFlux

using Flux
using FFTW
using GaussQuadrature
using LinearAlgebra
using Tullio
using ChainRulesCore

export
    FourierTransform,
    Chebyshev,
    SpectralConv,
    SpectralKernelOperator,
    forward,
    inverse,
    truncate_modes,
    pad_modes

"""
    Supertype for all spectral transforms.
"""
abstract type AbstractTransform end

"""
    Supertype for all operators.
"""
abstract type AbstractOperator end

"""
    forward(trafo, x)

Forward transform of `trafo` applied to `x`.
"""
forward(trafo::AbstractTransform, _...) =
    error("forward not implemented for given trafo")

"""
    inverse(trafo, x)

Inverse transform of `trafo` applied to `x`.
"""
inverse(trafo::AbstractTransform, _...) =
    error("inverse not implemented for given trafo")

"""
    truncate_modes(trafo, c)

Spectral mode truncation for transform of `trafo` applied to `c`.
"""
truncate_modes(trafo::AbstractTransform, _...) =
    error("truncate_modes not implemented for given trafo")

"""
    pad_modes(trafo, c, size_pad)

Spectral mode padding of size `size_pad` for transform of `trafo` applied to `c`.
"""
pad_modes(trafo::AbstractTransform, _...) =
    error("pad_modes not implemented for given trafo")

include("utils.jl")
include("fourier.jl")
include("chebyshev.jl")
include("legendre.jl")
include("operators.jl")

end # module
