using OperatorFlux
using Test
using FFTW
using Flux
using LinearAlgebra
using Revise
using ChainRulesCore
using ChainRulesTestUtils

include("fourier.jl")
include("chebyshev.jl")
include("operators.jl")
