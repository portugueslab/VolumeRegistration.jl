module VolumeRegistration

using AbstractFFTs
using DocStringExtensions
using FFTW
using JuliennedArrays
using LinearAlgebra
using StaticArrays
using Statistics
using PaddedBlocks

include("utilities.jl")
include("window.jl")
include("kriging.jl")
include("phase_correlation.jl")
include("translation.jl")
include("nonrigid.jl")
include("shifting.jl")

export find_translation, translate

end
