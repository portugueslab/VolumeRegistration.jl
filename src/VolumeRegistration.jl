module VolumeRegistration

using AbstractFFTs
using DocStringExtensions
using FFTW
using Interpolations
using JuliennedArrays
using CoordinateTransformations
using LinearAlgebra
using StaticArrays
using Statistics
using PaddedBlocks

include("utilities.jl")
include("window.jl")
include("kriging.jl")
include("reference.jl")
include("phase_correlation.jl")
include("translation.jl")
include("nonrigid.jl")
include("shifting.jl")

export find_translation, translate, find_reference, find_deformation_map, refine_reference

end
