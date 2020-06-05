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
using SparseArrays
using PaddedBlocks

include("utilities.jl")
include("window.jl")
include("kriging.jl")
include("reference.jl")
include("phase_correlation.jl")
include("translation.jl")
include("nonrigid.jl")
include("transformation.jl")

export make_reference,
    find_translation, translate, find_deformation_map, apply_deformation_map

end
