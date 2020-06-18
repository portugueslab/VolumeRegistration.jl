module VolumeRegistration

using AbstractFFTs
using DocStringExtensions
using FFTW
using Interpolations
using Logging
using JuliennedArrays
using CoordinateTransformations
using LinearAlgebra
using StaticArrays
using Statistics
using SparseArrays
using Strided
using PaddedBlocks
using ProgressMeter
using ThreadTools

include("utilities.jl")
include("window.jl")
include("kriging.jl")
include("reference.jl")
include("phase_correlation.jl")
include("translation.jl")
include("deformation.jl")
include("transformation.jl")
include("pipelines.jl")

export make_reference,
    find_translation,
    translate,
    find_deformation_map,
    apply_deformation_map,
    register_volumes!,
    make_planewise_reference,
    register_planewise!
end
