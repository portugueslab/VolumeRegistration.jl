using VolumeRegistration

using FFTW
using ImageTransformations
using CoordinateTransformations
using Test

include("utilities.jl")
include("kriging.jl")
include("nonrigid.jl")
include("translation.jl")
include("reference.jl")