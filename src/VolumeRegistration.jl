module VolumeRegistration

using AbstractFFTs
using FFTW
using LinearAlgebra
using StaticArrays
using Statistics


include("window.jl")
include("kriging.jl")
include("phase_correlation.jl")
include("translation.jl")

export find_translation

end
