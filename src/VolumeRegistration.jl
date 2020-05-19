module VolumeRegistration

using AbstractFFTs
using LinearAlgebra
using StaticArrays
using FFTW

include("window.jl")
include("registration.jl")
include("kriging.jl")

end
