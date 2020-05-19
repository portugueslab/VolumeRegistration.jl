# Kriging upsampling, translated into readable Julia from
# Suite2p by Carsen Stringer and Marius Pachitariu
# https://github.com/MouseLand/suite2p/blob/f3ef05d5d97e0fa879d30aaddd60c50ce57ee486/suite2p/registration/nonrigid.py

struct KrigingUpsampler{N,T}
    upsampling::NTuple{N,Int}
    original_size::NTuple{N,Int}
    upsampled_size::NTuple{N,Int}
    upsampling_mat::T
end

"""
Construct an upsampler with its matrix

"""
function KrigingUpsampler(; padding = (3, 3), upsampling = (10, 10), σ = 0.85)
    return KrigingUpsampler(
        upsampling,
        padding .* 2 .+ 1,
        padding .* upsampling .* 2 .+ 1,
        upsampling_matrix(padding, upsampling, σ),
    )
end

function kernel_distance(p1, p2, σ)
    return exp(-sum((p1 .- p2) .^ 2) / (2 * (σ^2)))
end

function upsampling_matrix(padding = (3, 3), upsampling = (10, 10), σ = 0.85)
    window_ranges = [(-pad:pad) for pad in padding]
    window_ranges_upsampled =
        [(-pad*ups:pad*ups) ./ ups for (pad, ups) in zip(padding, upsampling)]
    points_original, points_upsampled = (
        [SVector(xs) for xs in Iterators.product(ranges...)][:] for ranges in [window_ranges, window_ranges_upsampled]
    )
    kernel_dist_original =
        kernel_distance.(points_original, permutedims(points_original), σ)
    kernel_dist_original_ups =
        kernel_distance.(points_original, permutedims(points_upsampled), σ)
    return inv(kernel_dist_original) * kernel_dist_original_ups
end

function upsample(u::KrigingUpsampler{N,AT}, v::AbstractArray{T,N}) where {T,N,AT}
    return reshape(permutedims(v[:]) * u.upsampling_mat, u.upsampled_size...)
end

function upsample(u::KrigingUpsampler{N,AT}, v::AbstractArray{T,M}) where {T,N,AT,M}
    n_els = size(v)[N+1]
    return reshape(permutedims(reshape(v, :, n_els)) * u.upsampling_mat, n_els, u.upsampled_size...)
end

"""
Finds the shift in the area around the center of the orignial window with 
upsampling
"""
function upsampled_shift(u::KrigingUpsampler{N,AT},  v::AbstractArray{T,N}) where {T,N,AT}
    upsampled_indices = CartesianIndices(u.upsampled_size)
    max_val, max_ind = findmax(permutedims(v[:]) * u.upsampling_mat)
    half_ups_size = u.upsampled_size .÷ 2 .+ 1
    return (upsampled_indices[max_ind[2]].I .- half_ups_size) ./ u.upsampling
end

function upsampled_shift(u::KrigingUpsampler{N,AT},  v::AbstractArray{T,M}) where {T,N,AT,M}
    n_els = size(v)[N+1]
    upsampled_indices = CartesianIndices(u.upsampled_size)
    max_vals, max_inds = findmax(permutedims(reshape(v, :, n_els)) * u.upsampling_mat, dims=2)
    half_ups_size = u.upsampled_size .÷ 2 .+ 1
    return [(upsampled_indices[mi.I[2]].I .- half_ups_size) ./ u.upsampling for mi in max_inds]
end