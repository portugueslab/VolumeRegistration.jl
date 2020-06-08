function sum_shift(t, shift, size)
    return sum(((t.I .- size .÷ 2) .* shift) ./ size)
end

function fft_translate(a::AbstractArray{T,N}, shifts) where {T,N}
    return fft_translate(fft(a), shifts)
end

"Translate using discrete fourier transform"
function fft_translate(a::AbstractArray{T,N}, shifts) where {N,T<:Complex}
    shift_mat =
        T.(fftshift(exp.(
            -im * 2π .* sum_shift.(CartesianIndices(a), Ref(.-shifts), Ref(size(a))),
        )))
    return abs.(ifft(a .* shift_mat))
end

function translate!(
    translated,
    a::AbstractArray{T,N},
    translation::Translation{SVector{N,TT}},
) where {N,T,TT<:Integer}
    shifts = Tuple(translation.translation)
    full_indices = CartesianIndices(a)
    min_corner = first(full_indices)
    max_corner = last(full_indices)
    shift_idx = CartesianIndex(shifts)
    src_indices =
        max(min_corner, min_corner + shift_idx):min(max_corner, max_corner + shift_idx)
    dest_indices =
        max(min_corner, min_corner - shift_idx):min(max_corner, max_corner - shift_idx)
    translated[dest_indices] .= a[src_indices]
    return translated
end

function translate!(
    translated::AbstractArray{T,N},
    a::AbstractArray{T,N},
    translation::Translation{SVector{N,TT}},
) where {N,T<:AbstractFloat,TT<:AbstractFloat}
    return translated .= fft_translate(a, Tuple(translation.translation))
end

function translate!(
    translated::AbstractArray{T,N},
    a::AbstractArray{T,N},
    translation::Translation{SVector{N,TT}},
) where {N,T<:Integer,TT<:AbstractFloat}
    return translated .= round.(T, fft_translate(a, Tuple(translation.translation)))
end

# preallocation functions: allocates an union for missing
function preallocate(::Type{T}, ::Missing, dimensions) where {T}
    return Array{Union{T,Missing}}(missing, dimensions)
end

function preallocate(::Type{T}, val::T, dimensions) where {T}
    return fill(val, dimensions)
end

"""
Translate an image by known shift using either reindexing
(for integer shifts) of FFT-based phase-space translation
for non-integer shifts.

"""
function translate(
    a::AbstractArray{T,N},
    shift::Translation,
    fill_value = zero(T),
) where {N,T}
    translated = preallocate(T, fill_value, size(a))
    translate!(translated, a, shift)
    return translated
end

function translate(
    a::AbstractArray{T,M},
    shifts::AbstractArray{TR,1},
    fill_value = zero(T),
) where {M,T,TR<:Translation}
    translated = preallocate(T, fill_value, size(a))
    Base.Threads.@threads for i_slice in 1:size(a, M)
        translate!(
            view(translated,ncolons(Val{M-1}())..., i_slice),
            view(a,ncolons(Val{M-1}())..., i_slice),
             shifts[i_slice]
             )
    end
    return translated
end

# Nonrigid part

function warp_nonrigid!(
    dest::AbstractArray{T,N},
    moving::AbstractArray{T,N},
    shifts,
    blocks,
    image_points,
    nonmorphed_points,
) where {T,N}
    morph = shifts_to_extrapolation(shifts, blocks)
    morphed_points = morph.(image_points...) .+ nonmorphed_points
    im_interp = extrapolate(interpolate(moving, BSpline(Linear())), Flat())
    dest .= T.(im_interp.((getindex.(morphed_points, i) for i in 1:N)...))
    return dest
end


"""
Corrects a plane or volume with a transformation found through non-rigid registration

"""
function apply_deformation_map(
    moving::AbstractArray{T,N},
    shifts::AbstractArray{NTuple{N,TS},N},
    blocks,
) where {T,N,TS}
    image_points =
        [[idx[i_dim] for idx in Iterators.product(axes(moving)...)] for i_dim in 1:N]
    nonmorphed_points = [SVector(idx) for idx in Iterators.product(axes(moving)...)]
    moved = similar(moving)
    warp_nonrigid!(
        moved,
        moving,
        shifts,
        blocks,
        image_points,
        nonmorphed_points
    )
    return moved
end

"""
Corrects a sequence of imaging stacks with a transformation found through non-rigid registration

"""
function apply_deformation_map(
    moving::AbstractArray{T,M},
    shifts::Array{Array{NTuple{N,TS},N}},
    blocks,
) where {T,N,TS,M}
    image_points =
        [[idx[i_dim] for idx in Iterators.product(axes(moving)[1:N]...)] for i_dim in 1:N]
    morphed_points = Array{SVector{N,Float32},N}(undef, size(image_points[1]))
    nonmorphed_points = [SVector(idx) for idx in Iterators.product(axes(moving)[1:N]...)]
    moved = similar(moving)
    Base.Threads.@threads for i_slice in 1:size(moved, M)
        warp_nonrigid!(
            view(moved, ncolons(Val{M-1}())..., i_slice),
            view(moving, ncolons(Val{M-1}())..., i_slice),
            shifts[i_slice],
            blocks,
            image_points,
            nonmorphed_points)
    end
    return moved
end

# function fft_translate(a::CuArray{T, N}, shifts) where {N, T <: Complex }
#     shift_mat = fftshift(exp.( -im*2π .* cu(Complex{Float32}.(sum_shift.(CartesianIndices(a), Ref(-shifts), Ref(size(a)))))));
#     return abs.(ifft(a.*shift_mat))
# end
