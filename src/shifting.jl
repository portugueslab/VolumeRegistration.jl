function sum_shift(t, shift, size)
    return sum(((t.I .- size .÷ 2) .* shift)./size)
 end

function fft_translate(a::AbstractArray{T, N}, shifts) where {T, N}
    return fft_translate(fft(a), shifts)
end

"Translate using discrete fourier transform"
function fft_translate(a::AbstractArray{T, N}, shifts) where {N, T <: Complex }
    shift_mat = T.(fftshift(exp.( -im*2π .* sum_shift.(CartesianIndices(a), Ref(.-shifts), Ref(size(a))))));
    return abs.(ifft(a.*shift_mat))
end


function translate!(translated, a::AbstractArray{T, N}, translation::Translation{SVector{N, TT}}) where {N, T, TT <: Integer}
    shifts = Tuple(translation.translation)
    full_indices = CartesianIndices(a)
    min_corner = first(full_indices)
    max_corner = last(full_indices)
    shift_idx = CartesianIndex(shifts)
    src_indices = max(min_corner, min_corner + shift_idx):min(max_corner, max_corner + shift_idx)
    dest_indices = max(min_corner, min_corner - shift_idx):min(max_corner, max_corner - shift_idx)
    translated[dest_indices] .= a[src_indices]
    return translated
end

function translate!(translated, a::AbstractArray{T, N}, translation::Translation{SVector{N, TT}}) where {N, T, TT <: Real}
    translated .= fft_translate(a, Tuple(translation.translation))
end

# preallocation functions: allocates an union for missing
function preallocate(::Type{T}, ::Missing, dimensions) where {T}
    return Array{Union{T, Missing}}(missing, dimensions)
end

function preallocate(::Type{T}, val::T, dimensions) where {T}
    return fill(val, dimensions)
end

function translate(a::AbstractArray{T, N}, shift::Translation, fill_value=zero(T)) where {N, T}
    translated = preallocate(T, fill_value, size(a))
    translate!(translated, a, shift)
    return translated
end

function translate(a::AbstractArray{T, M}, shifts::AbstractArray{TR, 1}, fill_value=zero(T)) where {M, T, TR<:Translation}
    translated = preallocate(T, fill_value, size(a))
    foreach(translate!, eachslice(translated, dims=M), eachslice(a, dims=M), shifts)
    return translated
end

# function fft_translate(a::CuArray{T, N}, shifts) where {N, T <: Complex }
#     shift_mat = fftshift(exp.( -im*2π .* cu(Complex{Float32}.(sum_shift.(CartesianIndices(a), Ref(-shifts), Ref(size(a)))))));
#     return abs.(ifft(a.*shift_mat))
# end