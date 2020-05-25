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


function translate!(translated, a::AbstractArray{T, N}, shifts::NTuple{N, Integer}) where {N, T}
    full_indices = CartesianIndices(a)
    min_corner = first(full_indices)
    max_corner = last(full_indices)
    shift_idx = CartesianIndex(shifts)
    src_indices = max(min_corner, min_corner + shift_idx):min(max_corner, max_corner + shift_idx)
    dest_indices = max(min_corner, min_corner - shift_idx):min(max_corner, max_corner - shift_idx)
    translated[dest_indices] .= a[src_indices]
    return translated
end

function translate!(translated, a::AbstractArray{T, N}, shifts::NTuple{N, Real}) where {N, T}
    translated .= fft_translate(a, shifts)
end

function translate(a::AbstractArray{T, N}, shift::NTuple{N, Integer}) where {N, T}
    translated = Array{Union{Missing, T}}(missing, size(a))
    translate!(translated, a, shift)
    return translated
end

function translate(a::AbstractArray{T, N}, shift::NTuple{N, Real}) where {T, N}
    translated = Array{T}(undef, size(a))
    translate!(translated, a, shift)
    return translated
end

function translate(a::AbstractArray{T, M}, shifts::AbstractArray{NTuple{N, TS}, 1}) where {M, N, T, TS<:Integer}
    translated = Array{Union{Missing, T}}(missing, size(a))
    sliced_source = Slices(a, (1:N)...)
    sliced_target = Slices(translated, (1:N)...)
    for (i_sl, sl_source) in enumerate(sliced_source)
        translate!(sliced_target[i_sl], sl_source, shifts[i_sl])
    end
    return translated
end


# version for subpixel shifts, does not have missings TODO handle without copy pasting
function translate(a::AbstractArray{T, M}, shifts::AbstractArray{NTuple{N, TS}, 1}) where {M, N, T, TS<:Real}
    translated = Array{T}(undef, size(a))
    sliced_source = Slices(a, (1:N)...)
    sliced_target = Slices(translated, (1:N)...)
    for (i_sl, sl_source) in enumerate(sliced_source)
        translate!(sliced_target[i_sl], sl_source, shifts[i_sl])
    end
    return translated
end

# function fft_translate(a::CuArray{T, N}, shifts) where {N, T <: Complex }
#     shift_mat = fftshift(exp.( -im*2π .* cu(Complex{Float32}.(sum_shift.(CartesianIndices(a), Ref(-shifts), Ref(size(a)))))));
#     return abs.(ifft(a.*shift_mat))
# end