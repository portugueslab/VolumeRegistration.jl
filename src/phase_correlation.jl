function aamax(x::Array)
    return argmax(x)
end

function aamax(x)
    i_m = argmax(x)
    indices = CartesianIndices(size(x))
    return indices[i_m]
end

function normalize(x::Complex{T}) where {T}
    return x / (abs(x) + eps(T))
end

function phase_correlation(src_img::AbstractArray{T,N}, target_img::AbstractArray{T,N};  kwargs...) where {T <:Real, N}
    return phase_correlation(fft(src_img), normalize.(conj.(fft(target_img))) , kwargs...) 
end

function phase_correlation(src_img::AbstractArray{T,N}, target_img::AbstractArray{T2,N};  kwargs...) where {T <:Real, T2 <:Complex, N}
    return phase_correlation(fft(src_img), target_img, kwargs...) 
end

function phase_correlation(src_freq::AbstractArray{T, N}, target_freq::AbstractArray{T, N}; filter::Union{Nothing, AbstractArray{T, N}}=nothing) where {T <: Complex{T2}, N} where {T2}
    if filter === nothing
        image_product = src_freq .* normalize.(target_freq);
    else
        image_product = src_freq .* normalize.(target_freq) .* filter;
    end
    ifft!(image_product)

    return image_product
end

"""
Takes the data corresponding to the real part of the corners of the 
phase correlation array of interest for shift finding

"""
function extract_low_frequencies(data::AbstractArray{Complex{T}, N}, corner_size::NTuple{N, Integer}) where {T, N}
    corner = Array{T}(undef, corner_size)
    mid_val = corner_size .÷ 2 
    data_size = size(data)

    for idx in CartesianIndices(corner_size)
        idx_original = ((x, mid, full) -> (x > mid) ? (x - mid) : full - mid + x).(idx.I, mid_val, data_size)
        corner[idx] = real(data[idx_original...])
    end
    return corner
end

function phase_correlation_shift(pc, window_size)
    window_mid  = window_size .÷ 2 .+ 1
    lf = extract_low_frequencies(pc, window_size)
    max_loc = argmax(lf).I
    return max_loc .- window_mid
end

function phase_correlation_shift(pc, window_size, us)
    window_mid  = window_size .÷ 2 .+ 1
    os_mid = us.original_size .÷ 2
    lf = VolumeRegistration.extract_low_frequencies(pc, window_size)
    max_loc = argmax(lf[CartesianIndex(os_mid):CartesianIndex(size(lf) .- os_mid)]).I .+ os_mid .-1
    ups_shift = VolumeRegistration.upsampled_shift(us, lf[CartesianIndex(max_loc .- os_mid) : CartesianIndex(max_loc .+ os_mid)])
    return ups_shift .+ max_loc .- window_mid
end