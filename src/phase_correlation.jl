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

function gaussian_fft_filter(shape, σ)
    mid_shape = shape ./ 2
    σ2 = 2*(σ^2)
    gauss_filt = Float32.([exp(-sum((idx.I .- mid_shape).^2)/σ2) for idx in CartesianIndices(shape)])
    gauss_filt ./= sum(gauss_filt)
    return Float32.(real(fft(ifftshift(gauss_filt))))
end

function prepare_fft_reference(target_img, σ_ref::Real=1.15f0)
    return normalize.(conj.(fft(target_img))) .* gaussian_fft_filter(size(target_img), σ_ref)
end

function prepare_fft_reference(target_img, σ_ref::Nothing)
    return normalize.(conj.(fft(target_img)))
end

function phase_correlation(src_img::AbstractArray{T,N}, target_img::AbstractArray{T,N}; σ_ref::Union{Nothing, Real}=T(1.15)) where {T <:Real, N}
    return phase_correlation(fft(src_img), prepare_fft_reference(target_img, filter)) 
end

function phase_correlation(src_img::AbstractArray{T,N}, target_img::AbstractArray{T2,N}) where {T <:Real, T2 <:Complex, N}
    return phase_correlation(fft(src_img), target_img) 
end

function phase_correlation(src_freq::AbstractArray{T, N}, target_freq::AbstractArray{T, N}) where {T <: Complex{T2}, N} where {T2}
    image_product = normalize.(src_freq) .* target_freq;
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
    os_mid = us.original_size .÷ 2
    window_size = min.(size(pc), window_size)
    window_mid  = window_size .÷ 2 .+ 1
    lf = VolumeRegistration.extract_low_frequencies(pc, window_size)
    max_loc = argmax(lf[CartesianIndex(os_mid):CartesianIndex(size(lf) .- os_mid)]).I .+ os_mid .-1
    ups_shift = VolumeRegistration.upsampled_shift(us, lf[CartesianIndex(max_loc .- os_mid) : CartesianIndex(max_loc .+ os_mid)])
    return ups_shift .+ max_loc .- window_mid
end