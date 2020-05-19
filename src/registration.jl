function aamax(x::Array)
    return argmax(x)
end

function aamax(x)
    i_m = argmax(x)
    indices = CartesianIndices(size(x))
    return indices[i_m]
end

function register_translation(src_img::AbstractArray{T,N}, target_img::AbstractArray{T,N};  kwargs...) where {T,N}
    return register_translation(fft(src_img), conj.(fft(target_img)), kwargs...) 
end

function register_translation(src_img::AbstractArray{T,N}, target_img::AbstractArray{T2,N};  kwargs...) where {T, T2 <:Complex, N}
    return register_translation(fft(src_img), target_img, kwargs...) 
end

function phase_correlation(src_freq::AbstractArray{T, N}, target_freq::AbstractArray{T, N}; upsample_factor = 1, filter::Union{Nothing, AbstractArray{T, N}}=nothing) where {T <: Complex, N}
    if filter === nothing
        image_product = src_freq .* target_freq;
    else
        image_product = src_freq .* target_freq .* filter;
    end
    cross_correlation = ifft(image_product);

    shifts = collect((aamax(abs2.(cross_correlation))).I) .- 1
    midpoints = size(src_freq) .÷ 2
    shifts[shifts .> midpoints] .-= size(src_freq)[shifts .> midpoints];

    if upsample_factor > 1
        upsampled_region_size = ceil(Int, upsample_factor * 1.5)
        dftshift  = upsampled_region_size ÷ 2
        sample_region_offset = dftshift .- shifts * upsample_factor
        cross_correlation = conj_upsampled_dft(image_product,
                                    upsampled_region_size,
                                        upsample_factor,
                                    sample_region_offset)
        shifts_up = aamax(abs2.(cross_correlation)).I .- 1
        shifts = shifts .+ (shifts_up .- dftshift) ./ upsample_factor
    end
        
    return shifts
end

