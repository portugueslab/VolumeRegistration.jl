using OMEinsum
using FFTW
using CuArrays

function conj_upsampled_dft(data::AbstractArray{T,N}, upsampled_region_size,
    upsample_factor = 1, axis_offsets = zeros(Int32, N)) where {T,N}
    im2π = 1im * 2 * π
    dim_kernels = []
    for (n_items, ax_offset) in zip(size(data), axis_offsets)
        kernel_a = (im2π / (n_items * upsample_factor)) .* (-ax_offset:(upsampled_region_size - ax_offset - 1))
        kernel_b = ifftshift(0:n_items - 1) .- (n_items ÷ 2)
        push!(dim_kernels, exp.(kernel_a * reshape(kernel_b, 1, n_items)))
    end
    
    if N == 2
        res = ein"ij,li,mj->lm"(data, dim_kernels...)
    elseif N == 3
        res = ein"ijk,li,mj,nk->lmn"(data, dim_kernels...)
    end
    return res
end

function conj_upsampled_dft(data::CuArray{T,N}, upsampled_region_size,
    upsample_factor = 1, axis_offsets = zeros(Int32, N)) where {T,N}
    im2π = 1im * 2 * π
    dim_kernels = []
    for (n_items, ax_offset) in zip(size(data), axis_offsets)
        kernel_a = cu(collect(im2π / (n_items * upsample_factor)) .*
                     (-ax_offset:(upsampled_region_size - ax_offset - 1)))
        kernel_b = cu(ifftshift(0:n_items - 1)) .- (n_items ÷ 2)
        push!(dim_kernels, exp.(kernel_a * reshape(kernel_b, 1, n_items)))
    end
    
    if N == 2
        res = ein"(ij,li),mj->lm"(data, dim_kernels...)
    elseif N == 3
        res = ein"((ijk,li),mj),nk->lmn"(data, dim_kernels...)
    end
    return res
end

function aamax(x::AbstractArray)
    return argmax(x)
end

function aamax(x::CuArray)
    i_m = argmax(x)
    indices = CartesianIndices(size(x))
    return indices[i_m]
end

function register_translation(src_img::AbstractArray{T,N}, target_img::AbstractArray{T,N};  kwargs...) where {T,N}
    return register_translation(fft(src_img), fft(target_img), kwargs...) 
end

function register_translation(src_img::AbstractArray{T,N}, target_img::AbstractArray{T2,N};  kwargs...) where {T, T2 <:Complex, N}
    return register_translation(fft(src_img), target_img, kwargs...) 
end

function register_translation(src_freq::AbstractArray{T, N}, target_freq::AbstractArray{T, N}; upsample_factor = 1, filter::Union{Nothing, AbstractArray{T, N}}=nothing) where {T <: Complex, N}
    if filter === nothing
        image_product = src_freq .* conj.(target_freq);
    else
        image_product = src_freq .* conj.(target_freq) .* filter;
    end
    cross_correlation = FFTW.ifft(image_product);

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