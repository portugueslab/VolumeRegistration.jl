function gaussian_border_mask(T::Type, dimensions::NTuple{N, Integer}, σ) where {N}
    mask = ones(T, dimensions)
    mask_indices = CartesianIndices(mask)
    last_idx = last(mask_indices).I
    mid_idx = (last(mask_indices).I .+ first(mask_indices).I) ./ 2
    correction = mid_idx .- (2 .* σ)
    for idx in mask_indices
        mask[idx] = prod((1 ./ (1 .+ exp.((abs.(idx.I .- mid_idx) .- correction) ./ σ))))
    end
    return mask
end

function gaussian_border_mask(dimensions::NTuple{N, Integer}, σ) where {N}
    return gaussian_border_mask(Float32, dimensions, σ)
end

function mask_offset(image, mask)
    return mean(image) .* (1 .- mask)
end

function spatial_smooth(a::AbstractArray{T,N}, n_smooth::NTuple{N, Integer}) where {T, N}
    smoothing_array = zeros(T, size(a) .+ (n_smooth .÷ 2))
    smoothing_array[CartesianIndex(1 .+ n_smooth .÷ 2):CartesianIndex(n_smooth .÷ 2 .+ size(a))] .= a
    for (i_dim, n_dim_smooth) in enumerate(n_smooth)
        if n_dim_smooth == 0
            continue
        end
        smoothing_array[]
    end
end