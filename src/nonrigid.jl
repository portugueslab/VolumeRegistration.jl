function block_distance_kernel(b::Blocks)
    distances = [
        exp(-sum((idx_a .- idx_b) .^ 2))
        for
        (idx_a, idx_b) in Iterators.product(
            CartesianIndices(b.blocks_per_dim)[:],
            CartesianIndices(b.blocks_per_dim)[:],
        )
    ]
    return distances ./ sum(distances, dims = 2)
end


"""
Splits data into an array of blocks, augmenting the last blocks which may not be full
to keep consistent size. Cast immediately into complex for further processing

"""
function split_into_blocks(data::AbstractArray{T, N}, blocks::Blocks{N}, output_type=Complex{T}) where{T, N}
    blocked_data = zeros(output_type, (blocks.block_size .+ blocks.padding)..., length(blocks))
    for (i, sl) in enumerate(slices(blocks))
        blocked_data[(s .- s.start .+ 1 for s in sl)..., i] .= data[sl...]
    end
    return blocked_data
end

function default_block_size(N)
    return N == 2 ? (128, 128) : (128, 128, 4)
end

function block_masks_offsets(reference::AbstractArray{T, N}, mask, blocks, block_border_σ) where {T, N}
    full_block_size = blocks.block_size .+ blocks.padding
    block_mask = gaussian_border_mask(full_block_size, to_ntuple(Val{N}(), block_border_σ))
    blocked_masks = ones(T, full_block_size..., length(blocks))
    blocked_offsets = zeros(T, full_block_size..., length(blocks))
    if mask != 1
        for (i, sl) in enumerate(slices(blocks))
            blocked_masks[(s .- s.start .+ 1 for s in sl)..., i] .= mask[sl...]
        end
    end

    blocked_masks .*= reshape(block_mask, size(block_mask)..., 1)
    
    reference_slice_means = [mean(reference[sl...]) for sl in slices(blocks)]

    blocked_offsets = (1 .- blocked_masks) .* reshape(reference_slice_means, nones(Val{N}())..., :) 
    
    return blocked_masks, blocked_offsets
end

function apply_mask(a, mask, mask_offset)
    return a .* mask .+ mask_offset
end

function find_deformation_map(moving::AbstractArray{T, N}, reference::AbstractArray{T, N};
     block_size=default_block_size(N), block_border_σ, border_σ=nothing, σ_filter=nothing) where{T, N}

    # prepare blocks and split the reference into the blocks
    blocks = Blocks(size(reference), block_size .÷ 2, padding = block_size .÷ 2)
    if border_σ === nothing
        mask = 1
    else
        mask = gaussian_border_mask(size(reference), border_σ)
    end
    split_reference = split_into_blocks(reference, blocks)
    blockwise_reference = Slices(split_reference, (1:N)...)
    prepare_fft_reference!.(blockwise_reference, σ_filter)

    blocked_masks, blocked_offsets = block_masks_offsets(reference, mask, blocks, block_border_σ)

    blocked_moving = Slices(split_into_blocks(moving, blocks), (1:N)...)

    block_fft_plan = plan_fft(blocked_moving[1])

    # phase correlations and signal to noise ratios
    block_correlations = phase_correlation.(Ref(block_fft_plan) .* apply_mask.(blocked_moving,
     Slices(blocked_masks, (1:N)...), Slices(blocked_offsets, (1:N)...)), blockwise_reference)

    # blur phase correlations weighted by SNR

    # find interpolated displacements

    return block_correlations
end
