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
    blocked_data = zeros(output_type, blocks.block_size..., length(blocks))
    for (i, sl) in enumerate(slices(blocks))
        blocked_data[(s .- s.start .+ 1 for s in sl)..., i] .= data[sl...]
    end
    return blocked_data
end

function default_block_size(N)
    return N == 2 ? (128, 128) : (128, 128, 4)
end

function block_masks_offsets(reference::AbstractArray{T, N}, mask, blocks, block_border_σ) where {T, N}
    block_mask = gaussian_border_mask(blocks.block_size, to_ntuple(N, block_border_σ))
    blocked_masks = ones(T, blocks.block_size..., length(blocks))
    blocked_offsets = zeros(T, blocks.block_size..., length(blocks))
    if mask != 1
        for (i, sl) in enumerate(slices(blocks))
            blocked_masks[(s .- s.start .+ 1 for s in sl)..., i] .= mask[sl...]
        end
    end
    blocked_masks .*= reshape(block_mask, 1, size(block_mask)...)
    # TODO efficient offset calculation 
    for (i, sl) in enumerate(slices(blocks))
        blocked_offsets[(Colon() for _ in 1:N)..., i] .= mean(reference[sl...]) .* (1-blocked_masks)
    end
    
    return blocked_masks, blocked_offsets
end

function find_deformation_map(moving::AbstractArray{T, N}, reference::AbstractArray{T, N};
     block_size=default_block_size(N), block_border_σ, border_σ, σ_filter=nothing) where{T, N}
    blocks = Blocks(size(reference), block_size ÷ 2, padding = block_size ÷ 2)
    if border_σ
        mask = 1
    else
        mask = gaussian_border_mask(size(reference), border_σ)
    end
    split_reference = split_into_blocks(reference, blocks)
    blockwise_reference = Slices(split_reference, 1, 2)
    prepare_fft_reference!.(blockwise_reference, σ_filter)

    blocked_masks, blocked_offsets = block_masks_offsets(reference, mask, blocks, block_border_σ)

    # phase correlations and signal to noise ratios

    # blur phase correlations weighted by SNR

    # find interpolated displacements

end
