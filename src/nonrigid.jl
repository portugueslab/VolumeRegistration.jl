function block_distance_kernel(b::Blocks)
    distances = [
        exp(-sum((idx_a.I .- idx_b.I) .^ 2))
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
function split_into_blocks(
    data::AbstractArray{T,N},
    blocks::Blocks{N},
    output_type = Complex{T},
) where {T,N}
    blocked_data =
        zeros(output_type, (blocks.block_size .+ blocks.padding)..., length(blocks))
    for (i, sl) in enumerate(slices(blocks))
        blocked_data[(s .- s.start .+ 1 for s in sl)..., i] .= data[sl...]
    end
    return blocked_data
end

function default_block_size(N)
    return N == 2 ? (128, 128) : (128, 128, 4)
end

function block_masks_offsets(
    reference::AbstractArray{T,N},
    mask,
    blocks,
    block_border_σ,
) where {T,N}
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

    blocked_offsets =
        (1 .- blocked_masks) .* reshape(reference_slice_means, nones(Val{N}())..., :)

    return blocked_masks, blocked_offsets
end

function apply_mask(a, mask, mask_offset)
    return a .* mask .+ mask_offset
end

"""
Computes the signal to noise ratio of a phase correlation
array, defined as the ratio of the maximum phase correlation
in the window divided by the maximal phase correlation
outside of the region where the maximum is (the idea 
being that it will be low if the phase correlation is flat
versus peaked)
"""
function calc_snr(cc::AbstractArray{T,N}, n_pad) where {T,N}
    pad_tuple = to_ntuple(Val{N}(), n_pad)
    max_val, max_ind =
        findmax(cc[CartesianIndex(1 .+ pad_tuple):CartesianIndex(size(cc) .- pad_tuple)])
    cc_masked = copy(cc)
    cc_masked[max_ind:(max_ind+CartesianIndex(2 .* pad_tuple))] .= 0
    max_noise_val = max(maximum(cc_masked), 0)
    return max_val / max_noise_val
end

function find_deformation_map(
    moving::AbstractArray{T,N},
    reference::AbstractArray{T,N};
    block_size = default_block_size(N),
    block_border_σ,
    max_shift=15,
    border_σ = nothing,
    σ_filter = nothing,
    snr_n_interpolations = 2,
    snr_threshold = 1.15,
    snr_n_pad = 3,
) where {T,N}

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

    blocked_masks, blocked_offsets =
        block_masks_offsets(reference, mask, blocks, block_border_σ)

    blocked_moving = Slices(split_into_blocks(moving, blocks), (1:N)...)

    block_fft_plan = plan_fft(blocked_moving[1])

    window_size = to_ntuple(Val{N}(), max_shift) .* 2 .+ 1

    # phase correlations and signal to noise ratios
    block_correlations =
        extract_low_frequencies.(phase_correlation.(
            Ref(block_fft_plan) .*
            apply_mask.(
                blocked_moving,
                Slices(blocked_masks, (1:N)...),
                Slices(blocked_offsets, (1:N)...),
            ),
            blockwise_reference,
        ), Ref(window_size))

    block_correlation_blur = block_distance_kernel(blocks)
    
    # blur phase correlations weighted by SNR
    blurred_correlations = [block_correlations]
    selected_correlations = deepcopy(block_correlations)
    for i in 1:snr_n_interpolations
        push!(blurred_correlations, block_correlation_blur * blurred_correlations[end])
    end
    for i_block in 1:length(selected_correlations)
        i_blur = 0
        snr_current = 1f0
        while snr_current < snr_threshold && i_blur < snr_n_interpolations
            snr_current = calc_snr(blurred_correlations[i_blur+1][i_block], snr_n_pad)
            i_blur += 1
        end
        if i_blur != 1
            selected_correlations[i_block] .= blurred_correlations[i_blur][i_block]
        end
    end
    
    us = KrigingUpsampler()

    return reshape(shift_around_maximum.(Ref(us), selected_correlations), blocks.blocks_per_dim...)
end
