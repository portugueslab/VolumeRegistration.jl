function sparsify(a, sparse_threshold = 1e-3)
    a[a.<sparse_threshold] .= 0
    sp = sparse(a)
    dropzeros!(sp)
    return sp
end

function block_distance_kernel(b::Blocks)
    distances = [
        exp(-sum((idx_a.I .- idx_b.I) .^ 2))
        for
        (idx_a, idx_b) in Iterators.product(
            CartesianIndices(b.blocks_per_dim)[:],
            CartesianIndices(b.blocks_per_dim)[:],
        )
    ]
    return sparsify(permutedims(distances ./ sum(distances, dims = 2)))
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

function shifts_to_extrapolation(
    shifts::AbstractArray{NTuple{N,T},N},
    blks,
) where {N,T<:Real}
    axes_interp = Tuple(
        ((0:nb-1) * (bs) .+ (bs + pd) / 2)
        for (nb, bs, pd) in zip(blks.blocks_per_dim, blks.block_size, blks.padding)
    )

    shifts_dim =
        LinearInterpolation(axes_interp, SVector.(shifts), extrapolation_bc = Line())

    return shifts_dim
end

struct PrecompuationDeformationMap
    blocks::Any
    blockwise_reference::Any
    blocked_masks::Any
    blocked_offsets::Any
    block_fft_plan::Any
    upsampler::Any
    window_size::Any
    block_correlation_blur::Any
end

"""
Prepare everything common for calculating deformation maps

"""
function prepare_find_deformation_map(
    reference::AbstractArray{T,N};
    block_size = N == 2 ? (128, 128) : (128, 128, 5),
    block_border_σ = 1f0,
    max_shift = 3,
    border_σ = nothing,
    σ_filter = nothing,
    upsampling = N == 2 ? (10, 10) : (8, 8, 4),
    upsample_padding = N == 2 ? (3, 3) : (3, 3, 2),
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
    prepare_fft_reference!.(blockwise_reference, Ref(σ_filter))

    blocked_masks, blocked_offsets =
        block_masks_offsets(reference, mask, blocks, block_border_σ)

    # prepare the upsampling and fft computations
    block_fft_plan = plan_fft(blockwise_reference[1])
    us = KrigingUpsampler(upsampling = upsampling, padding = upsample_padding)

    window_size = to_ntuple(Val{N}(), max_shift) .* 2 .+ 1

    block_correlation_blur = block_distance_kernel(blocks)
    return PrecompuationDeformationMap(
        blocks,
        blockwise_reference,
        blocked_masks,
        blocked_offsets,
        block_fft_plan,
        us,
        window_size,
        block_correlation_blur,
    )
end

function calc_block_offsets(
    moving::AbstractArray{T,N},
    pc::PrecompuationDeformationMap;
    snr_n_smooths = 2,
    snr_threshold = 1.15f0,
    snr_n_pad = N == 2 ? (3, 3) : (3, 3, 1),
    interpolate_middle=false,
) where {T,N}

    blockwise_moving = Slices(split_into_blocks(moving, pc.blocks), (1:N)...)

    # phase correlations and signal to noise ratios
    block_correlations =
        extract_low_frequencies.(
            phase_correlation.(
                Ref(pc.block_fft_plan) .*
                apply_mask.(
                    blockwise_moving,
                    Slices(pc.blocked_masks, (1:N)...),
                    Slices(pc.blocked_offsets, (1:N)...),
                ),
                pc.blockwise_reference,
            ),
            Ref(pc.window_size),
            interpolate_middle
        )

    # blur phase correlations weighted by SNR
    blurred_correlations = [block_correlations]
    selected_correlations = deepcopy(block_correlations)

    for i in 1:snr_n_smooths
        push!(
            blurred_correlations,
            [
                sum(
                    blurred_correlations[end][i] * weight
                    for (i, weight) in zip(findnz(pc.block_correlation_blur[:, i_vec])...)
                ) for i_vec in 1:length(blurred_correlations[end])
            ],
        )
    end
    for i_block in 1:length(selected_correlations)
        i_blur = 0
        snr_current = 1f0
        while snr_current < snr_threshold && i_blur < snr_n_smooths
            snr_current = calc_snr(blurred_correlations[i_blur+1][i_block], snr_n_pad)
            i_blur += 1
        end
        if i_blur != 1
            selected_correlations[i_block] .= blurred_correlations[i_blur][i_block]
        end
    end

    return reshape(
        getindex.(shift_around_maximum.(Ref(pc.upsampler), selected_correlations), 1),
        pc.blocks.blocks_per_dim...,
    )
end

"""
Find deformation maps by splitting the dataset in blocks
and aligning blocks with subpixel precision

# Arguments

- `moving`: the stack to be registered
- `reference`: the stack to be registered to
- `border_σ`: how far to fade out the borders of the whole image/volume
- `block_size::NTuple{N, Integer}`: size of blocks to compute deformations
- `block_border_σ::Union{Real, NTuple{N, Real}}`: how far to fade out the
- `max_shift::Union{Real, NTuple{N, Integer}}`: maximum displacement of a block in each dimension
- `σ_filter`: low-pass filter width
- `upsampling`: upsampling of the registration for subpixel alignment
- `upsample_padding`: how far to pad from the local maximum for upsampling
- `snr_n_smooths::Integer=2`: number of times the correlation matrices are smoothed by wieghting with adjacent ones (if they are under signal-to-noise ratio)
- `snr_threshold::Real`: the threshold of the "peakiness" of the correlation matrix, if it's smaller than that, it's value is obtained by smoothing neighbors
- `snr_n_pad::Integer`: window size of the signal-to-noise calculation
"""
function find_deformation_map(
    moving::AbstractArray{T,N},
    reference::AbstractArray{T,N};
    snr_n_smooths = 2,
    snr_threshold = 1.15f0,
    snr_n_pad = N == 2 ? (3, 3) : (3, 3, 1),
    interpolate_middle=false,
    kwargs...,
) where {T,N}
    pc = prepare_find_deformation_map(reference; kwargs...)
    return (
        shifts = calc_block_offsets(
            moving,
            pc;
            snr_n_smooths = snr_n_smooths,
            snr_threshold = snr_threshold,
            snr_n_pad = snr_n_pad,
            interpolate_middle = interpolate_middle
        ),
        blocks = pc.blocks,
    )
end

# variant to align stack
function find_deformation_map(
    moving::AbstractArray{T,M},
    reference::AbstractArray{T,N};
    snr_n_smooths = 2,
    snr_threshold = 1.15f0,
    snr_n_pad = N == 2 ? (3, 3) : (3, 3, 1),
    interpolate_middle=false,
    kwargs...,
) where {T,N,M}
    pc = prepare_find_deformation_map(reference; kwargs...)

    return (
        shifts = tmap(mov ->
            calc_block_offsets(
                mov,
                pc;
                snr_n_smooths = snr_n_smooths,
                snr_threshold = snr_threshold,
                snr_n_pad = snr_n_pad,
                interpolate_middle = interpolate_middle
            ), eachslice(moving, dims = M)
        ),
        blocks = pc.blocks,
    )
end
