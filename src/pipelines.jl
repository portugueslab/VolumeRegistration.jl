"""
    $(SIGNATURES)

Register volumes, by first finding a rigid translationa and then applying non-rigid shifts

"""
function register_volumes!(
    destination,
    dataset,
    reference;
    max_shift_rigid = (20, 20, 2),
    σ_filter_rigid = (3.0f0, 3.0f0, 0.1f0),
    border_σ_rigid = (10f0, 10f0, 0.1f0),
    interpolate_middle_rigid = true,
    max_shift_nonrigid = (5, 5, 2),
    σ_filter_nonrigid = 1.3f0,
    border_σ_nonrigid = (5f0, 5f0, 1f0),
    block_border_σ = (5f0, 5f0, 0.5f0),
    block_size = (256, 256, 4),
    upsampling = (10, 10, 4),
    snr_n_smooths = 2,
    snr_threshold = 1.15f0,
    interpolate_middle_nonrigid = false,
    t_block_size,
    output_time_first = true,
    interpolation_nonrigid = Linear(),
)

    @info "Precomputing translation"
    float_reference = Float32.(reference)

    prepared_translation = prepare_find_translation(
        float_reference;
        max_shift = max_shift_rigid,
        σ_filter = σ_filter_rigid,
        border_σ = border_σ_rigid,
    )

    @info "Precomputing nonrigid transformation"

    prepared_nonrigid = prepare_find_deformation_map(
        float_reference;
        block_size = block_size,
        max_shift = max_shift_nonrigid,
        upsampling = upsampling,
        σ_filter = σ_filter_nonrigid,
        border_σ = border_σ_nonrigid,
        block_border_σ = block_border_σ,
    )
    n_t = size(dataset, 4)

    shifts = Array{Translation{SVector{3,Int64}}}(undef, n_t)
    correlations = Array{Float32}(undef, n_t)

    @info "Correcting blockwise"

    @showprogress for frame_range in Iterators.partition(1:n_t, t_block_size)
        source_slice = dataset[:, :, :, frame_range]

        n_tsub = length(frame_range)

        moving_slices = @strided(Float32.(source_slice))

        Base.Threads.@threads for i_t in 1:n_tsub
            shift, correlation = prepared_find_translation(
                moving_slices[:, :, :, i_t],
                prepared_translation...,
                interpolate_middle_rigid,
            )
            idx = first(frame_range) + i_t - 1
            shifts[idx] = shift
            correlations[idx] = correlation
        end

        translated = translate(moving_slices, shifts[frame_range])

        block_shifts = tmap(
            mov -> calc_block_offsets(
                mov,
                prepared_nonrigid;
                snr_n_smooths = snr_n_smooths,
                snr_threshold = snr_threshold,
                snr_n_pad = (3, 3, 1),
                interpolate_middle = interpolate_middle_nonrigid,
            ),
            eachslice(translated, dims = 4),
        )

        undeformed = apply_deformation_map(
            translated,
            block_shifts,
            prepared_nonrigid.blocks,
            spline_type = interpolation_nonrigid,
        )

        if output_time_first
            destination[frame_range, :, :, :] = @strided(permutedims(
                round.(UInt16, clamp.(undeformed, 0, typemax(UInt16))),
                (4, 1, 2, 3),
            ))
        else
            destination[:, :, :, frame_range] =
                @strided( round.(UInt16, clamp.(undeformed, 0, typemax(UInt16))))
        end

    end
    return shifts, correlations
end

function find_plane_shifts(reference, select_area = nothing; kwargs...)
    n_planes = size(reference, 3)
    i_mid = n_planes ÷ 2

    ranges = [i_mid+1:n_planes, i_mid-1:-1:1]
    deltas = [-1, 1]

    # if only a middle part is used for alignement, the slice is constructed
    if select_area === nothing
        selection = (Colon(), Colon())
    else
        selection = select_area
    end
    shifts = fill((0, 0), n_planes)
    for (plane_range, dy) in zip(ranges, deltas)
        previous_shift = (0, 0)
        for i in plane_range
            relative_shift, correlation =
                find_translation(reference[:, :, i], reference[:, :, i+dy]; kwargs...)
            previous_shift = Tuple(relative_shift.translation) .+ previous_shift
            shifts[i] = previous_shift
        end
    end
    return shifts
end

"""
    $(SIGNATURES)

Make a plane-by-plane reference for 2-photon imaging data. 
The keyword arguments are from [`find_translation`](@ref)

"""
function make_planewise_reference(stack; reference_kw...)
    @info "Making initial reference"
    n_x, n_y, n_planes, n_t = size(stack)

    # make initial reference
    initial_reference = Array{Float32}(undef, (n_x, n_y, n_planes))
    # disable all the messages when finding references for each plane
    with_logger(SimpleLogger(stderr, Logging.Warn)) do
        @showprogress for i_plane in 1:n_planes
            initial_reference[:, :, i_plane] = make_reference(stack[:, :, i_plane, :]; reference_kw...)
        end
    end
    @info "Finding shifts to align reference"
    # align reference so there are minimal shifts between planes
    interplane_shifts = find_plane_shifts(initial_reference)

    @info "Aligning reference"
    reference = reduce(
        (x, y) -> cat(x, y, dims = 3),
        translate.(eachslice(initial_reference, dims = 3), Translation.(interplane_shifts)),
    )
    return  (reference = reference,
             interplane_shifts = interplane_shifts)
end


"""
    $(SIGNATURES)

Registers a stack of data acquired plane-by-plane
The keyword arguments are from [`find_translation`](@ref)

"""
function register_planewise!(
    destination,
    stack,
    reference;
    output_time_first = true,
    translation_kw...,
)
    n_x, n_y, n_planes, n_t = size(stack)

    @info "Aligning stack"
    plane_translations = []
    plane_correlations = []
    @showprogress for i_plane in 1:n_planes
        current_plane = stack[:, :, i_plane, :]
        translations, correlations =
            find_translation(current_plane, reference[:, :, i_plane]; translation_kw...)
        push!(plane_translations, translations)
        push!(plane_correlations, correlations)

        translated = translate(current_plane, translations)

        if output_time_first
            destination[:, :, :, i_plane] .= permutedims(translated, (3, 1, 2))
        else
            destination[:, :, i_plane, :] .= translated
        end
    end
    return (
        plane_shiffts = plane_translations,
        plane_correlations = plane_correlations,
    )
end
