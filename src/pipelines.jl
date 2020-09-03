"""
    $(SIGNATURES)

Register volumes, by first finding a rigid translationa and then optionally applying piecewise affine shifts.

# Arguments
- `destination`: the destination for the registered dataset, usually a Float32 subtype of a DiskArray
- `dataset`: the stack to be registered
- `reference`: the static volume regerence
- `deform`: whether to additionally perform non-rigid registration
- `output_time_first`: whether to reorder the array to have time-first or time-last order

The arguments are from [`find_translation`](@ref) for the pure translation, and from [`find_deformation_map`](@ref) for the
piecewise affine transfomration (with the suffix `_deform`). 

"""
function register_volumes!(
    destination,
    dataset,
    reference;
    max_shift = (20, 20, 2),
    σ_filter = (3.0f0, 3.0f0, 0.1f0),
    border_σ = (10f0, 10f0, 0.1f0),
    interpolate_middle = true,
    upsampling = 1,
    upsample_padding = nothing,
    deform = true,
    max_shift_deform = (5, 5, 2),
    σ_filter_deform = 1.3f0,
    border_σ_deform = (5f0, 5f0, 1f0),
    block_border_σ = (5f0, 5f0, 0.5f0),
    block_size = (256, 256, 4),
    upsampling_deform = (10, 10, 4),
    snr_n_smooths = 2,
    snr_threshold = 1.15f0,
    interpolate_middle_deform = false,
    t_block_size,
    output_time_first = true,
    interpolation_deform = Linear(),
)

    @info "Precomputing translation"
    float_reference = Float32.(reference)

    prepared_translation = prepare_find_translation(
        float_reference;
        max_shift = max_shift,
        σ_filter = σ_filter,
        border_σ = border_σ,
        upsampling = upsampling,
        upsample_padding = upsample_padding,
    )

    @info "Precomputing nonrigid transformation"

    if deform
        prepared_nonrigid = prepare_find_deformation_map(
            float_reference;
            block_size = block_size,
            max_shift = max_shift_deform,
            upsampling = upsampling_deform,
            σ_filter = σ_filter_deform,
            border_σ = border_σ_deform,
            block_border_σ = block_border_σ,
        )
    end

    n_t = size(dataset, 4)

    if upsampling == 1
        shifts = Array{Translation{SVector{3,Int64}}}(undef, n_t)
    else
        shifts = Array{Translation{SVector{3,Float64}}}(undef, n_t)
    end
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
                interpolate_middle,

            )
            idx = first(frame_range) + i_t - 1
            shifts[idx] = shift
            correlations[idx] = correlation
        end

        translated = translate(moving_slices, shifts[frame_range])

        if deform
            block_shifts = tmap(
                mov -> calc_block_offsets(
                    mov,
                    prepared_nonrigid;
                    snr_n_smooths = snr_n_smooths,
                    snr_threshold = snr_threshold,
                    snr_n_pad = (3, 3, 1),
                    interpolate_middle = interpolate_middle_deform,
                ),
                eachslice(translated, dims = 4),
            )

            undeformed = apply_deformation_map(
                translated,
                block_shifts,
                prepared_nonrigid.blocks,
                spline_type = interpolation_deform,
            )
        else
            undeformed = translated
        end

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

Make a plane-by-plane reference for 2-photon imaging data,
by first finding a reference image for each plane, and then finding inter-plane shifts
    and correcting them.
The keyword arguments are from [`make_reference`](@ref)

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
            destination[:, :, :, i_plane] .= reshape(permutedims(translated, (3, 1, 2)), size(translated)[[3, 1, 2]]...)
        else
            destination[:, :, i_plane, :] .= reshape(translated, size(translated)[1:2]..., size(translated, 3))
        end
    end
    return (
        plane_shifts = plane_translations,
        plane_correlations = plane_correlations,
    )
end
