function initial_reference(
    ds;
    time_range = Colon(),
    corr_win = (30, 30, 3),
    n_take_mean = 20,
)
    N = ndims(ds) - 1
    ds_mid = size(ds)[1:N] .÷ 2
    ds_timeseries =
        ds[((s:e) for (s, e) in zip(ds_mid .- corr_win, ds_mid .+ corr_win))..., time_range]
    ds_timeseries = Float32.(reshape(ds_timeseries, :, size(ds_timeseries)[end]))
    dev = ds_timeseries .- mean(ds_timeseries, dims = 2)
    corr = transpose(dev) * dev
    amp = sqrt.(corr[diagind(corr)])
    corr ./= amp * permutedims(amp)
    mean_corr = mean(corr, dims = 2)

    # find the frame the most correlated 
    max_corr, max_i_frame = findmax(mean_corr[:])
    most_correlated = sortperm(corr[max_i_frame, :], rev = true)
    reference_indices = most_correlated[1:n_take_mean]

    # mean
    mn = zeros(Float32, size(ds)[1:N])
    for i_fr in reference_indices
        mn .+= ds[(Colon() for _ in 1:N)..., i_fr]
    end

    t0 = time_range == Colon() ? 0 : first(time_range) - 1
    return (stack = mn ./ n_take_mean, indices = reference_indices .+ t0)
end

"""
Iterative refinement of a reference image by aligning more frames and then
making a new reference out of the most correlated, aligned frames

"""
function refine_reference(
    reference::AbstractArray{T,N},
    frames;
    n_average = min(size(frames)[end], 100),
    n_iterations = 3,
    translation_kwargs...,
) where {T,N}
    average_stack = zeros(Float32, (size(reference)..., n_average))
    for i_iteration in 1:n_iterations
        @info("Refining reference at iteration $(i_iteration) of $(n_iterations)")
        translations, correlations =
            find_translation(frames, reference, translation_kwargs...)
        best_corr_order = sortperm(correlations, rev = true)[1:n_average]
        foreach(
            translate!,
            eachslice(average_stack, dims = N + 1),
            eachslice(frames[(Colon() for _ in 1:N)..., best_corr_order], dims = N + 1),
            translations[best_corr_order],
        )
        reference = mean(average_stack, dims = N + 1)[(Colon() for _ in 1:N)..., 1]
    end
    return reference
end

"""
    make_reference(stack; kwargs...)

Make a reference for a stack

# Arguments
- `stack`: the data to be aligned, time is assumed to be the list dimension
- `time_range::AbstractUnitRange=Colon()`: the range of time to take the reference from
- `corr_win`: size of the window around the middle of the stack to calculate the optimal reference
- `n_refine_from`: how many frames to make the refined reference from
- `n_average`: number of frames to average for a nice reference
- `n_iterations`: how many times to recalculate the reference from the best correlated moved frames

more keyword arguments for (@ref find_translation) can be supplied

"""
function make_reference(
    stack;
    time_range = Colon(),
    corr_win = (30, 30, 3),
    n_refine_from = 200,
    n_average = 20,
    n_average_refine = 50,
    n_iterations = 3,
    translation_kwargs...,
)
    N = ndims(stack) - 1
    @info("Finding best inital frame for the reference")
    ref_init_stack, ref_init_frames = initial_reference(
        stack;
        corr_win = corr_win,
        time_range = time_range,
        n_take_mean = n_average,
    )
    reference_mid = round(Int, median(ref_init_frames))
    reference_mid = min(
        max(reference_mid, n_refine_from ÷ 2),
        size(stack, N + 1) - n_refine_from ÷ 2 - 1,
    )
    stack_range =
        max(
            reference_mid - n_refine_from ÷ 2 + 1,
            1,
        ):min(reference_mid + n_refine_from ÷ 2, size(stack, N + 1))

    return if n_iterations > 0
        return refine_reference(
            ref_init_stack,
            stack[(Colon() for _ in 1:N)..., stack_range],
            n_average = min(n_average_refine, length(stack_range)),
            n_iterations = n_iterations,
        )
    else
        return ref_init_stack
    end
end
