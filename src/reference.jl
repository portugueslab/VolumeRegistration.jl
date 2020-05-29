function initial_reference(
    ds;
    time_range = Colon(),
    corr_win = (30, 30, 3),
    n_take_mean = 20,
)
    ds_mid = size(ds)[1:3] .÷ 2
    ds_timeseries =
        ds[((s:e) for (s, e) in zip(ds_mid .- corr_win, ds_mid .+ corr_win))..., time_range]
    ds_timeseries = Float32.(reshape(ds_timeseries, :, size(ds_timeseries)[end]))
    dev = ds_timeseries .- mean(ds_timeseries, dims = 2)
    corr = transpose(dev) * dev
    amp = sqrt.(corr[diagind(corr)])
    corr ./= amp * permutedims(amp)
    mean_corr = mean(corr, dims = 2)

    # find the frame the most correlated 
    max_corr, max_i_frame = findmax(mean_corr[:][1:end])
    max_i_frame += 1
    most_correlated = sortperm(corr[max_i_frame, :], rev = true)
    reference_indices = most_correlated[1:n_take_mean]

    # mean
    mn = zeros(Float32, size(ds)[1:3])
    for i_fr in reference_indices
        mn .+= ds[:, :, :, i_fr]
    end

    return (stack = mn ./ n_take_mean, indices = reference_indices .+ first(time_range))
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
    translation_kwargs...
) where {T,N}
    average_stack = zeros(Float32, (size(reference)..., n_average))
    for i_iteration in 1:n_iterations
        translations, correlations = find_translation(frames, reference, translation_kwargs...)
        best_corr_order = sortperm(correlations, rev = true)[1:n_average]
        foreach(
            translate!,
            eachslice(average_stack, dims = N + 1),
            eachslice(frames[:, :, :, best_corr_order], dims = N + 1),
            translations[best_corr_order],
        )
        reference = mean(average_stack, dims = N + 1)[:, :, :, 1]
    end
    return reference
end

"""
    make_reference(stack; kwargs...)

Make a reference for a stack

# Arguments
- `stack`: the data to be aligned, time is assumed to be the list dimension
- `ref_win`

"""
function make_reference(
    stack;
    time_range = Colon(),
    corr_win = (30, 30, 3),
    n_refine_from = 400,
    n_average = 20,
    n_average_refine = 50,
    n_iterations = 3,
    translation_kwargs...
)
    N = ndims(stack) - 1
    ref_init_stack, ref_init_frames = initial_reference(
        stack;
        corr_win = corr_win,
        time_range = time_range,
        n_take_mean = n_average,
    )
    reference_mid = round(Int, median(ref_init_frames))
    reference_mid =
        min(max(reference_mid, n_refine_from ÷ 2), size(stack, N + 1) - n_refine_from ÷ 2)

    if n_iterations > 0
        return refine_reference(
            ref_init_stack,
            stack[
                (Colon() for _ in 1:N)...,
                reference_mid-n_refine_from÷2+1:reference_mid+n_refine_from÷2,
            ],
            n_average = n_average_refine,
            n_iterations = n_iterations,
        )
    else
        return ref_init_stack
    end
end
