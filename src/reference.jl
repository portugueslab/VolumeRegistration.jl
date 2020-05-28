function find_reference(ds; time_range=Colon(), ref_wind = (30, 30, 3), n_skip_first = 10, n_take_mean=20)
    ds_mid = size(ds)[1:3] .÷2
    ds_timeseries = ds[((s:e) for (s, e) in zip(ds_mid .- ref_wind, ds_mid .+ ref_wind))..., time_range]
    ds_timeseries = Float32.(reshape(ds_timeseries, :, size(ds_timeseries)[end]))
    dev = ds_timeseries .- mean(ds_timeseries, dims=2)
    corr = transpose(dev) * dev
    amp = sqrt.(corr[diagind(corr)])
    corr ./= amp * permutedims(amp)
    mean_corr = mean(corr, dims=2);

    max_corr, max_i_frame = findmax(mean_corr[:][n_skip_first+1:end])
    max_i_frame += n_skip_first
    most_correlated = sortperm(-corr[max_i_frame, :])
    reference_indices = most_correlated[1:n_take_mean]
    
    # mean
    mn = zeros(Float32, size(ds)[1:3])
    for i_fr in reference_indices
        mn .+= ds[:,:,:,i_fr]
    end
    return (stack=mn ./ n_take_mean, indices=reference_indices .+ first(time_range))
end


"""
Iterative refinement of a reference image by aligning more frames and then
making a new reference out of the most correlated aligned frames

"""
function refine_reference(reference::AbstractArray{T, N}, frames; n_average=min(size(frames)[end], 100), n_iterations=8) where {T, N}
    for i_iteration in 1:n_iterations
        translations, correlations = find_translation(reference, frames)
        best_corr_order = sortperm(-correlations)[1:n_average]
        average_stack = zeros(Float32, (size(reference)..., n_average))
        foreach(translate!, mapslices(average_stack, N+1),
                            mapslices(frames[:, :, :, best_corr_order], N+1),
                            translations[best_corr_order])
        reference = mean(average_stack, dims=N+1)[:, :, :, 1]
    end
    return reference
end