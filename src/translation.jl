"""
Find the shift to move `moving` by to align with `reference`

# Arguments
- `border_σ`: the width of the border kernal for a smooth faloff towards the edges
- `upsampling::Integer=1`: If bigger than 1, how much to upsample the shifts by (for subpixel registration)
"""
function find_translation(moving::AbstractArray{T, N}, reference::AbstractArray{T, N}; max_shift=10, border_σ=0, upsampling=1) where {N, T}
    if border_σ > 0
        mask = gaussian_border_mask(size(reference), border_σ)
        reference_mask_offset = mask_offset(reference, mask)
        moving_corr = phase_correlation((moving .* mask) .+ mask_offset, reference)
    else
        moving_corr = phase_correlation(moving, reference)
    end

    window_size = max_shift_to_window_size(N, max_shift)
    if upsampling !== 1
        us = KrigingUpsampler() # TODO Proper argument conversion, check dealing with NTuples in JuliaImages
        return phase_correlation_shift(moving_corr, window_size, us)
    else
        return phase_correlation_shift(moving_corr, window_size)
    end
end

"""
Small utilities to get window size from iterables or single number
"""
function max_shift_to_window_size(N, max_shift::Integer)
    ws = max_shift * 2 + 1 
    return tuple((ws for _ in 1:N)...)
end

function max_shift_to_window_size(N, max_shift)
    return tuple((ms*2 + 1 for ms in max_shift)...)
end
