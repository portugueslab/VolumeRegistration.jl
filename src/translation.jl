"""
Find the shift to move `moving` by to align with `reference`

# Arguments
- `border_σ`: the width of the border kernal for a smooth faloff towards the edges
- `upsampling::Integer=1`: If bigger than 1, how much to upsample the shifts by (for subpixel registration)
"""
function find_translation(moving::AbstractArray{T, N}, reference::AbstractArray{T, N}; max_shift=10, border_σ=0, upsampling=1, upsample_padding=nothing) where {N, T}
    if border_σ > 0
        mask = gaussian_border_mask(size(reference), border_σ)
        reference_mask_offset = mask_offset(reference, mask)
        moving_corr = phase_correlation((moving .* mask) .+ reference_mask_offset, reference)
    else
        moving_corr = phase_correlation(moving, reference)
    end

    window_size = to_ntuple(Val{N}(), max_shift) .* 2 .+ 1
    if upsampling !== 1
        upsampling = to_ntuple(Val{N}(), upsampling)
        if upsample_padding === nothing
            upsample_padding = (upsampling .÷ 2 .- 2)
        else
            upsample_padding = to_ntuple(Val{N}(), upsample_padding)
        end
        us = KrigingUpsampler(upsampling=upsampling, padding=upsample_padding)
        return phase_correlation_shift(moving_corr, window_size, us)
    else
        return phase_correlation_shift(moving_corr, window_size)
    end
end


# method for time-series with the same reference
function find_translation(movings::AbstractArray{T, M}, reference::AbstractArray{T, N}; max_shift=10, border_σ=0, upsampling=1, upsample_padding=nothing) where {N, M, T}
    if border_σ > 0
        mask = gaussian_border_mask(size(reference), border_σ)
        reference_mask_offset = mask_offset(reference, mask)
        reference .*= mask
    end

    # Move the reference to Fourier domain as it is common for the whole registration
    fft_ref = prepare_fft_reference(reference)

    window_size = to_ntuple(Val{N}(), max_shift) .* 2 .+ 1

    n_t = size(movings)[end]    
    shifts = Array{NTuple{N, Float32}}(undef, n_t)

    if upsampling !== 1
        upsampling = to_ntuple(Val{N}(), upsampling)
        if upsample_padding === nothing
            upsample_padding = (upsampling .÷ 2 .- 2)
        else
            upsample_padding = to_ntuple(Val{N}(), upsample_padding)
        end
        us = KrigingUpsampler(upsampling=upsampling, padding=upsample_padding)
    end

    # Prpare the FFT plan for faster FFTs of images of same size
    fft_plan = plan_fft(reference)

    for i_t in 1:n_t
        if border_σ > 0
            moving_corr = phase_correlation(fft_plan*((movings[(Colon() for i in 1:M-1)..., i_t] .* mask) .+ reference_mask_offset), fft_ref)
        else
            moving_corr = phase_correlation(fft_plan*(movings[(Colon() for i in 1:M-1)..., i_t]), fft_ref)
        end
    
        if upsampling !== 1
            shifts[i_t] = phase_correlation_shift(moving_corr, window_size, us)
        else
            shifts[i_t] = phase_correlation_shift(moving_corr, window_size)
        end
    end
    return shifts
end

"""
Small utilities to get window size from iterables or single number
"""
function to_ntuple(::Val{N}, val::Number) where {N}
    return ntuple(x->val, Val{N}())
end

function to_ntuple(::Val{N}, val::NTuple{N, T}) where {N, T}
    return val
end

function to_ntuple(::Val{N}, val) where {N}
    error("Unsupported paramter format")
end