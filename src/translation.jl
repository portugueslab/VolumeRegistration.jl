"""

    $(SIGNATURES)

Find the shift to move `moving` by to align with `reference`

# Arguments
- `max_shift::Union{Integer, Tuple}`: the maximal shift in each dimension 
- `border_σ`: the width of the border kernel for a smooth faloff towards the edges
- `upsampling::Union{Integer, Tuple}=1`: If bigger than 1, how much to upsample the shifts by (for subpixel registration)
- `upsample_padding::Union{Integer, Tuple}=nothing` how much ± pixels to take around maximum to upsample
- `interpolate_middle::Bool`: whether to interpolate the middle correlation pixel (to avoid static camera noise)
"""
function find_translation(
    moving::AbstractArray{T,N},
    reference::AbstractArray{T,N};
    σ_filter = nothing,
    max_shift = 10,
    border_σ = 0,
    upsampling = 1,
    upsample_padding = nothing,
    interpolate_middle = false,
) where {N,T}
    if border_σ != 0
        mask = gaussian_border_mask(size(reference), border_σ)
        reference_mask_offset = mask_offset(reference, mask)
        moving_corr = phase_correlation(
            (moving .* mask) .+ reference_mask_offset,
            reference,
            σ_ref = σ_filter,
        )
    else
        moving_corr = phase_correlation(moving, reference, σ_ref = σ_filter)
    end

    window_size = to_ntuple(Val{N}(), max_shift) .* 2 .+ 1
    if upsampling !== 1
        upsampling = to_ntuple(Val{N}(), upsampling)
        if upsample_padding === nothing
            upsample_padding = (upsampling .÷ 2 .- 2)
        else
            upsample_padding = to_ntuple(Val{N}(), upsample_padding)
        end
        us = KrigingUpsampler(upsampling = upsampling, padding = upsample_padding)
        shift, corr = phase_correlation_shift(moving_corr, window_size, us)
    else
        shift, corr = phase_correlation_shift(moving_corr, window_size)
    end
    return Translation(shift), corr
end

# method for aligning a time-series with the same reference
# the steps are identical as above, apart from some optimizations
function find_translation(
    movings::AbstractArray{T,M},
    reference::AbstractArray{T,N};
    σ_filter = nothing,
    max_shift = N == 2 ? 15 : (15, 15, 5),
    border_σ = 0,
    upsampling = 1,
    upsample_padding = nothing,
    interpolate_middle = false,
) where {N,M,T}
    if border_σ != 0
        mask = gaussian_border_mask(size(reference), border_σ)
        reference_mask_offset = mask_offset(reference, mask)
    end

    # Move the reference to Fourier domain as it is common for the whole registration
    fft_ref = prepare_fft_reference(reference, σ_filter)

    window_size = to_ntuple(Val{N}(), max_shift) .* 2 .+ 1

    n_t = size(movings)[end]
    if upsampling != 1
        shifts = Array{Translation{SVector{N,Float64}}}(undef, n_t)
    else
        shifts = Array{Translation{SVector{N,Int64}}}(undef, n_t)
    end

    correlations = Array{Float32}(undef, n_t)

    if upsampling !== 1
        upsampling = to_ntuple(Val{N}(), upsampling)
        if upsample_padding === nothing
            upsample_padding = (upsampling .÷ 2 .- 2)
        else
            upsample_padding = to_ntuple(Val{N}(), upsample_padding)
        end
        us = KrigingUpsampler(upsampling = upsampling, padding = upsample_padding)
    end

    # Prepare the FFT plan for faster FFTs of images of same size
    fft_plan = plan_fft(reference)

    moving_slices = Slices(movings, (1:N)...)

    for i_t in 1:n_t
        if border_σ != 0
            moving_corr = phase_correlation(
                fft_plan * (moving_slices[i_t] .* mask .+ reference_mask_offset),
                fft_ref,
            )
        else
            moving_corr = phase_correlation(fft_plan * (moving_slices[i_t]), fft_ref)
        end

        if upsampling !== 1
            shift, corr = phase_correlation_shift(moving_corr, window_size, us, interpolate_middle=interpolate_middle)
        else
            shift, corr = phase_correlation_shift(moving_corr, window_size, interpolate_middle=interpolate_middle)
        end
        shifts[i_t] = Translation(shift)
        correlations[i_t] = corr
    end
    return shifts, correlations
end

function find_translation(
    movings::AbstractArray{TM,M},
    reference::AbstractArray{T,N};
    kwargs...,
) where {TM,T,M,N}
    return find_translation(T.(movings), reference; kwargs...)
end
