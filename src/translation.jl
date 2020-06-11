"""
Prepare everything common for all translations
"""
function prepare_translation(reference::AbstractArray{T, N};
    σ_filter = nothing,
    max_shift = N == 2 ? 15 : (15, 15, 5),
    border_σ = 0,
    upsampling = 1,
    upsample_padding = nothing) where {T, N}
    
    if border_σ != 0
        mask = gaussian_border_mask(size(reference), border_σ)
        reference_mask_offset = mask_offset(reference, mask)
    else
        mask = nothing
        reference_mask_offset = nothing
    end

    window_size = to_ntuple(Val{N}(), max_shift) .* 2 .+ 1

    if upsampling != 1
        upsampling = to_ntuple(Val{N}(), upsampling)
        if upsample_padding === nothing
            upsample_padding = (upsampling .÷ 2 .- 2)
        else
            upsample_padding = to_ntuple(Val{N}(), upsample_padding)
        end
        us = KrigingUpsampler(upsampling = upsampling, padding = upsample_padding)
    else
        us = nothing
    end

    # Move the reference to Fourier domain as it is common for the whole registration
    fft_ref = prepare_fft_reference(reference, σ_filter)

    # Prepare the FFT plan for faster FFTs of images of same size
    fft_plan = plan_fft(reference)

    return (mask, reference_mask_offset, window_size, us, fft_ref, fft_plan)
end

function prepared_find_translation(moving::AbstractArray{T,N}, mask, reference_mask_offset, window_size, us, fft_ref, fft_plan, interpolate_middle=false) where {T, N}
    if mask !== nothing
        moving_corr = phase_correlation(
            fft_plan * (moving .* mask .+ reference_mask_offset),
            fft_ref,
        )
    else
        moving_corr = phase_correlation(fft_plan * moving, fft_ref)
    end

    if us !== nothing
        shift, corr = phase_correlation_shift(
            moving_corr,
            window_size,
            us,
            interpolate_middle = interpolate_middle,
        )
    else
        shift, corr = phase_correlation_shift(
            moving_corr,
            window_size,
            interpolate_middle = interpolate_middle,
        )
    end

    return Translation(shift), corr
end

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
    movings::AbstractArray{T,M},
    reference::AbstractArray{T,N};
    interpolate_middle = false,
    kwargs...
) where {N,M,T}
   
    (mask, reference_mask_offset, window_size, us, fft_ref, fft_plan) = prepare_translation(reference; kwargs...)

    moving_slices = Slices(movings, (1:N)...)

    n_t = size(movings)[end]

    if us !== nothing
        shifts = Array{Translation{SVector{N,Float64}}}(undef, n_t)
    else
        shifts = Array{Translation{SVector{N,Int64}}}(undef, n_t)
    end

    correlations = Array{Float32}(undef, n_t)

    Base.Threads.@threads for i_t in 1:n_t
        shifts[i_t], correlations[i_t] = prepared_find_translation(moving_slices[i_t], mask, reference_mask_offset, window_size, us, fft_ref, fft_plan)
    end
    return (shifts=shifts, correlations=correlations)
end

function find_translation(
    moving::AbstractArray{T,N},
    reference::AbstractArray{T,N};
    interpolate_middle = false,
    kwargs...
) where {N,T}
    (mask, reference_mask_offset, window_size, us, fft_ref, fft_plan) = prepare_translation(reference; kwargs...)
    shift, correlation = prepared_find_translation(moving, mask, reference_mask_offset, window_size, us, fft_ref, fft_plan)
    return (shift=shift, correlation=correlation)
end

function find_translation(
    movings::AbstractArray{TM,M},
    reference::AbstractArray{T,N};
    kwargs...,
) where {TM,T,M,N}
    return find_translation(T.(movings), reference; kwargs...)
end
