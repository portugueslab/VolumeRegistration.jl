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

function ncolons(::Val{N}) where {N}
    return ntuple(x->Colon(), Val{N}())
end

function nones(::Val{N}) where {N}
    return ntuple(x->1, Val{N}())
end

function make_test_image(dimensions; σ_filt = 0.9f0, σ_border = 10f0)
    im = rand(Float32, dimensions)
    im_filtered = real.(ifft(fft(im) .* gaussian_fft_filter(dimensions, σ_filt)))
    mask = gaussian_border_mask(dimensions, σ_border)
    return im_filtered .* mask
end