function fft_translate(a::AbstractArray{T, N}, shifts) where {T, N}
    return fft_translate(fft(a), shifts)
end

"Translate using discrete fourier transform"
function fft_translate(a::AbstractArray{T, N}, shifts) where {N, T <: Complex }
    shift_mat = fftshift(exp.( -im*2π .* sum_shift.(CartesianIndices(a), Ref(-shifts), Ref(size(a)))));
    return abs.(ifft(a.*shift_mat))
end

# function fft_translate(a::CuArray{T, N}, shifts) where {N, T <: Complex }
#     shift_mat = cu(fftshift(exp.( -im*2π .* sum_shift.(CartesianIndices(a), Ref(-shifts), Ref(size(a))))));
#     return abs.(ifft(a.*shift_mat))
# end