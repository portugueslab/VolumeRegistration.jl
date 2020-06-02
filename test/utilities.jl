function make_test_image(dimensions; levels=3, σ_filt = 0.5f0, σ_border = 10f0)
    final_img = zeros(Float32, dimensions)
    for i_level in 0:levels-1
        factor = (2^i_level)
        im = rand(Float32, dimensions .÷ factor)
        im_filtered = real.(ifft(fft(im) .* VolumeRegistration.gaussian_fft_filter(dimensions .÷ factor, σ_filt)))
        final_img .+= imresize(im_filtered, ratio=factor) ./ levels
    end
    mask = VolumeRegistration.gaussian_border_mask(dimensions, σ_border)
    return final_img .* mask
end