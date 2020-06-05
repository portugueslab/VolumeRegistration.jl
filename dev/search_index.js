var documenterSearchIndex = {"docs":
[{"location":"#","page":"Home","title":"Home","text":"CurrentModule = VolumeRegistration","category":"page"},{"location":"#VolumeRegistration-1","page":"Home","title":"VolumeRegistration","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Modules = [VolumeRegistration]","category":"page"},{"location":"#VolumeRegistration.apply_deformation_map-Union{Tuple{M}, Tuple{TS}, Tuple{N}, Tuple{T}, Tuple{AbstractArray{T,M},Array{Array{Tuple{Vararg{TS,N}},N},N1} where N1,Any}} where M where TS where N where T","page":"Home","title":"VolumeRegistration.apply_deformation_map","text":"Corrects a sequence of imaging stacks with a transformation found through non-rigid registration\n\n\n\n\n\n","category":"method"},{"location":"#VolumeRegistration.apply_deformation_map-Union{Tuple{TS}, Tuple{N}, Tuple{T}, Tuple{AbstractArray{T,N},AbstractArray{Tuple{Vararg{TS,N}},N},Any}} where TS where N where T","page":"Home","title":"VolumeRegistration.apply_deformation_map","text":"Corrects a plane or volume with a transformation found through non-rigid registration\n\n\n\n\n\n","category":"method"},{"location":"#VolumeRegistration.find_deformation_map-Union{Tuple{N}, Tuple{T}, Tuple{AbstractArray{T,N},AbstractArray{T,N}}} where N where T","page":"Home","title":"VolumeRegistration.find_deformation_map","text":"Find deformation maps by splitting the dataset in blocks and aligning blocks with subpixel precision\n\nArguments\n\nmoving: the stack to be registered\nreference: the stack to be registered to\nborder_σ: how far to fade out the borders of the whole image/volume\nblock_size::NTuple{N, Integer}: size of blocks to compute deformations\nblock_border_σ::Union{Real, NTuple{N, Real}}: how far to fade out the\nmax_shift::Union{Real, NTuple{N, Integer}}: maximum displacement of a block in each dimension\nσ_filter: low-pass filter width\nupsampling: upsampling of the registration for subpixel alignment\nupsample_padding: how far to pad from the local maximum for upsampling\nsnr_n_smooths::Integer=2: number of times the correlation matrices are smoothed by wieghting with adjacent ones (if they are under signal-to-noise ratio)\nsnr_threshold::Real: the threshold of the \"peakiness\" of the correlation matrix, if it's smaller than that, it's value is obtained by smoothing neighbors\nsnr_n_pad::Integer: window size of the signal-to-noise calculation\n\n\n\n\n\n","category":"method"},{"location":"#VolumeRegistration.find_translation-Union{Tuple{T}, Tuple{M}, Tuple{N}, Tuple{AbstractArray{T,M},AbstractArray{T,N}}} where T where M where N","page":"Home","title":"VolumeRegistration.find_translation","text":"find_translation(movings, reference; σ_filter, max_shift, border_σ, upsampling, upsample_padding, interpolate_middle)\n\n\nFind the shift to move moving by to align with reference\n\nArguments\n\nmax_shift::Union{Integer, Tuple}: the maximal shift in each dimension \nborder_σ: the width of the border kernel for a smooth faloff towards the edges\nupsampling::Union{Integer, Tuple}=1: If bigger than 1, how much to upsample the shifts by (for subpixel registration)\nupsample_padding::Union{Integer, Tuple}=nothing how much ± pixels to take around maximum to upsample\ninterpolate_middle::Bool: whether to interpolate the middle correlation pixel (to avoid static camera noise)\n\n\n\n\n\n","category":"method"},{"location":"#VolumeRegistration.make_reference-Tuple{Any}","page":"Home","title":"VolumeRegistration.make_reference","text":"make_reference(stack; kwargs...)\n\nMake a reference for a stack\n\nArguments\n\nstack: the data to be aligned, time is assumed to be the list dimension\ntime_range::AbstractUnitRange=Colon(): the range of time to take the reference from\ncorr_win: size of the window around the middle of the stack to calculate the optimal reference\nn_refine_from: how many frames to make the refined reference from\nn_average: number of frames to average for a nice reference\nn_iterations: how many times to recalculate the reference from the best correlated moved frames\n\nmore keyword arguments for (@ref find_translation) can be supplied\n\n\n\n\n\n","category":"method"},{"location":"#VolumeRegistration.translate-Union{Tuple{T}, Tuple{N}, Tuple{AbstractArray{T,N},CoordinateTransformations.Translation}, Tuple{AbstractArray{T,N},CoordinateTransformations.Translation,Any}} where T where N","page":"Home","title":"VolumeRegistration.translate","text":"Translate an image by known shift using either reindexing (for integer shifts) of FFT-based phase-space translation for non-integer shifts.\n\n\n\n\n\n","category":"method"},{"location":"#VolumeRegistration.KrigingUpsampler-Tuple{}","page":"Home","title":"VolumeRegistration.KrigingUpsampler","text":"Construct an upsampler with its matrix\n\n\n\n\n\n","category":"method"},{"location":"#VolumeRegistration.calc_snr-Union{Tuple{N}, Tuple{T}, Tuple{AbstractArray{T,N},Any}} where N where T","page":"Home","title":"VolumeRegistration.calc_snr","text":"Computes the signal to noise ratio of a phase correlation array, defined as the ratio of the maximum phase correlation in the window divided by the maximal phase correlation outside of the region where the maximum is (the idea  being that it will be low if the phase correlation is flat versus peaked)\n\n\n\n\n\n","category":"method"},{"location":"#VolumeRegistration.extract_low_frequencies-Union{Tuple{N}, Tuple{T}, Tuple{AbstractArray{Complex{T},N},Tuple{Vararg{Integer,N}}}, Tuple{AbstractArray{Complex{T},N},Tuple{Vararg{Integer,N}},Any}} where N where T","page":"Home","title":"VolumeRegistration.extract_low_frequencies","text":"Takes the data corresponding to the real part of the corners of the  phase correlation array of interest for shift finding\n\nextract_low_frequencies(data, corner_size)\nextract_low_frequencies(data, corner_size, interpolate_middle)\n\n\nArugments\n\n\n\n\n\n","category":"method"},{"location":"#VolumeRegistration.fft_translate-Union{Tuple{T}, Tuple{N}, Tuple{AbstractArray{T,N},Any}} where T<:Complex where N","page":"Home","title":"VolumeRegistration.fft_translate","text":"Translate using discrete fourier transform\n\n\n\n\n\n","category":"method"},{"location":"#VolumeRegistration.gaussian_fft_filter-Union{Tuple{T}, Tuple{N}, Tuple{Tuple{Vararg{Integer,N}},Union{Tuple{Vararg{T,N}}, T}}} where T where N","page":"Home","title":"VolumeRegistration.gaussian_fft_filter","text":"Gaussian filter in the fourier domain (a Gaussian in the fourier domain is again a gaussian with inverse variance)\n\n\n\n\n\n","category":"method"},{"location":"#VolumeRegistration.prepare_deformation_map_calc-Union{Tuple{AbstractArray{T,N}}, Tuple{N}, Tuple{T}} where N where T","page":"Home","title":"VolumeRegistration.prepare_deformation_map_calc","text":"Prepare everything common for calculating deformation maps\n\n\n\n\n\n","category":"method"},{"location":"#VolumeRegistration.refine_reference-Union{Tuple{N}, Tuple{T}, Tuple{AbstractArray{T,N},Any}} where N where T","page":"Home","title":"VolumeRegistration.refine_reference","text":"Iterative refinement of a reference image by aligning more frames and then making a new reference out of the most correlated, aligned frames\n\n\n\n\n\n","category":"method"},{"location":"#VolumeRegistration.split_into_blocks-Union{Tuple{N}, Tuple{T}, Tuple{AbstractArray{T,N},PaddedBlocks.Blocks{N}}, Tuple{AbstractArray{T,N},PaddedBlocks.Blocks{N},Any}} where N where T","page":"Home","title":"VolumeRegistration.split_into_blocks","text":"Splits data into an array of blocks, augmenting the last blocks which may not be full to keep consistent size. Cast immediately into complex for further processing\n\n\n\n\n\n","category":"method"},{"location":"#VolumeRegistration.to_ntuple-Union{Tuple{N}, Tuple{Val{N},Number}} where N","page":"Home","title":"VolumeRegistration.to_ntuple","text":"Small utilities to get window size from iterables or single number\n\n\n\n\n\n","category":"method"},{"location":"#VolumeRegistration.upsampled_shift-Union{Tuple{AT}, Tuple{N}, Tuple{T}, Tuple{VolumeRegistration.KrigingUpsampler{N,AT},AbstractArray{T,N}}} where AT where N where T","page":"Home","title":"VolumeRegistration.upsampled_shift","text":"Finds the shift in the area around the center of the orignial window with  upsampling\n\n\n\n\n\n","category":"method"}]
}
