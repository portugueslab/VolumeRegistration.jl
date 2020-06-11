```@meta
CurrentModule = VolumeRegistration
```

# Usage examples

## Registering a large volumetric dataset that doesn't fit in memory

```julia
using ProgressMeter
using Strided
using VolumeRegistration

nt, nx, ny, nz = size(ds)

n_frames_block = 50

shifts = []
@showprogress for frame_range in Iterators.partition(1:nt, n_frames_block)
    # load a block that fits twice in memory
    source_slice = ds[:,:,:,frame_range]
    
    # using the strided macro for threaded broadcasting
    big_slice = @strided(Float32.(source_slice))
    
    slice_shifts = find_translation(big_slice, reference,
        max_shift=(20, 20, 2), σ_filter=(3.0f0, 3.0f0, 0.1f0),
        border_σ=(10f0, 10f0, 0.1f0), interpolate_middle=true)[1]

    push!(shifts, slice_shifts)

    translated = translate(big_slice, slice_shifts)

    block_shifts, shift_blocks = find_deformation_map(translated, reference,
        max_shift=(5,5,3), upsampling=(10,10,4),
        σ_filter=1.3f0, border_σ=(5f0,5f0,1f0),
        block_size=(256,256,4), block_border_σ=(5f0,5f0, 0.5f0));

    undeformed = apply_deformation_map(translated, block_shifts, shift_blocks)
    
    # flipping the order of dimensions for more efficient source extraction
    aligned[frame_range, :, :, :] = @strided(permutedims(
            round.(UInt16,clamp.(undeformed, 0, typemax(UInt16))),
            (4, 1, 2, 3)))
end
```

## Plane-wise registration of two-photon data