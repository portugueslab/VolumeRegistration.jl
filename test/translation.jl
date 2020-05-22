@testset "Translation-only registration" begin
    imsize = 128
    values = range(-5f0, stop=5f0, length=imsize)
    dim_shifts = [-1, 2.9, 3.2]
    upsampling_paramters = [(upsampling=20, upsample_padding=7), (upsampling=5, upsample_padding=3)]
    
    n_dims = 2
    reference = exp.(-[sum(xs .^2) for xs in Iterators.product([values for _ in 1:n_dims]...)])
    current_shift = dim_shifts[1:n_dims]
    moving = warp(reference, Translation(current_shift...), axes(reference), ImageTransformations.Linear(), 0)

    tr = find_translation(moving, reference)
    @test all(tr .== -round.(current_shift))

    tr = find_translation(moving, reference, upsampling=(10, 10))
    @show tr current_shift
    @test all(abs.(tr .+ current_shift) .< 0.3)

end