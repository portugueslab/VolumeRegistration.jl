@testset "Translation-only registration" begin
    imsize = 128
    values = range(-5f0, stop = 5f0, length = imsize)
    dim_shifts = (-1, 2.9, 3.2)
    n_images = 5

    upsampling_paramters = [(upsampling = 10,), (upsampling = 5, upsample_padding = 3)]

    for (n_dims, ups_params) in zip(2:3, upsampling_paramters)
        reference =
            exp.(-[sum(xs .^ 2) for xs in Iterators.product([values for _ in 1:n_dims]...)])

        
        current_shift =
            [dim_shifts[1:n_dims] .+ extra_shift for extra_shift in 0:n_images-1]
        moving = reduce(
            (x, y) -> cat(x, y, dims = n_dims + 1),
            reshape(
                warp(
                    reference,
                    Translation(sh...),
                    axes(reference),
                    ImageTransformations.Linear(),
                    0,
                ), size(reference)...,
                1,
            ) for sh in current_shift
        )
        
        im = moving[(Colon() for _ in 1:n_dims)..., 1]

        # test single image
        tr = find_translation(im, reference)
        @test all(tr .== .- round.(current_shift[1]))

        tr = find_translation(im, reference; ups_params...)
        @test all(abs.(tr .+ current_shift[1]) .< 0.3)

        # test image sequence
        tr = find_translation(moving, reference)
        @show tr current_shift
        @test all((all(t .== .- round.(h)) for (h, t) in zip(current_shift, tr)))

        tr = find_translation(moving, reference; ups_params...)
        @show tr current_shift
        @test all((all(abs.(h .+ t) .< 0.3) for (h, t) in zip(current_shift, tr)))

    end
end
