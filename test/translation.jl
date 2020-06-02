const registration_image_error_tolerance = 0.05

@testset "Translation-only registration" begin
    imsize = 128
    values = range(-5f0, stop = 5f0, length = imsize)
    dim_shifts = (-3.2, 2.9, 1.2)
    n_images = 4

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
                ),
                size(reference)...,
                1,
            ) for sh in current_shift
        )

        im = moving[(Colon() for _ in 1:n_dims)..., 1]

        # calculate the pad to test the equality of reference when transfomed back
        pad_test = CartesianIndex(maximum(
            ceil.(
                Int,
                abs.(reshape(
                    reinterpret(typeof(current_shift[1][1]), current_shift),
                    n_dims,
                    :,
                )),
            ),
            dims = 2,
        )...)

        # test single image
        tr = find_translation(im, reference)[1]

        @test all(Tuple(tr.translation) .== .-round.(current_shift[1]))

        tr = find_translation(im, reference; ups_params...)[1]
        @test all(abs.(Tuple(tr.translation) .+ current_shift[1]) .< 0.3)

        # test that after the image has been moved back it is approximately the same
        @test all(
            abs.(translate(im, tr) .- reference)[pad_test:CartesianIndex(size(
                im,
            ))-pad_test] .< registration_image_error_tolerance,
        )

        # test image sequence
        tr = find_translation(moving, reference)[1]
        difs = [Tuple(t.translation) .== .- round.(h) for (h, t) in zip(current_shift, tr)]
        @test all((
            all(Tuple(t.translation) .== .- round.(h)) for (h, t) in zip(current_shift, tr)
        ))

        tr = find_translation(moving, reference; ups_params...)[1]
        dif = [abs.(h .+ Tuple(t.translation))
                    for (h, t) in zip(current_shift, tr)]
        @test all((
            all(abs.(h .+ Tuple(t.translation)) .< 0.3)
            for (h, t) in zip(current_shift, tr)
        ))

        # test that after the image has been moved back they are approximately the same
        @test all(
            abs.(translate(moving, tr) .- reshape(reference, size(reference)..., 1))[
                pad_test:CartesianIndex(size(im))-pad_test,
                :,
            ] .< registration_image_error_tolerance,
        )

    end
end
