const registration_image_error_tolerance = 0.15
const noise_level = 0.01

@testset "Translation-only registration" begin
    dim_shifts = (-3.2, 2.9, 1.2)
    n_images = 3

    registration_params = [(upsampling = 10,),
                           (upsampling = 5, upsample_padding = 3),
                           (upsampling = 10, interpolate_middle=true, border_σ=5f0)]

    for (dimensions, ups_params) in zip([(128, 128),
        (128,128, 16),
        (128, 128)
    ], registration_params)
        reference = make_test_image(dimensions)
        n_dims = length(dimensions)
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
                ) .+ rand(Float64, dimensions...)*noise_level,
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

        @test Tuple(tr.translation) == .- round.(current_shift[1])

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

        for (h, t) in zip(current_shift, tr)
            @test Tuple(t.translation) == .- round.(h)
        end

        tr = find_translation(moving, reference; ups_params...)[1]
        dif = [abs.(h .+ Tuple(t.translation))
                    for (h, t) in zip(current_shift, tr)]

        for (h, t) in zip(current_shift, tr)
            @test collect(.-round.(h)) ≈ collect(t.translation) atol=0.3
        end

        # test that after the image has been moved back they are approximately the same
        @test all(
            abs.(translate(moving, tr) .- reshape(reference, size(reference)..., 1))[
                pad_test:CartesianIndex(size(im))-pad_test,
                :,
            ] .< registration_image_error_tolerance,
        )

    end
end
