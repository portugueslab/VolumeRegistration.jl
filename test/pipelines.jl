using Rotations

@testset "Pipeline" begin
    # test multiple images
    imsize, blocksize = ((128, 128, 32), (32, 32, 5))
    n_rotations = 3

    test_img = round.(UInt16, make_test_image(imsize) * 255)

    rotated = similar(test_img, size(test_img)..., n_rotations)

    for i_rot in 1:n_rotations
        block_rotation = RotXYZ(0.0, -0.05, π * i_rot / 300 )
        rotation = recenter(block_rotation, collect(size(test_img)) ./ 2)

        rotated[:, :, :, i_rot] =
            round.(UInt16, warp(test_img, rotation, axes(test_img), Linear(), 0))
    end

    unrotated = similar(rotated)

    shifts, correlations = register_volumes!(
        unrotated,
        rotated,
        test_img,
        output_time_first = false,
        t_block_size = 3,
        block_size = blocksize,
        upsampling = (8,8,4),
        σ_filter_nonrigid = nothing,
        border_σ_nonrigid = nothing,
        block_border_σ = 1f0,
        interpolation_nonrigid=Cubic(Line(OnGrid())),
    )

    im_mid = imsize .÷ 2
    bshalf = blocksize .÷ 2
    check_range = CartesianIndex(im_mid .- bshalf):CartesianIndex(im_mid .+ bshalf)

    for i_rot in 1:n_rotations
        @test all(isapprox.(
            unrotated[check_range, i_rot]*1f0,
            test_img[check_range]*1f0,
            atol = 255*0.3,
        ))
    end

end
