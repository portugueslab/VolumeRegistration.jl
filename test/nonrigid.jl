using Rotations

@testset "Deformation registration" begin
    for (imsize, blocksize, block_rotation) in [
        ((128, 128), (32, 32), LinearMap(RotMatrix(π / 80))),
        ((128, 128, 32), (32, 32, 5), RotXYZ(0.1, -0.01, -0.05)),
    ]
        test_img = make_test_image(imsize)

        rotation = recenter(block_rotation, collect(size(test_img)) ./ 2)

        rotated = warp(test_img, rotation, axes(test_img))

        shifts, blocks = find_deformation_map(rotated, test_img, block_size=blocksize)

        unrotated = apply_deformation_map(rotated, shifts, blocks)

        im_mid = imsize .÷ 2
        bshalf = blocksize .÷ 2
        check_range = CartesianIndex(im_mid .- bshalf):CartesianIndex(im_mid .+ bshalf)
        @test all(isapprox.(unrotated[check_range],
                            test_img[check_range], atol=3e-1))
    end


    # test multiple images
    imsize, blocksize = (128, 128), (32, 32)
    n_rotations = 3

    test_img = make_test_image(imsize)

    rotated = similar(test_img, size(test_img)..., n_rotations)

    for i_rot in 1:n_rotations
        block_rotation = LinearMap(RotMatrix(π*i_rot / 600))
        rotation = recenter(block_rotation, collect(size(test_img)) ./ 2)

        rotated[:, :, i_rot] = warp(test_img, rotation, axes(test_img))
    end

    shifts, blocks = find_deformation_map(rotated, test_img, block_size=blocksize)


    unrotated = apply_deformation_map(rotated, shifts, blocks)

    im_mid = imsize .÷ 2
    bshalf = blocksize .÷ 2
    check_range = CartesianIndex(im_mid .- bshalf):CartesianIndex(im_mid .+ bshalf)

    for i_rot in 1:n_rotations
        @test all(isapprox.(unrotated[check_range, i_rot],
                            test_img[check_range], atol=3e-1))
    end

end
