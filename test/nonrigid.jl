using Rotations

@testset "Deformation registration" begin
    for (imsize, blocksize, block_rotation) in [
        ((128, 128), (32, 32), LinearMap(RotMatrix(π / 80))),
        ((128, 128, 32), (32, 32, 5), RotXYZ(0.1, -0.01, -0.05)),
    ]
        test_img = VolumeRegistration.make_test_image(imsize)

        rotation = recenter(block_rotation, collect(size(test_img)) ./ 2)

        rotated = warp(test_img, rotation, axes(test_img))

        shifts, blocks = find_deformation_map(rotated, test_img)
    end
end
