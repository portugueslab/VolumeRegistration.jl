using Rotations

@testset "Volume pipeline" begin
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
        upsampling_deform = (8,8,4),
        σ_filter_deform = nothing,
        border_σ_deform= nothing,
        block_border_σ = 1f0,
        interpolation_deform=Cubic(Line(OnGrid())),
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

@testset "Planar pipeline" begin
    image_size = (128, 128)
    n_planes = 20
    n_t = 4
    single_plane_image = round.(UInt16, make_test_image(image_size, σ_border=20f0)*255)
    moved = zeros(UInt16, size(single_plane_image)..., n_planes);
    translations = [Translation(-i_plane+10, round(Int,6*sin((i_plane-10)/10))) for i_plane in 1:n_planes]
    for i_plane in 1:n_planes
        moved[:,:, i_plane] = translate(single_plane_image, translations[i_plane])
    end
    wiggled = zeros(UInt16, size(moved)..., n_t)
    for i_t in 1:n_t
        for i_plane in 1:n_planes
            wiggled[:, :, i_plane, i_t] = translate(moved[:, :, i_plane], Translation(sin((i_t-i_plane)*4π/n_t), cos((i_t+i_plane)*2π/n_t)))
        end
    end
    found_reference = make_planewise_reference(wiggled, n_average=3)

    corrected = similar(wiggled);
    register_planewise!(corrected, wiggled, found_reference.reference, output_time_first=false);

    idx_mid = image_size .÷ 2
    win_size=20
    test_range = CartesianIndex(idx_mid .- win_size):CartesianIndex(idx_mid .+ win_size);

    @test all(corrected[test_range, :, :] .== single_plane_image[test_range, 1, 1])
end