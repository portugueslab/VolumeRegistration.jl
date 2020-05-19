@testset "kriging_upsampling" begin
    ks = VolumeRegistration.KrigingUpsampler()
    test_ups = zeros(ks.original_size)
    test_ups[4,3:5] .= 0.5
    test_ups[3:5,4] .= 0.5
    test_ups[4,4] = 1
    upsampled = VolumeRegistration.upsample(ks, test_ups)
    shift_loc = VolumeRegistration.upsampled_shift(ks, reshape(test_ups, size(test_ups)..., 1))
    @test shift_loc[1] == (0.0, 0.0)
end