@testset "Translation-only registration" begin
    reference = Float32.(testimage("moonsurface"))
    moving = warp(reference, Translation(-2, 5.9), axes(reference), ImageTransformations.Linear(), 0)

    tr = find_translation(moving, reference)
    @test tr == (-2, 6) .* -1

    tr = find_translation(moving, reference, upsampling=(10, 10))

    @test all(isapprox.(tr, (-2, 5.9) .* -1))
end