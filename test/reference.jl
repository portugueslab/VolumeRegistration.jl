@testset "Reference creation" begin
    for (test_size, corr_win) in [((100,100,10,30), (30,30,3)),
                                  ((100,100,30), (30,30))]
        test_data = rand(Float32, test_size)
        ref = make_reference(test_data, corr_win=corr_win)
        @test size(ref) == test_size[1:end-1]
    end
end