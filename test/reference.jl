@testset "Reference creation" begin
    for test_size in [(100,100,10,5), (100,100,5)]
        test_data = rand(Float32, test_size)
        ref = make_reference(test_data)
        @test size(ref) == test_size[1:end-1]
    end
end