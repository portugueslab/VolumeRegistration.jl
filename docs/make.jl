using VolumeRegistration
using Documenter

makedocs(;
    modules=[VolumeRegistration],
    authors="Vilim <vilim@neuro.mpg.de> and contributors",
    repo="https://github.com/vilim/VolumeRegistration.jl/blob/{commit}{path}#L{line}",
    sitename="VolumeRegistration.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://vilim.github.io/VolumeRegistration.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/portugueslab/VolumeRegistration.jl",
)
