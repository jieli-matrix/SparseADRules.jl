using NiSparseArrays
using Documenter

DocMeta.setdocmeta!(NiSparseArrays, :DocTestSetup, :(using NiSparseArrays); recursive=true)

makedocs(;
    modules=[NiSparseArrays],
    authors="JieLi",
    repo="https://github.com/jieli-matrix/NiSparseArrays.jl/blob/{commit}{path}#{line}",
    sitename="NiSparseArrays.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jieli-matrix.github.io/NiSparseArrays.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jieli-matrix/NiSparseArrays.jl",
)
