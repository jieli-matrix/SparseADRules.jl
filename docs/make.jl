using SparseArraysAD
using Documenter

DocMeta.setdocmeta!(SparseArraysAD, :DocTestSetup, :(using SparseArraysAD); recursive=true)

makedocs(;
    modules=[SparseArraysAD],
    authors="Jie Li",
    repo="https://github.com/jieli-matrix/SparseArraysAD.jl/blob/{commit}{path}#{line}",
    sitename="SparseArraysAD.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jieli-matrix.github.io/SparseArraysAD.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jieli-matrix/SparseArraysAD.jl",
)
