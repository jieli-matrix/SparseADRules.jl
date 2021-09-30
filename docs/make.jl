using SparseADRules
using Documenter

DocMeta.setdocmeta!(SparseADRules, :DocTestSetup, :(using SparseADRules); recursive=true)

makedocs(;
    modules=[SparseADRules],
    authors="Jie Li",
    repo="https://github.com/jieli-matrix/SparseADRules.jl/blob/{commit}{path}#{line}",
    sitename="SparseADRules.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jieli-matrix.github.io/SparseADRules.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jieli-matrix/SparseADRules.jl",
)
