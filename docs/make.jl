if joinpath(@__DIR__, "..") ∉ LOAD_PATH
    push!(LOAD_PATH, joinpath(@__DIR__, ".."))
end
using Documenter
using OperatorFlux

makedocs(
    modules = [OperatorFlux],
    sitename = "OperatorFlux.jl",
    authors = "CliMA",
    format = Documenter.HTML(
        prettyurls = !isempty(get(ENV, "CI", "")),
        collapselevel = 1,
        mathengine = MathJax3(),
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        # "Installation instructions" => "installation_instructions.md",
        # "Running instructions" => "running_instructions.md",
        # "Contributor Guide" => "contributor_guide.md",
        # "Function Index" => "function_index.md",
    ],
)

deploydocs(repo = "github.com/bischtob/OperatorFlux.jl.git", devbranch = "main")
