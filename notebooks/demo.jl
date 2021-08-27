### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ d84b4ef4-0708-11ec-1987-0d965b91965c
begin
	using Pkg
	Pkg.activate()
	using NiLang, NiLang.AD, Zygote, Plots, Optim, LinearAlgebra, ChainRules
	using PlutoUI
end

# ╔═╡ d9d5977b-7bce-490b-a685-7644eeee3607
md"## 2. Inverse engineering a Hamiltonian"

# ╔═╡ e85bf94e-dea6-4dfc-9415-0f234ef9fb3f
md"""
This problem is from "Notes on Adjoint Methods for 18.335", Steven G. Johnson

Consider a 1D Shrodinger equation
```math
\left[-\frac{d^2}{dx^2} + V(x)\right]\Psi(x) = E\Psi(x), x \in [-1,1]
```

"""

# ╔═╡ d27230f9-603c-4729-a459-0b622780688f
md"We can solve its gound state numerically by discretizing the space and diagonalize the Hamiltonian matrix. The Hamiltonian matrix is

```math
A = \frac{1}{Δx^2}\left(
\begin{matrix}
2 & -1 & 0 & \ldots & 0 & -1\\
-1 & 2 & -1 & 0 & \ldots & \\
0 & -1 & 2 & -1 & 0 & \ldots \\
\vdots &  &  & \ddots &  & \\
 & & & -1 & 2 & -1\\
-1 & 0 & \ldots & 0 & -1 & 2
\end{matrix}
\right) + {\rm diag}(V)
```
"

# ╔═╡ 67e118e5-2d47-4623-883e-753296990b7c
md"where the matrix size is equal the descretized lattice size"

# ╔═╡ 5823612b-2f75-429a-b9fd-d6e70cb86fca
dx = 0.02;

# ╔═╡ 184db591-4837-4b07-b5d7-c86bbc9f6b71
xgrid = -1.0:dx:1.0;

# ╔═╡ 4f3c3182-92fd-4446-afce-86cfcec461d9
@i function hamiltonian!(a, x, V::AbstractVector{T}) where T
	@routine begin
		@zeros T dx2 invdx2
		n ← length(x)
		dx2 += (@const Float64(x.step))^2
		invdx2 += 1/dx2
	end
	@safe @assert size(a) == (n, n)
	for i=1:n
		a[i, i] += 2 * invdx2
		a[i, i] += V[i]
		a[i, mod1(i+1, n)] -= invdx2
		a[mod1(i+1, n), i] -= invdx2
	end
	~@routine
end

# ╔═╡ 6042763b-4bfc-44b0-b4a4-07d826a876d5
hamiltonian(x, V) = hamiltonian!(zeros(length(x), length(x)), x, V)[1]

# ╔═╡ 61e8d134-2002-4f74-af20-a5a36cb5aa40
hamiltonian(xgrid, randn(length(xgrid)))

# ╔═╡ 6d12dc89-7c26-488a-ac4a-ecf149e6b7f7
md"Because we are going to use Zygote (with rules set defined in ChainRules)"

# ╔═╡ 2f332b33-c7c4-4e39-b006-03fbf8a781f6
function ChainRules.rrule(::typeof(hamiltonian), x, V)
	y = hamiltonian(x, V)
	function hamiltonian_pullback(Δy)
		gV = NiLang.AD.grad((~hamiltonian!)(GVar.(y, Δy), x, GVar.(V))[3])
		return (ChainRules.NoTangent(), ChainRules.NoTangent(), gV)
	end
	return y, hamiltonian_pullback
end

# ╔═╡ 8e863e09-c4f7-4a79-a773-38b977096f58
md"We want the ground state be a house."

# ╔═╡ fe459fde-2522-4b1a-bae0-92767fd7d7ef
ψ0 = [abs(xi)<0.5 ? 1 - abs(xi) : 0 for xi in xgrid]; normalize!(ψ0);

# ╔═╡ cc954e2c-d8a5-4944-9002-985ff256be05
plot(xgrid, ψ0)

# ╔═╡ 1631234d-99a8-498d-87b5-a959053311b7
md"So we define a loss function
```math
\begin{align}
E, \psi &= {\rm eigensolve}(A)\\
\mathcal{L} &= \sum_i |(|(\psi_0)_i| - |(\psi_G)_i|)|
\end{align}
```
"

# ╔═╡ 20ac32cb-a1cb-4b0b-85e6-893aba85e014
md"where $\psi_G$ is state vector in $\psi$ that corresponds to the minimum energy."

# ╔═╡ 44e22fc9-b42e-4ddf-93b7-48c0f3a7a98c
function solve_wave(x, V)
	a = hamiltonian(x, V)
	ψ = LinearAlgebra.eigen(LinearAlgebra.Hermitian(a)).vectors[:,1]
end

# ╔═╡ 0438da40-8b47-476e-a28a-17f6603c71d0
function loss(x, V, ψ0)
	ψ = solve_wave(x, V)
	sum(map(abs, map(abs, ψ) - map(abs, ψ0))) * dx
end

# ╔═╡ c0d55cf2-ad77-4084-848c-00ef48309fe8
loss(xgrid, randn(length(xgrid)), ψ0)

# ╔═╡ f55d53fb-f832-4370-a3b7-fc8e762d07e3
solve_wave(xgrid, randn(length(xgrid))) |> norm

# ╔═╡ 42aa7127-4565-480d-bd40-c056a88181ff
loss(xgrid, randn(length(xgrid)), ψ0)

# ╔═╡ dc402db3-6818-450a-8ed6-8ebe1d2b3829
Zygote.gradient(v->loss(xgrid, v, ψ0), randn(length(xgrid)))

# ╔═╡ f1fa1373-333b-4a31-b37b-89ba328a1878
res = optimize(v->loss(xgrid, v, ψ0), x->Zygote.gradient(v->loss(xgrid, v, ψ0), x)[1], randn(length(xgrid)), LBFGS(); inplace=false)

# ╔═╡ ea78566c-6896-4811-8ef1-a62e400a42d1
let
	v = Optim.minimizer(res)
	ψ = solve_wave(xgrid, v)
	@show loss(xgrid, v, ψ0)
	plot(xgrid, abs.(ψ); label="ψ")
	plot!(xgrid, abs.(ψ0); label="ψ0")
	plot!(xgrid, normalize(v); label="V")
end |> PlutoUI.as_svg

# ╔═╡ Cell order:
# ╠═d84b4ef4-0708-11ec-1987-0d965b91965c
# ╟─d9d5977b-7bce-490b-a685-7644eeee3607
# ╟─e85bf94e-dea6-4dfc-9415-0f234ef9fb3f
# ╟─d27230f9-603c-4729-a459-0b622780688f
# ╟─67e118e5-2d47-4623-883e-753296990b7c
# ╠═5823612b-2f75-429a-b9fd-d6e70cb86fca
# ╠═184db591-4837-4b07-b5d7-c86bbc9f6b71
# ╠═4f3c3182-92fd-4446-afce-86cfcec461d9
# ╠═6042763b-4bfc-44b0-b4a4-07d826a876d5
# ╠═61e8d134-2002-4f74-af20-a5a36cb5aa40
# ╟─6d12dc89-7c26-488a-ac4a-ecf149e6b7f7
# ╠═2f332b33-c7c4-4e39-b006-03fbf8a781f6
# ╠═8e863e09-c4f7-4a79-a773-38b977096f58
# ╠═fe459fde-2522-4b1a-bae0-92767fd7d7ef
# ╠═cc954e2c-d8a5-4944-9002-985ff256be05
# ╟─1631234d-99a8-498d-87b5-a959053311b7
# ╟─20ac32cb-a1cb-4b0b-85e6-893aba85e014
# ╠═44e22fc9-b42e-4ddf-93b7-48c0f3a7a98c
# ╠═0438da40-8b47-476e-a28a-17f6603c71d0
# ╠═c0d55cf2-ad77-4084-848c-00ef48309fe8
# ╠═f55d53fb-f832-4370-a3b7-fc8e762d07e3
# ╠═42aa7127-4565-480d-bd40-c056a88181ff
# ╠═dc402db3-6818-450a-8ed6-8ebe1d2b3829
# ╠═f1fa1373-333b-4a31-b37b-89ba328a1878
# ╠═ea78566c-6896-4811-8ef1-a62e400a42d1
