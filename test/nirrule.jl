using ChainRules
using NiSparseArrays:imul!

function ChainRules.rrule(::typeof(imul!), C::StridedVecOrMat, A::AbstractSparseMatrix, B::DenseInputVecOrMat, α::Number, β::Number)
    out = imul!(copy(C), A, B, α, β)[1]
    function pullback(ȳ)
        ChainRules.NoTangent(), grad((~imul!)(GVar(out, ȳ), GVar(C), GVar(A), GVar(B))[2])
    end
    out, pullback
end

# need to test
# initialize A B C 1.0 1.0 
# imul'(C, A, B, 1.0, 1.0) ≈ original_grad
# what is original_grad? 