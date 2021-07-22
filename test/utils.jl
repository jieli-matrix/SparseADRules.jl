using NiSparseArrays:params, fitparams
"""
    forwarddiff_mv_jacobian(A, x) -> AbstractArray J 
forwarddiff_mv_jacobian calculates the jacobian with respect to parameter A and x based on multiplication in ForwardDiff
"""
function forwarddiff_mm_jacobian(A, x)
    pA = params(A)
    px = params(x)
    pAx = vcat(pA, px)
    ForwardDiff.jacobian(pAx) do pAx
        pA = pAx[1:length(pA)]
        px = pAx[length(pA)+1:end]
        A2 = fitparams(A, pA)
        x2 = fitparams(x, px)
        y = params(A2 * x2)
    end
end