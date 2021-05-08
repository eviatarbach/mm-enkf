module ETKF

export etkf

using Statistics
using LinearAlgebra
using Distributions

"""
Ensemble transform Kalman filter (ETKF)
"""
function etkf(; E::AbstractMatrix{float_type}, R_inv::AbstractMatrix{float_type},
                inflation::float_type=1.0, H::AbstractMatrix,
                y::AbstractVector{float_type}) where {float_type<:AbstractFloat}
    D, m = size(E)

    x_m = mean(E, dims=2)
    X = (E .- x_m)/sqrt(m - 1)

    X = sqrt(inflation)*X

    y_m = H*x_m
    Y = (hcat([H*E[:, i] for i=1:m]...) .- y_m)/sqrt(m - 1)
    Ω = real((I + Y'*R_inv*Y)^(-1))
    w = Ω*Y'*R_inv*(y - y_m)

    E = real(x_m .+ X*(w .+ sqrt((m - 1)*Ω)))

    return E
end

end
