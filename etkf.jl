module ETKF

export etkf

using Statistics
using LinearAlgebra

"""
Ensemble transform Kalman filter (ETKF)
"""
function etkf(;E, R_inv, inflation, H, y)
    D, m = size(E)

    x_m = mean(E, dims=2)
    X = (E .- x_m)/sqrt(m - 1)

    X = inflation*X

    y_m = H(x_m)
    Y = (vcat([H(E[:, i]) for i=1:m]...) .- y_m)'/sqrt(m - 1)
    Ω = real((I + Y'*R_inv*Y)^(-1))
    w = Ω*Y'*R_inv*(y - y_m)'

    E = real(x_m .+ X*(w .+ sqrt(m - 1)*Ω^(1/2)))

    return E
end

end