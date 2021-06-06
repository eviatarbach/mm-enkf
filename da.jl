module DA

export etkf, ensrf, gaspari_cohn

using Statistics
using LinearAlgebra
using Distributions

function gaspari_cohn(c, D)
    ρ = zeros(D, D)
    for i=1:D
        for j=1:i
            r = abs(i - j)/c
            if 0 <= r < 1
                G = 1 - 5/3*r^2 + 5/8*r^3 + 1/2*r^4 - 1/4*r^5
            elseif 1 <= r < 2
                G = 4 - 5*r + 5/3*r^2 + 5/8*r^3 - 1/2*r^4 + 1/12*r^5 - 2/(3*r)
            elseif r >= 2
                G = 0
            end
            ρ[i, j] = G
        end
    end
    return Symmetric(ρ, :L)
end

"""
Ensemble transform Kalman filter (ETKF)
"""
function etkf(; E::AbstractMatrix{float_type}, R_inv::Symmetric{float_type},
                inflation::float_type=1.0, H::AbstractMatrix,
                y::AbstractVector{float_type}) where {float_type<:AbstractFloat}
    D, m = size(E)

    x_m = mean(E, dims=2)
    X = (E .- x_m)/sqrt(m - 1)

    X = sqrt(inflation)*X

    y_m = H*x_m
    Y = (H*E .- y_m)/sqrt(m - 1)
    Ω = inv(Symmetric(I + Y'*R_inv*Y))
    w = Ω*Y'*R_inv*(y - y_m)

    E = x_m .+ X*(w .+ sqrt(m - 1)*sqrt(Ω))

    return E
end

function ensrf(; E::AbstractMatrix{float_type}, R_inv::Symmetric{float_type},
                 inflation::float_type=1.0, H::AbstractMatrix,
                 y::AbstractVector{float_type}, ρ=nothing) where {float_type<:AbstractFloat}
    D, m = size(E)

    x_m = mean(E, dims=2)
    X = (E .- x_m)/sqrt(m - 1)

    if ρ === nothing
        P = inflation*X*X'
    else
        P = inflation*ρ.*(X*X')
    end

    K = P*H'*inv(H*P*H' + inv(R_inv))
    x_m .+= K*(y - H*x_m)

    E = x_m .+ inv(sqrt(I + P*H'*R_inv*H))*X
end

end
