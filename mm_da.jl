include("etkf.jl")
using .ETKF

n_models = 1
m = 20
H = (x)->x

for cycle=1:n_cycles
    for m=1:n_models
        x_m = mean(E, dims=2)
        X = (E .- x_m)/sqrt(m - 1)
        P_f_inv = inv(X*X')
        H_m = (x)->x
        for i=1:N_m
            E = etkf(E=E, R_inv=P_f_inv, H=H_m, y=E[:, i], Q=Q_m)
        end
    end

    E = etkf(E=E, R_inv=R_inv, H=H, y=y)
end
