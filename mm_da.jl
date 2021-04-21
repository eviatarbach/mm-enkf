using Statistics

include("etkf.jl")
using .ETKF

include("ens_forecast.jl")
using .ens_forecast

include("models.jl")
using .Models

include("integrators.jl")
using .Integrators

models = [Models.lorenz63_true, Models.lorenz63_err]
model_true = Models.lorenz63_true
n_models = length(models)
obs_ops = [identity, identity]
ens_sizes = [20, 20]
integrator = rk4
x0 = rand(3)
t0 = 0.0
Δt = 0.05
outfreq = 5
transient = 200
ensembles = [init_ens(model=models[model], integrator=integrator, x0=x0, t0=t0,
                      outfreq=outfreq, Δt=Δt, ens_size=ens_sizes[model],
                      transient=transient) for model=1:n_models]

n_cycles = 100

info = mmda()


# for cycle=1:n_cycles
#     for model=1:n_models-1
#         x_m = mean(E, dims=2)
#         X = (E .- x_m)/sqrt(m - 1)
#         P_f_inv = inv(X*X')
#         H_m = H
#         N_m = m
#         for i=1:N_m
#             ensembles[model] = etkf(E=ensembles[model], R_inv=P_f_inv, H=H_m, y=E[:, i], Q=Q_m)
#         end
#     end

#     E = etkf(E=E, R_inv=R_inv, H=H, y=y)
# end