using Statistics
using LinearAlgebra

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
D = 3
obs_ops = [identity, identity]
H = identity
R = diagm(0=>0.1*ones(D))
model_errs = [zeros(D, D), zeros(D, D)]
ens_sizes = [20, 20]
model_sizes = [D, D]
integrator = rk4
x0 = rand(D)
t0 = 0.0
Δt = 0.05
outfreq = 5
window = 10
transient = 200
ensembles = [init_ens(model=models[model], integrator=integrator, x0=x0, t0=t0,
                      outfreq=outfreq, Δt=Δt, ens_size=ens_sizes[model],
                      transient=transient) for model=1:n_models]
n_cycles = 100
ρ = 0.5

info = mmda(x0, ensembles=ensembles, models=models, model_true=model_true,
            obs_ops=obs_ops, H=H, model_errs=model_errs, integrator=integrator,
            ens_sizes=ens_sizes, Δt=Δt, window=window, n_cycles=n_cycles,
            outfreq=outfreq, model_sizes=model_sizes, R=R, ρ=ρ)