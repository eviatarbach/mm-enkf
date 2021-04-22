using Statistics
using LinearAlgebra

include("etkf.jl")
import .ETKF

include("ens_forecast.jl")
import .ens_forecast

include("models.jl")
import .Models

include("integrators.jl")
import .Integrators

models = [Models.lorenz63_true, Models.lorenz63_err]
model_true = Models.lorenz63_true
n_models = length(models)
D = 3
obs_ops = [I(D), I(D)]
H = I(D)
R = diagm(0=>0.1*ones(D))
model_errs = [0.01*I(D), 0.01*I(D)]
ens_sizes = [20, 20]
model_sizes = [D, D]
integrator = Integrators.rk4
x0 = rand(D)
t0 = 0.0
Δt = 0.05
outfreq = 5
window = 10
transient = 200
ensembles = [ens_forecast.init_ens(model=models[model], integrator=integrator, x0=x0, t0=t0,
                      outfreq=outfreq, Δt=Δt, ens_size=ens_sizes[model],
                      transient=transient) for model=1:n_models]
n_cycles = 100
ρ = 0.0

info = ens_forecast.mmda(x0=x0, ensembles=ensembles, models=models, model_true=model_true,
            obs_ops=obs_ops, H=H, model_errs=model_errs, integrator=integrator,
            ens_sizes=ens_sizes, Δt=Δt, window=window, n_cycles=n_cycles,
            outfreq=outfreq, model_sizes=model_sizes, R=R, ρ=ρ)