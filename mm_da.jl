using Statistics
using LinearAlgebra
using Random

include("etkf.jl")
import .ETKF

include("ens_forecast.jl")
import .ens_forecast

include("models.jl")
import .Models

include("integrators.jl")
import .Integrators

Random.seed!(1)

models = [Models.lorenz63_err, Models.lorenz63_err2]
model_true = Models.lorenz63_true
n_models = length(models)
D = 3
obs_ops = [I(D), I(D)]
H = I(D)
R = diagm(0=>1*ones(D))
ens_sizes = [20, 20]
model_sizes = [D, D]
integrator = Integrators.rk4
x0 = rand(D)
t0 = 0.0
Δt = 0.05
outfreq = 5
window = 5
transient = 200
x0 = integrator(models[1], x0, t0, transient*outfreq*Δt, Δt)
ensembles = [ens_forecast.init_ens(model=models[model], integrator=integrator,
                                   x0=x0, t0=t0, outfreq=outfreq, Δt=Δt,
                                   ens_size=ens_sizes[model]) for model=1:n_models]
x0 = ensembles[1][:, end]
n_cycles = 100
ρ = 0.0

model_errs = [ens_forecast.model_err(model_true=model_true, model_err=models[model],
                                     integrator=integrator, x0=x0, t0=t0,
                                     outfreq=outfreq, Δt=Δt, window=window,
                                     n_samples=100) for model=1:n_models]

info = ens_forecast.mmda(x0=x0, ensembles=ensembles, models=models,
                         model_true=model_true, obs_ops=obs_ops, H=H,
                         model_errs=model_errs, integrator=integrator,
                         ens_sizes=ens_sizes, Δt=Δt, window=window,
                         n_cycles=n_cycles, outfreq=outfreq,
                         model_sizes=model_sizes, R=R, ρ=ρ)