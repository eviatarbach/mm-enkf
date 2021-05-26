using Statistics
using LinearAlgebra
using Random

using Distributions

include("etkf.jl")
import .ETKF

include("ens_forecast.jl")
import .ens_forecast

include("models.jl")
import .Models

include("integrators.jl")
import .Integrators

Random.seed!(1)

models = [Models.lorenz96_err.func, Models.lorenz96_err2.func,
Models.lorenz96_err3.func, Models.lorenz96_err4.func]
model_true = Models.lorenz96_true.func
orders = [[1, 2, 3, 4], [2, 1, 3, 4], [3, 1, 2, 4], [4, 1, 2, 3]]
n_models = length(models)
D = 40
obs_ops = [I(D), I(D), I(D), I(D)]
H = I(D)
ens_sizes = [10, 10, 10, 10]
model_sizes = [D, D, D, D]
integrator = Integrators.rk4
x0 = randn(D)
t0 = 0.0
Δt = 0.05
outfreq = 1
window = 12
transient = 2000
x0 = integrator(models[1], x0, t0, transient*outfreq*Δt, Δt, inplace=false)
R = Symmetric(diagm(0=>0.04*ones(D)))
x0 = x0[end, :]
#ensembles = [ens_forecast.init_ens(model=models[model], integrator=integrator,
#                                   x0=x0, t0=t0, outfreq=outfreq, Δt=Δt,
#                                   ens_size=ens_sizes[model]) for model=1:n_models]
#x0 = ensembles[1][:, end]
n_cycles = 2000
#spinup = 14600
ρ = 1e-3

infos = Vector(undef, n_models)
for model=1:length(models)
    model_errs = [1*diagm(0=>ones(D))]#Vector{Matrix{Float64}}(undef, 1)]
    biases = [zeros(D)]

    ensembles = [x0 .+ rand(MvNormal(R), 40)]

    info, _, _ = ens_forecast.mmda(x0=x0, ensembles=ensembles,
                         models=[models[model]],
                         model_true=model_true, orders=[orders[1]],
                         obs_ops=[obs_ops[model]], H=H,
                         model_errs=model_errs, biases=biases, integrator=integrator,
                         ens_sizes=[40], Δt=Δt, window=window,
                         n_cycles=n_cycles, outfreq=outfreq,
                         model_sizes=model_sizes, R=R, ρ=ρ)
    infos[model] = info
end


#incs1 = hcat(info1.increments...)
#incs2 = hcat(info2.increments...)
#model_errs = [cov(incs1'), cov(incs2')]
#biases = [mean(incs1, dims=2)[:], mean(incs2, dims=2)[:]]
biases = [zeros(D), zeros(D), zeros(D), zeros(D)]#[mean(infos[1].bias_hist[1000:end]), mean(infos[2].bias_hist[1000:end])]
#[zeros(D), zeros(D)]
ensembles = [x0 .+ rand(MvNormal(R), ens_sizes[model]) for model=1:n_models]

model_errs = [1*diagm(0=>ones(D)) for model=1:n_models]#[mean(infos[1].Q_hist[1000:end]),
             # mean(infos[2].Q_hist[1000:end])]


info_mm, _, _ = ens_forecast.mmda(x0=x0, ensembles=ensembles, models=models,
                         model_true=model_true, orders=orders, obs_ops=obs_ops, H=H,
                         model_errs=model_errs, biases=biases, integrator=integrator,
                         ens_sizes=ens_sizes, Δt=Δt, window=window,
                         n_cycles=n_cycles, outfreq=outfreq,
                         model_sizes=model_sizes, R=R, ρ=ρ, fixed=false)