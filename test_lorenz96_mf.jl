using Statistics
using LinearAlgebra
using Random

using Distributions
using BandedMatrices

include("da.jl")
import .DA

include("ens_forecast.jl")
import .ens_forecast

include("models.jl")
import .Models

include("integrators.jl")
import .Integrators

Random.seed!(1)

D = 40
models = [Models.lorenz96_true, Models.lorenz96_true]

B = brand(40, 40, 20, 20)
Q_true = Matrix((B .- 0.4)*(B .- 0.4)')/10

model_errs_prescribed = [Q_true/10, Q_true]
model_true = Models.lorenz96_true
n_models = length(models)
obs_ops = [I(D), I(D)]
H = I(D)
ens_sizes = [5, 40]
model_sizes = [D, D]
integrators = [Integrators.rk4, Integrators.rk4]
integrator_true = Integrators.rk4
da_method = DA.ensrf
localization_radius = 4
localization = DA.gaspari_cohn_localization(localization_radius, D, cyclic=true)
x0 = randn(D)
t0 = 0.0
Δt = 0.05
outfreq = 1
transient = 2000
x = integrators[1](models[1], x0, t0, transient*outfreq*Δt, Δt, inplace=false)
R = Symmetric(diagm(0.45*ones(D)))
ens_errs = [Symmetric(diagm(0.25*ones(D))), Symmetric(diagm(0.25*ones(D)))]
gen_ensembles = true
all_orders = true
assimilate_obs = false
save_Q_hist = false
save_P_hist = false

leads = 7
x0 = x[end, :]

n_cycles = 3000*leads
ρ = 1e-3
ρ_all = 1e-2

window = 4

infos = Vector(undef, n_models)
for model=1:n_models
    model_errs = [0.1*diagm(ones(D))]

    ens_size = ens_sizes[model]
    ensembles = [x0 .+ rand(MvNormal(R), ens_size)]

    info = ens_forecast.da_cycles(x0=x0, ensembles=ensembles, models=[models[model]],
                                  model_true=model_true, obs_ops=[obs_ops[model]], H_true=H,
                                  model_errs=model_errs,
                                  model_errs_prescribed=[model_errs_prescribed[model]],
                                  integrators=integrators, integrator_true=integrator_true,
                                  da_method=da_method,
                                  localization=localization, ens_sizes=[ens_size], Δt=Δt,
                                  window=window, n_cycles=n_cycles, outfreq=outfreq,
                                  model_sizes=model_sizes, R=R, ens_errs=ens_errs, ρ=ρ,
                                  ρ_all=ρ_all,
                                  gen_ensembles=gen_ensembles,
                                  assimilate_obs=assimilate_obs, save_analyses=false,
                                  leads=leads, save_Q_hist=save_Q_hist,
                                  save_P_hist=save_P_hist)
    infos[model] = info
end

ensembles = [x0 .+ rand(MvNormal(R), ens_sizes[model]) for model=1:n_models]

model_errs = [0.1*diagm(ones(D)) for model=1:n_models]

info_mm = ens_forecast.da_cycles(x0=x0, ensembles=ensembles, models=models,
                                 model_true=model_true, obs_ops=obs_ops, H_true=H,
                                 model_errs=model_errs,
                                 model_errs_prescribed=model_errs_prescribed,
                                 integrators=integrators, integrator_true=integrator_true,
                                 da_method=da_method,
                                 localization=localization, ens_sizes=ens_sizes, Δt=Δt,
                                 window=window, n_cycles=n_cycles, outfreq=outfreq,
                                 model_sizes=model_sizes, R=R, ens_errs=ens_errs, ρ=ρ,
                                 ρ_all=ρ_all,
                                 all_orders=all_orders, combine_forecasts=true,
                                 gen_ensembles=gen_ensembles, assimilate_obs=assimilate_obs,
                                 leads=leads, save_Q_hist=save_Q_hist)

ensembles = [x0 .+ rand(MvNormal(R), ens_sizes[model]) for model=1:n_models]

model_errs = [0.1*diagm(ones(D)) for model=1:n_models]

info_mm2 = ens_forecast.da_cycles(x0=x0, ensembles=ensembles, models=models,
                                  model_true=model_true, obs_ops=obs_ops, H_true=H,
                                  model_errs=model_errs,
                                  model_errs_prescribed=model_errs_prescribed,
                                  integrators=integrators, integrator_true=integrator_true,
                                  da_method=da_method,
                                  localization=localization, ens_sizes=ens_sizes, Δt=Δt,
                                  window=window, n_cycles=n_cycles, outfreq=outfreq,
                                  model_sizes=model_sizes, R=R, ens_errs=ens_errs, ρ=ρ,
                                  ρ_all=ρ_all,
                                  combine_forecasts=false, gen_ensembles=gen_ensembles,
                                  assimilate_obs=assimilate_obs, leads=leads,
                                  save_Q_hist=save_Q_hist)