using Statistics
using LinearAlgebra
using Random

using Distributions
using Plots

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
models = [Models.lorenz96_err, Models.lorenz96_err2, Models.lorenz96_err3,
          Models.lorenz96_err4]

model_errs_prescribed = [nothing, nothing, nothing, nothing]
model_true = Models.lorenz96_true
n_models = length(models)
obs_ops = [I(D), I(D), I(D), I(D)]
H = I(D)
ens_sizes = [20, 20, 20, 20]
model_sizes = [D, D, D, D]
integrators = [Integrators.rk4, Integrators.rk4, Integrators.rk4, Integrators.rk4]
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
R = Symmetric(diagm(0.25*ones(D)))
ens_errs = [Symmetric(diagm(0.25*ones(D))), Symmetric(diagm(0.25*ones(D))),
            Symmetric(diagm(0.25*ones(D))), Symmetric(diagm(0.25*ones(D)))]
gen_ensembles = false
all_orders = false
assimilate_obs = true
save_Q_hist = false
save_P_hist = false

leads = 1
x0 = x[end, :]

n_cycles = 3000*leads
ρ = 1e-3
ρ_all = 1e-2

window = 4

infos = Vector(undef, n_models)
for model=1:n_models
    Random.seed!(1)

    model_errs = [0.1*diagm(ones(D))]

    ens_size = sum(ens_sizes)
    ensembles = [x0 .+ rand(MvNormal(R), ens_size)]

    info = ens_forecast.da_cycles(x0=x0, ensembles=ensembles, models=[models[model]],
                                  model_true=model_true, obs_ops=[obs_ops[model]], H_true=H,
                                  model_errs=model_errs,
                                  model_errs_prescribed=[model_errs_prescribed[model]],
                                  integrators=integrators, integrator_true=integrator_true, da_method=da_method,
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

Random.seed!(1)

ensembles = [x0 .+ rand(MvNormal(R), ens_sizes[model]) for model=1:n_models]

model_errs = [0.1*diagm(ones(D)) for model=1:n_models]

info_mm = ens_forecast.da_cycles(x0=x0, ensembles=ensembles, models=models,
                                 model_true=model_true, obs_ops=obs_ops, H_true=H,
                                 model_errs=model_errs,
                                 model_errs_prescribed=model_errs_prescribed,
                                 integrators=integrators, integrator_true=integrator_true, da_method=da_method,
                                 localization=localization, ens_sizes=ens_sizes, Δt=Δt,
                                 window=window, n_cycles=n_cycles, outfreq=outfreq,
                                 model_sizes=model_sizes, R=R, ens_errs=ens_errs, ρ=ρ,
                                 ρ_all=ρ_all,
                                 all_orders=all_orders, combine_forecasts=true,
                                 gen_ensembles=gen_ensembles, assimilate_obs=assimilate_obs,
                                 leads=leads, save_Q_hist=save_Q_hist)

Random.seed!(1)

ensembles = [x0 .+ rand(MvNormal(R), ens_sizes[model]) for model=1:n_models]

model_errs = [0.1*diagm(ones(D)) for model=1:n_models]

info_mm_all = ens_forecast.da_cycles(x0=x0, ensembles=ensembles, models=models,
                                model_true=model_true, obs_ops=obs_ops, H_true=H,
                                model_errs=model_errs,
                                model_errs_prescribed=model_errs_prescribed,
                                integrators=integrators, integrator_true=integrator_true, da_method=da_method,
                                localization=localization, ens_sizes=ens_sizes, Δt=Δt,
                                window=window, n_cycles=n_cycles, outfreq=outfreq,
                                model_sizes=model_sizes, R=R, ens_errs=ens_errs, ρ=ρ,
                                ρ_all=ρ_all,
                                all_orders=true, combine_forecasts=true,
                                gen_ensembles=gen_ensembles, assimilate_obs=assimilate_obs,
                                leads=leads, save_Q_hist=save_Q_hist)

Random.seed!(1)

ensembles = [x0 .+ rand(MvNormal(R), ens_sizes[model]) for model=1:n_models]

model_errs = [0.1*diagm(ones(D)) for model=1:n_models]

info_mme = ens_forecast.da_cycles(x0=x0, ensembles=ensembles, models=models,
                                  model_true=model_true, obs_ops=obs_ops, H_true=H,
                                  model_errs=model_errs,
                                  model_errs_prescribed=model_errs_prescribed,
                                  integrators=integrators, integrator_true=integrator_true, da_method=da_method,
                                  localization=localization, ens_sizes=ens_sizes, Δt=Δt,
                                  window=window, n_cycles=n_cycles, outfreq=outfreq,
                                  model_sizes=model_sizes, R=R, ens_errs=ens_errs, ρ=ρ,
                                  ρ_all=ρ_all,
                                  combine_forecasts=false, gen_ensembles=gen_ensembles,
                                  assimilate_obs=assimilate_obs, leads=leads,
                                  save_Q_hist=save_Q_hist)


scatter(1:4, [mean(infos[i].crps[2000:end]) for i=1:4], thickness_scaling=1.4, legend=false)
scatter!([5], [mean(info_mme.crps[2000:end])])
scatter!([6, 7], [mean(info_mm.crps[2000:end]), mean(info_mm_all.crps[2000:end])])
xticks!(1:7, ["Model 1", "Model 2", "Model 3", "Model 4", "MME", "MM-EnKF 1", "MM-EnKF 2"], xrotation=45)
ylabel!("Analysis CRPS")
savefig("crps_analysis.pdf")

scatter(1:4, [mean(infos[i].crps_fcst[2000:end]) for i=1:4], thickness_scaling=1.4, legend=false)
scatter!([5], [mean(info_mme.crps_fcst[2000:end])])
scatter!([6, 7], [mean(info_mm.crps_fcst[2000:end]), mean(info_mm_all.crps_fcst[2000:end])])
xticks!(1:7, ["Model 1", "Model 2", "Model 3", "Model 4", "MME", "MM-EnKF 1", "MM-EnKF 2"], xrotation=45)
ylabel!("Forecast CRPS")
savefig("crps_forecast.pdf")

scatter(1:4, [sqrt(mean(infos[i].errs[2000:end, :].^2)) for i=1:4], thickness_scaling=1.4, legend=false)
scatter!([5], [sqrt(mean(info_mme.errs[2000:end, :].^2))])
scatter!([6, 7], [sqrt(mean(info_mm.errs[2000:end, :].^2)), sqrt(mean(info_mm_all.errs[2000:end, :].^2))])
xticks!(1:7, ["Model 1", "Model 2", "Model 3", "Model 4", "MME", "MM-EnKF 1", "MM-EnKF 2"], xrotation=45)
ylabel!("Analysis RMSE")
savefig("rmse_analysis.pdf")

scatter(1:4, [sqrt(mean(infos[i].errs_fcst[2000:end, :].^2)) for i=1:4], thickness_scaling=1.4, legend=false)
scatter!([5], [sqrt(mean(info_mme.errs_fcst[2000:end, :].^2))])
scatter!([6, 7], [sqrt(mean(info_mm.errs_fcst[2000:end, :].^2)), sqrt(mean(info_mm_all.errs_fcst[2000:end, :].^2))])
xticks!(1:7, ["Model 1", "Model 2", "Model 3", "Model 4", "MME", "MM-EnKF 1", "MM-EnKF 2"], xrotation=45)
ylabel!("Forecast RMSE")
savefig("rmse_forecast.pdf")
