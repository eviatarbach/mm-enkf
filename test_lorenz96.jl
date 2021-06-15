using Statistics
using LinearAlgebra
using Random

using Distributions

include("da.jl")
import .DA

include("ens_forecast.jl")
import .ens_forecast

include("models.jl")
import .Models

include("integrators.jl")
import .Integrators

using BandedMatrices

Random.seed!(1)

D = 40
models = [Models.lorenz96_err.func, Models.lorenz96_err2.func, Models.lorenz96_err3.func,
Models.lorenz96_err4.func][1:2]
#models = [Models.lorenz96_true.func, Models.lorenz96_true.func]
#C = brand(40,40,20,20) .- 0.4
diag1 = 0.1*ones(D)
diag1[1:20] .+= 0.4
diag2 = 0.1*ones(D)
diag2[21:40] .+= 0.4
#model_errs_prescribed = [diagm(diag1), diagm(diag2)]
model_errs_prescribed = [nothing, nothing, nothing, nothing]#[Matrix(C*C')/10, Matrix(C*C')]
model_true = Models.lorenz96_true.func
orders = [[1, 2, 3, 4], [2, 1, 3, 4], [3, 1, 2, 4], [4, 1, 2, 3]][1:2]
n_models = length(models)
obs_ops = [I(D), I(D), I(D), I(D)]
H = I(D)
ens_sizes = [20, 20, 20, 20]
model_sizes = [D, D, D, D]
integrator = Integrators.rk4
da_method = DA.ensrf
localization = DA.gaspari_cohn(5, D)
x0 = randn(D)
t0 = 0.0
Δt = 0.05
outfreq = 1
window = 1
transient = 2000
x0 = integrator(models[1], x0, t0, transient*outfreq*Δt, Δt, inplace=false)
R = Symmetric(diagm(0=>0.25*ones(D)))
ens_err = Symmetric(diagm(0=>0.04*ones(D)))
fcst = false
x0 = x0[end, :]
#ensembles = [ens_forecast.init_ens(model=models[model], integrator=integrator,
#                                   x0=x0, t0=t0, outfreq=outfreq, Δt=Δt,
#                                   ens_size=ens_sizes[model]) for model=1:n_models]
#x0 = ensembles[1][:, end]
n_cycles = 5000
#spinup = 14600
ρ = 1e-4

infos = Vector(undef, n_models)
for model=1:length(models)
    model_errs = [0.1*diagm(0=>ones(D))]#Vector{Matrix{Float64}}(undef, 1)]
    biases = [zeros(D)]

    ensembles = [x0 .+ rand(MvNormal(R), cumsum(ens_sizes)[n_models])]

    info, _, _ = ens_forecast.mmda(x0=x0, ensembles=ensembles,
                         models=[models[model]],
                         model_true=model_true, orders=[orders[1]],
                         obs_ops=[obs_ops[model]], H=H,
                         model_errs=model_errs,
                         model_errs_prescribed=[model_errs_prescribed[model]],
                         biases=biases, integrator=integrator, da_method=da_method,
                         localization=localization,
                         ens_sizes=[cumsum(ens_sizes)[n_models]], Δt=Δt, window=window,
                         n_cycles=n_cycles, outfreq=outfreq,
                         model_sizes=model_sizes, R=R, ens_err=ens_err,
                         ρ=ρ, fcst=fcst, save_analyses=true)
    infos[model] = info
end

#incs1 = hcat(info1.increments...)
#incs2 = hcat(info2.increments...)
#model_errs = [cov(incs1'), cov(incs2')]
#biases = [mean(incs1, dims=2)[:], mean(incs2, dims=2)[:]]
biases = [zeros(D), zeros(D), zeros(D), zeros(D)]#[mean(infos[1].bias_hist[1000:end]), mean(infos[2].bias_hist[1000:end])]
#[zeros(D), zeros(D)]
ensembles = [x0 .+ rand(MvNormal(R), ens_sizes[model]) for model=1:n_models]

model_errs = [0.1*diagm(0=>ones(D)) for model=1:n_models]#[mean(infos[1].Q_hist[1000:end]),
             # mean(infos[2].Q_hist[1000:end])]

info_mm, _, _ = ens_forecast.mmda(x0=x0, ensembles=ensembles, models=models,
                         model_true=model_true, orders=orders, obs_ops=obs_ops, H=H,
                         model_errs=model_errs, model_errs_prescribed=model_errs_prescribed,
                         biases=biases, integrator=integrator, da_method=da_method,
                         localization=localization,
                         ens_sizes=ens_sizes, Δt=Δt, window=window,
                         n_cycles=n_cycles, outfreq=outfreq,
                         model_sizes=model_sizes, R=R, ens_err=ens_err,
                         ρ=ρ, fixed=false, fcst=fcst)#, prev_analyses=infos[1].analyses)

biases = [zeros(D), zeros(D), zeros(D), zeros(D)]#[mean(infos[1].bias_hist[1000:end]), mean(infos[2].bias_hist[1000:end])]
#[zeros(D), zeros(D)]
ensembles = [x0 .+ rand(MvNormal(R), ens_sizes[model]) for model=1:n_models]

model_errs = [0.1*diagm(0=>ones(D)) for model=1:n_models]#[mean(infos[1].Q_hist[1000:end]),
            # mean(infos[2].Q_hist[1000:end])]

info_mm2, _, _ = ens_forecast.mmda(x0=x0, ensembles=ensembles, models=models,
                         model_true=model_true, orders=orders, obs_ops=obs_ops, H=H,
                         model_errs=model_errs, model_errs_prescribed=model_errs_prescribed,
                         biases=biases, integrator=integrator, da_method=da_method,
                         localization=localization,
                         ens_sizes=ens_sizes, Δt=Δt, window=window,
                         n_cycles=n_cycles, outfreq=outfreq,
                         model_sizes=model_sizes, R=R, ens_err=ens_err,
                         ρ=ρ, fixed=false, mmm=true, fcst=fcst)