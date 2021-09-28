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

D1 = 220
D2 = 20
models = [Models.lorenz96_twoscale_err, Models.lorenz96_half_true]
model_errs_prescribed = [nothing, nothing, nothing, nothing]
model_true = Models.lorenz96_twoscale_true
n_models = length(models)
dd = zeros(D1)
dd[1:11:D1] .= 1
I1to2 = diagm(dd)[Vector{Bool}(dd), :]
I2to1 = I1to2'
obs_ops = [I(D1), I2to1]
mappings = Matrix{AbstractArray}(undef, n_models, n_models)
mappings[1, 1] = I(D1)
mappings[1, 2] = I1to2
mappings[2, 1] = I2to1
mappings[2, 2] = I(D2)

H = I(D1)
ens_sizes = [20, 20]
model_sizes = [D1, D2]
integrator = Integrators.rk4
da_method = DA.ensrf
localization = diagm(ones(D1))

indices = reshape(1:220, 11, :)
first_layer_indices = indices[1, :]
second_layer_indices = indices[2:end, :]

c = 4
for (ii, i) in enumerate(first_layer_indices)
    for (ij, j) in enumerate(first_layer_indices)
        r = min(mod(ii - ij, 0:20), mod(ij - ii, 0:20))/c
        localization[i, j] = DA.gaspari_cohn(r)
        localization[j, i] = DA.gaspari_cohn(r)
    end
end

for i=1:20
    layer_indices = indices[:, i]
    for j=layer_indices
        for k=layer_indices
            localization[j, k] = 1
            localization[k, j] = 1
        end
    end
end

x0 = rand(D1)
t0 = 0.0
Δt = 0.005
outfreq = 1
transient = 2000
x = integrator(models[1], x0, t0, transient*outfreq*Δt, Δt, inplace=false)
R = Symmetric(diagm(var(x, dims=1)[:]*0.01))
ens_errs = [R, Symmetric(diagm(0.25*ones(D2)))]
gen_ensembles = false
assimilate_obs = true
all_orders = false
save_Q_hist = false

leads = 1
ref_model = 1
x0 = x[end, :]

n_cycles = 500*leads
ρ = 1e-3

window = 10

infos = Vector(undef, n_models)
for model=1:n_models
    model_errs = [ens_errs[model]]

    ens_size = ens_sizes[model]
    ensembles = [mappings[ref_model, model]*x0 .+ rand(MvNormal(ens_errs[model]), ens_size)]

    info = ens_forecast.da_cycles(x0=x0, ensembles=ensembles, models=[models[model]],
                                  model_true=model_true, obs_ops=[obs_ops[model]], H_true=I,
                                  model_errs=model_errs,
                                  model_errs_prescribed=[model_errs_prescribed[model]],
                                  integrator=integrator, da_method=da_method,
                                  localization=mappings[ref_model, model]*localization*mappings[ref_model, model]',
                                  ens_sizes=[ens_size], Δt=Δt, window=window,
                                  n_cycles=n_cycles, outfreq=outfreq,
                                  model_sizes=[model_sizes[model]], R=R,
                                  ens_errs=[ens_errs[model]], ρ=ρ,
                                  gen_ensembles=gen_ensembles,
                                  assimilate_obs=assimilate_obs, save_analyses=false,
                                  leads=leads, save_Q_hist=save_Q_hist,
                                  mappings=mappings[model:model, model:model])
    infos[model] = info
end

ensembles = [mappings[ref_model, model]*x0 .+ rand(MvNormal(ens_errs[model]), ens_sizes[model]) for model=1:n_models]

model_errs = ens_errs

info_mm = ens_forecast.da_cycles(x0=x0, ensembles=ensembles, models=models,
                                 model_true=model_true, obs_ops=obs_ops,
                                 model_errs=model_errs,
                                 model_errs_prescribed=model_errs_prescribed,
                                 integrator=integrator, da_method=da_method,
                                 localization=localization, ens_sizes=ens_sizes, Δt=Δt,
                                 window=window, n_cycles=n_cycles, outfreq=outfreq,
                                 model_sizes=model_sizes, R=R, ens_errs=ens_errs, ρ=ρ,
                                 all_orders=all_orders, combine_forecasts=true,
                                 gen_ensembles=gen_ensembles, assimilate_obs=assimilate_obs,
                                 leads=leads, save_Q_hist=save_Q_hist, ref_model=ref_model,
                                 mappings=mappings)