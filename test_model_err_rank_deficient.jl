using Serialization

using Statistics
using LinearAlgebra
using Random

using Distributions
using BlockArrays
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
models = [Models.lorenz96_true.func]
B = brand(40, 40, 20, 20)
Q_true = Matrix((B .- 0.4)*(B .- 0.4)')/10
model_errs_prescribed = [Q_true]
model_true = Models.lorenz96_true.func
n_models = length(models)
obs_ops = [diagm(ones(D))[1:2:D, :]]
mappings = Matrix{AbstractArray}(undef, n_models, n_models)
mappings[1, 1] = I(D)

H_true = diagm(ones(D))[1:2:D, :]
ens_sizes = [80]
model_sizes = [D]
integrator = Integrators.rk4
da_method = DA.etkf
localization = DA.gaspari_cohn(4, D)
x0 = randn(D)
t0 = 0.0
Δt = 0.05
outfreq = 1
transient = 2000
x0 = integrator(models[1], x0, t0, transient*outfreq*Δt, Δt, inplace=false)
R = Symmetric(diagm(0=>0.4*ones(20)))
ens_errs = [Symmetric(diagm(0=>0.4*ones(D)))]
gen_ensembles = false
assimilate_obs = true
all_orders = false
leads = 1
ref_model = 1
x0 = x0[end, :]

n_cycles = 20000*leads
ρ = 1e-4

save_Q_hist = true

window = 1

model_errs = [diagm(0=>ones(D))]
biases = [zeros(D)]

ensembles = [x0 .+ rand(MvNormal(ens_errs[1]), ens_sizes[1])]

all_Q_p = []
for i=1:8
    for j=1:8
        Q_p = mortar(reshape([zeros(5, 5) for i=1:64], 8, 8))
        Q_p[Block(i, j)] = ones(5, 5)
        append!(all_Q_p, [Matrix(Q_p)])
    end
    #append!(all_Q_p, diagm([zeros(p-1); 1; zeros(D-p)]))
end

info, _, _ = ens_forecast.mmda(x0=x0, ensembles=ensembles, models=models,
                               model_true=model_true, obs_ops=obs_ops,
                               H_true=H_true, model_errs=model_errs,
                               model_errs_prescribed=model_errs_prescribed,
                               biases=biases, integrator=integrator,
                               da_method=da_method, localization=localization,
                               ens_sizes=ens_sizes, Δt=Δt, window=window,
                               n_cycles=n_cycles, outfreq=outfreq,
                               model_sizes=model_sizes, R=R, ens_errs=ens_errs,
                               ρ=ρ, Q_p=all_Q_p, gen_ensembles=gen_ensembles,
                               assimilate_obs=assimilate_obs,
                               save_analyses=false, leads=leads,
                               save_Q_hist=save_Q_hist, mappings=mappings)