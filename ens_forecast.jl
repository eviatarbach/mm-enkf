module ens_forecast

include("etkf.jl")

using .ETKF

export init_ens, mmda

using Statistics
using LinearAlgebra
using Random

using NearestNeighbors
using Distributions
using PyCall

struct Forecast_Info
    errs
    errs_uncorr
    crps
    crps_uncorr
    spread
    x_trues
    errs_m
end

xskillscore = pyimport("xskillscore")
xarray = pyimport("xarray")

function init_ens(; model, integrator, x0, t0, outfreq, Δt, ens_size,
                  transient=0)
    x0 = integrator(model, x0, 0., transient*outfreq*Δt, Δt)
    E = integrator(model, x0, t0, ens_size*Δt*outfreq, Δt,
                   inplace=false)[1:outfreq:end, :]'
    return E
end

function mmda(; x0, ensembles::Vector{Array{float_type, 2}}, models, model_true, obs_ops, H, model_errs,
                integrator,
                ens_sizes::Vector{Integer}, Δt::float_type, window::Integer, cycles::Integer,
                outfreq::Integer, D::Integer, k, means, R, ρ) where {float_type<:AbstractFloat}
    n_models = length(models)
    obs_err_dist = MvNormal(R)
    R_inv = inv(R)

    x_true = x0

    errs = []
    errs_uncorr = []
    crps = []
    crps_uncorr = []
    x_trues = []
    errs_m = []
    spread = []

    t = 0.0

    for cycle=1:cycles
        println(cycle)

        y = H(x_true) + rand(obs_err_dist)

        # Iterative multi-model data assimilation
        for model=2:n_models
            # Posterior ensemble of the previous model is used as the prior
            # ensemble for the next model
            E = ensembles[model-1]

            m = ens_sizes[model]
            H_model = obs_ops[model]
            H_model_prime = H*inv(H_model)

            x_m = mean(E, dims=2)

            X = (E .- x_m)/sqrt(m - 1)
            P_f = X*X'
            P_f_inv = inv(P_f)

            # Estimate model error covariance based on innovations
            d = y - H_model_prime*x_m
            Q_est = inv(H_model_prime)*(d*d' - R - H_model_prime*P_f*H_model_prime')*inv(H_model_prime)'

            # Time filtering
            Q = ρ*Q_est + (1 - ρ)*model_errs[model]

            # Assimilate the forecast of each ensemble member of the current
            # model as if it were an observation
            for i=1:m
                E = etkf(E=E, R_inv=P_f_inv, H=H_model, y=E[:, i], Q=Q)
            end

            append!(errs_uncorr, sqrt(mean((x_m .- x_true).^2)))

            E_array = xarray.DataArray(data=E, dims=["dim", "member"])
            E_corr_array = xarray.DataArray(data=E, dims=["dim", "member"])
            x_m = mean(E, dims=2)
            append!(errs, sqrt(mean((x_m .- x_true).^2)))
            append!(crps, xskillscore.crps_ensemble(x_true, E_corr_array).values[1])
            append!(crps_uncorr, xskillscore.crps_ensemble(x_true, E_array).values[1])

            append!(x_trues, x_true)
            append!(errs_m, sqrt(mean((means .- x_true).^2)))

            ens_spread = mean(std(E, dims=2))
            append!(spread, ens_spread)

            x_m = mean(E, dims=2)

            ensembles[model] = E
        end

        # Assimilate observations
        E_a = etkf(E=ensembles[n_models], R_inv=R_inv, H=H, y=y)

        # Integrate every model
        for model=1:n_models
            # Map from reference model space to the current model space
            E = obs_ops[model]*E_a

            for i=1:ens_sizes[model]
                integration = integrator(models[model], E[:, i], t,
                                         t + window*outfreq*Δt, Δt, inplace=false)
                E[:, i] = integration[end, :]
            end
            ensembles[model] = E
        end

        x_true = integrator(model_true, x_true, t, t + window*outfreq*Δt, Δt)

        t += window*outfreq*Δt
    end

    x_trues = reshape(x_trues, D, :)
    return Forecast_Info(errs, errs_uncorr, crps, crps_uncorr, spread,
                         x_trues, errs_m)
end

end
