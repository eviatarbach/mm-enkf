module ens_forecast

include("etkf.jl")

import .ETKF

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
end

xskillscore = pyimport("xskillscore")
xarray = pyimport("xarray")

function init_ens(; model::Function, integrator::Function,
                    x0::AbstractVector{float_type}, t0::float_type,
                    outfreq::int_type, Δt::float_type, ens_size::int_type,
                    transient::Real=0) where {float_type<:AbstractFloat,
                                              int_type<:Integer}
    x0 = integrator(model, x0, 0., transient*outfreq*Δt, Δt)
    E = copy(integrator(model, x0, t0, ens_size*Δt*outfreq, Δt,
                        inplace=false)[1:outfreq:end, :]')
    return E
end

function mmda(; x0::AbstractVector{float_type},
                ensembles::AbstractVector{<:AbstractMatrix{float_type}},
                models::AbstractVector{<:Function}, model_true::Function,
                obs_ops::AbstractVector{<:AbstractMatrix}, H::AbstractMatrix,
                model_errs::AbstractVector{<:Union{AbstractMatrix{float_type}, Nothing}},
                integrator::Function, ens_sizes::AbstractVector{int_type},
                Δt::float_type, window::int_type, n_cycles::int_type,
                outfreq::int_type, model_sizes::AbstractVector{int_type},
                R::AbstractMatrix{float_type}, ρ::float_type) where {float_type<:AbstractFloat, int_type<:Integer}
    n_models = length(models)
    obs_err_dist = MvNormal(R)
    R_inv = inv(R)

    x_true = x0

    errs = Array{float_type}(undef, n_models, n_cycles)
    errs_uncorr = Array{float_type}(undef, n_models, n_cycles)
    crps = Array{float_type}(undef, n_models, n_cycles)
    crps_uncorr = Array{float_type}(undef, n_models, n_cycles)
    spread = Array{float_type}(undef, n_models, n_cycles)

    t = 0.0

    for cycle=1:n_cycles
        println(cycle)

        for model=1:n_models
            E = ensembles[model]

            x_m = mean(E, dims=2)
            errs_uncorr[model, cycle] = sqrt(mean((x_m .- x_true).^2))

            E_array = xarray.DataArray(data=E, dims=["dim", "member"])
            crps_uncorr[model, cycle] = xskillscore.crps_ensemble(x_true, E_array).values[1]
        end

        y = H*x_true + rand(obs_err_dist)

        # Iterative multi-model data assimilation
        for model=2:n_models
            # Posterior ensemble of the previous model is used as the prior
            # ensemble for the next model
            E = ensembles[model-1]

            E_model = ensembles[model]

            m = ens_sizes[model]
            H_model = obs_ops[model]
            H_model_prime = H*inv(H_model)

            x_m = mean(E, dims=2)

            X = (E .- x_m)/sqrt(m - 1)
            P_f = X*X'
            P_f_inv = inv(P_f)

            if model_errs[model] !== nothing
                # Estimate model error covariance based on innovations
                d = y - H_model_prime*x_m
                Q_est = inv(H_model_prime)*(d*d' - R - H_model_prime*P_f*H_model_prime')*inv(H_model_prime)'

                # Time filtering
                Q = ρ*Q_est + (1 - ρ)*model_errs[model]
            else
                Q = nothing
            end

            # Assimilate the forecast of each ensemble member of the current
            # model as if it were an observation
            for i=1:m
                E = ETKF.etkf(E=E, R_inv=P_f_inv, H=H_model, y=E_model[:, i], Q=Q)
            end

            ensembles[model] = E
        end

        # Assimilate observations
        E_a = ETKF.etkf(E=ensembles[n_models], R_inv=R_inv, H=H, y=y)

        for model=1:n_models
            # Map from reference model space to the current model space
            E = obs_ops[model]*E_a
            x_m = mean(E, dims=2)

            errs[model, cycle] = sqrt(mean((x_m .- x_true).^2))
            E_corr_array = xarray.DataArray(data=E, dims=["dim", "member"])
            crps[model, cycle] = xskillscore.crps_ensemble(x_true, E_corr_array).values[1]
            spread[model, cycle] = mean(std(E, dims=2))

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

    return Forecast_Info(errs, errs_uncorr, crps, crps_uncorr, spread)
end

end
