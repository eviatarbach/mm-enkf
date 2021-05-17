module ens_forecast

include("etkf.jl")

import .ETKF

export model_err, init_ens, mmda

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
    increments
    errs_mmm
    Q_hist
end

xskillscore = pyimport("xskillscore")
xarray = pyimport("xarray")

function model_err(; model_true::Function, model_err::Function,
                     integrator::Function, x0::AbstractVector{float_type},
                     t0::float_type, outfreq::Integer, Δt::float_type,
                     window::Integer, n_samples::Integer) where {float_type<:AbstractFloat}
    D = length(x0)

    errs = Array{float_type}(undef, n_samples, D)

    t = t0
    x = x0
    for i=1:n_samples
        x_true = integrator(model_true, x, t, t + window*outfreq*Δt, Δt)
        x_err = integrator(model_err, x, t, t + window*outfreq*Δt, Δt)
        errs[i, :] = x_err - x_true
        t += window*outfreq*Δt
        x = x_true
    end
    return errs'*errs/(n_samples - 1), mean(errs, dims=1)
end

function init_ens(; model::Function, integrator::Function,
                    x0::AbstractVector{float_type}, t0::float_type,
                    outfreq::int_type, Δt::float_type,
                    ens_size::int_type) where {float_type<:AbstractFloat,
                                               int_type<:Integer}
    E = copy(integrator(model, x0, t0, ens_size*Δt*outfreq, Δt,
                        inplace=false)[1:outfreq:end, :]')
    return E
end

function make_psd(A)
    L, Q = eigen(A)
    L[L .< 0] .= 1e-6
    return Symmetric(Q*diagm(0=>L)*inv(Q))
end

function mmda(; x0::AbstractVector{float_type},
                ensembles::AbstractVector{<:AbstractMatrix{float_type}},
                models::AbstractVector{<:Function}, model_true::Function,
                model_jacs::AbstractVector{<:Function},
                obs_ops::AbstractVector{<:AbstractMatrix}, H::AbstractMatrix,
                model_errs::AbstractVector{<:Union{AbstractMatrix{float_type}, Nothing}},
                biases::AbstractVector{<:Union{AbstractVector{float_type}, Nothing}},
                integrator::Function, integrator_prop::Function,
                ens_sizes::AbstractVector{int_type},
                Δt::float_type, window::int_type, n_cycles::int_type,
                outfreq::int_type, model_sizes::AbstractVector{int_type},
                R::AbstractMatrix{float_type}, ρ::float_type, inflations::AbstractVector{float_type},
                α::float_type, mmm::Bool=false) where {float_type<:AbstractFloat, int_type<:Integer}
    n_models = length(models)
    obs_err_dist = MvNormal(R)
    R_inv = inv(R)

    x_true = x0

    ensembles_a = similar(ensembles)
    errs = Array{float_type}(undef, n_models, n_cycles)
    errs_mmm = Array{float_type}(undef, n_cycles)
    errs_uncorr = Array{float_type}(undef, n_models, n_cycles)
    crps = Array{float_type}(undef, n_models, n_cycles)
    crps_uncorr = Array{float_type}(undef, n_models, n_cycles)
    increments = Array{Vector{float_type}}(undef, n_models, n_cycles)
    innovations = Array{Vector{float_type}}(undef, n_models, n_cycles)
    propagators = Array{Matrix{float_type}}(undef, n_models)
    Q_hist = Array{Matrix{float_type}}(undef, n_models, n_cycles)
    spread = Array{float_type}(undef, n_models, n_cycles)
    P_past = Matrix{float_type}(undef, model_sizes[1], model_sizes[1])
    avg = 0.0

    t = 0.0

    for cycle=1:n_cycles
        println(cycle)

        mm_mean = mean(hcat(ensembles...), dims=2)
        errs_mmm[cycle] = sqrt(mean((mm_mean - x_true).^2))
        for model=1:n_models
            E = ensembles[model]

            x_m = mean(E, dims=2)
            errs_uncorr[model, cycle] = sqrt(mean((x_m .- x_true).^2))

            E_array = xarray.DataArray(data=E, dims=["dim", "member"])
            crps_uncorr[model, cycle] = xskillscore.crps_ensemble(x_true, E_array).values[1]
        end

        y = H*x_true + rand(obs_err_dist)

        for model=1:n_models
            x_f = mean(ensembles[model], dims=2)
            # Assimilate observations
            ensembles_a[model] = ETKF.etkf(E=ensembles[model], R_inv=R_inv, H=H, y=y)
            increments[model, cycle] = (x_f - mean(ensembles_a[model], dims=2))[:]
            innovations[model, cycle] = (y - x_f)[:]
        end

        if cycle > 50
            for model=1:n_models
                E = ensembles[model]
                E_a = ensembles_a[model]
                m = ens_sizes[model]
                propagator = propagators[model]

                P_est = (inv(propagator)*innovations[model, cycle]*innovations[model, cycle - 1]'
                         + increments[model, cycle - 1]*innovations[model, cycle - 1]')
                P_true = (E .- x_true)*(E .- x_true)'/(m - 1)
                P_e = (E .- y)*(E .- y)'/(m - 1)
                #R_est = (E_a .- y)*(E .- y)'/(m - 1)

                x_m = mean(E, dims=2)
                X = (E .- x_m)/sqrt(m - 1)
                P_f = X*X'

                #P_e = (x_m - y)*(x_m - y)'
                #A = zeros(model_sizes[model]^2, model_sizes[model])
                #for p=1:model_sizes[model]
                #    e = zeros(model_sizes[model], model_sizes[model])
                #    e[p, p] = H[p, p]
                #    A[:, p] = vec(e)
                #end
                #q = A \ vec(P_e - R - P_f)
                #Q_est = diagm(0=>q)

                #Q_est = (P_e - P_f)*tr(P_e - R - P_f)/tr(P_e - P_f)
                #Q_est = diagm(0=>diag(P_e - R - P_f))
                avg += tr(P_f)/tr(P_e)
                #println(avg/cycle)
                #println(tr(P_f + R)/tr(P_e))
                #println(Q_est - Q_est2)
                #Q_est = P_est - P_f
                #Q_est = P_e - P_f - diagm(0=>diag(R_est))
                #println(tr(Q_est)/tr(P_true))
                #Q_est_n = P_f - propagator*P_past*propagator'
                Q_est = diagm(0=>diag(P_est - propagator*P_past*propagator'))

                #println(R_est)
                println(Q_est)
                #Q_est = make_psd(Q_est)
                #println(norm(Q_est - Q_est_n)/norm(Q_est_n))
                #println(cor(Q_true[:], Q_est[:]))
                #println(minimum(eigvals(P_e - R - P_f)).^2/sum(eigvals(P_e - R - P_f).^2))

                if cycle <= 51
                    Q = Q_est
                else
                    if isposdef(Q_est)
                        Q = ρ*Q_est + (1 - ρ)*model_errs[model]
                    else
                        Q = model_errs[model]
                    end
                end

                #Q = make_psd(Q)
                #Q += 1e-5*I  # to make positive definite
                Q_hist[model, cycle] = Q
                model_errs[model] = Q

                # H_model = obs_ops[model]
                # H_model_prime = H*inv(H_model)

                # x_m = mean(E_model, dims=2)

                # if cycle > 1
                #     propagator = propagators[model]

                #     X = (E_model .- x_m)/sqrt(m - 1)
                #     P_f = X*X'

                #     d_old = innovations[model]
                #     d = y - H_model_prime*x_m
                #     innovations[model] = d

                #     println((eigvals(P_f)))
                # else
                #     innovations[model] = y - H_model_prime*x_m
                # end
                #println(tr(d*d') - tr(R + H_model_prime*P_f*H_model_prime'))

                #if cycle > 1
                    #if model_errs[model] !== nothing
                #        E -= α*rand(MvNormal(biases[model], model_errs[model]), ens_sizes[model])
                    #end
                #end
                ensembles[model] = E
            end
        end

        # Iterative multi-model data assimilation
        if ~mmm
            for model=2:n_models
                # Posterior ensemble of the previous model is used as the prior
                # ensemble for the next model
                E = ensembles[model-1]
                #E += rand(MvNormal(model_errs[model-1]), ens_sizes[model-1])

                E_model = ensembles[model]

                m = ens_sizes[model]
                H_model = obs_ops[model]
                H_model_prime = H*inv(H_model)

                x_m = mean(E_model, dims=2)

                X = (E_model .- x_m)/sqrt(m - 1)
                P_f = X*X'# + model_errs[model]
                P_f_diag = diagm(0=>diag(P_f))
                P_f_inv = inv(P_f_diag)

                #d = y - H_model_prime*x_m
                #λ_est = (d'*d - tr(R))/tr(H_model_prime*P_f*H_model_prime')

                # Time filtering
                # λ = ρ*λ_est + (1 - ρ)*inflations[model]

                #  if model_errs[model] !== nothing
                # #     # Estimate model error covariance based on innovations
                #      d = y - H_model_prime*x_m
                #      Q_est = inv(H_model_prime)*(d*d' - R - H_model_prime*P_f*H_model_prime')*inv(H_model_prime)'

                #      # Time filtering
                #      Q = ρ*Q_est + (1 - ρ)*model_errs[model-1]
                #      #println(minimum(eigvals(Q_est)))
                #  else
                #      Q = nothing
                #  end

                # Assimilate the forecast of each ensemble member of the current
                # model as if it were an observation
                #E += rand(MvNormal(Q), ens_sizes[model-1])
                E = ETKF.etkf(E=E, R_inv=P_f_inv, H=H_model, y=mean(E_model, dims=2)[:, 1])
                #for i=1:m
                #    E = ETKF.etkf(E=E, R_inv=P_f_inv/m, H=H_model, y=E_model[:, i])
                #end

                ensembles[model] = E
            end
        end

        if mmm
            E_a = ETKF.etkf(E=hcat(ensembles...), R_inv=R_inv, H=H, y=y)
        else
            if n_models > 1
                E_a = ETKF.etkf(E=ensembles[n_models], R_inv=R_inv, H=H, y=y)
            else
                E_a = ETKF.etkf(E=ensembles[n_models], R_inv=R_inv, H=H, y=y,
                                inflation=inflations[1])
            end
        end
        #E_a = ensembles[n_models]

        P_past = cov(E_a')

        for model=1:n_models
            # Map from reference model space to the current model space
            if ~mmm
                E = obs_ops[model]*E_a
            else
                E = obs_ops[model]*E_a[:, ens_sizes[model]*(model-1)+1:ens_sizes[model]*(model)]
            end
            x_m = mean(E, dims=2)

            errs[model, cycle] = sqrt(mean((x_m .- x_true).^2))
            E_corr_array = xarray.DataArray(data=E, dims=["dim", "member"])
            crps[model, cycle] = xskillscore.crps_ensemble(x_true, E_corr_array).values[1]
            spread[model, cycle] = mean(std(E, dims=2))

            propagators[model] = integrator_prop(model_jacs[model], x_m[:], t,
                                                 t + window*outfreq*Δt, Δt)
            #E = ensembles_a[model]
            for i=1:ens_sizes[model]
                integration = integrator(models[model], E[:, i], t,
                                         t + window*outfreq*Δt, Δt, inplace=false)
                E[:, i] = integration[end, :]
            end

            #if model_errs[model] !== nothing
            #    Q_sample = rand(MvNormal(model_errs[model]), ens_sizes[model])
            #    Q_sample .-= mean(Q_sample, dims=2)
            #    Q_sample *= sqrt(ens_sizes[model]/(ens_sizes[model] - 1))
            #    E += Q_sample
            #end

            #E = x_m .+ sqrt(inflations[model])*(E .- x_m)

            ensembles[model] = E
        end

        x_true = integrator(model_true, x_true, t, t + window*outfreq*Δt, Δt)

        t += window*outfreq*Δt
    end

    return Forecast_Info(errs, errs_uncorr, crps, crps_uncorr, spread,
                         increments, errs_mmm, Q_hist), ensembles, x_true
end

end
