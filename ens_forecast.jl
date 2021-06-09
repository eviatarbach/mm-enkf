module ens_forecast

export model_err, init_ens, mmda

using Statistics
using LinearAlgebra
using Random

using NearestNeighbors
using Distributions
using PyCall

struct Forecast_Info
    errs
    errs_fcst
    crps
    spread
    Q_hist
    Q_true_hist
    bias_hist
    analyses
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
                orders,
                obs_ops::AbstractVector{<:AbstractMatrix}, H::AbstractMatrix,
                model_errs::AbstractVector{<:Union{AbstractMatrix{float_type}, Nothing}},
                model_errs_prescribed,
                biases::AbstractVector{<:Union{AbstractVector{float_type}, Nothing}},
                integrator::Function, da_method::Function, localization,
                ens_sizes::AbstractVector{int_type},
                Δt::float_type, window::int_type, n_cycles::int_type,
                outfreq::int_type, model_sizes::AbstractVector{int_type},
                R::Symmetric{float_type}, ens_err=false, ρ::float_type,
                mmm::Bool=false, fixed=false, fcst=false,
                save_analyses::Bool=false, prev_analyses=nothing) where {float_type<:AbstractFloat, int_type<:Integer}
    n_models = length(models)
    obs_err_dist = MvNormal(R)
    R_inv = inv(R)
    obs_op_primes = [H*inv(H_model) for H_model in obs_ops]
    model_errs_fixed = copy(model_errs)

    x_true = x0

    errs = Array{float_type}(undef, n_cycles, model_sizes[1])
    errs_fcst = Array{float_type}(undef, n_cycles, model_sizes[1])
    crps = Array{float_type}(undef, n_models, n_cycles)
    Q_hist = Array{Matrix{float_type}}(undef, n_models, n_cycles)
    Q_true_hist = Array{Matrix{float_type}}(undef, n_models, n_cycles)
    bias_hist = Array{Vector{float_type}}(undef, n_models, n_cycles)
    spread = Array{float_type}(undef, n_models, n_cycles)
    if save_analyses
        analyses = Array{float_type}(undef, n_cycles, model_sizes[1], sum(ens_sizes))
    else
        analyses = nothing
    end

    t = 0.0

    for cycle=1:n_cycles
        println(cycle)

        y = H*x_true + rand(obs_err_dist)

        for model=1:n_models
            E = ensembles[model]

            H_model_prime = obs_op_primes[model]

            x_m = mean(E, dims=2)
            m = ens_sizes[model]
            P_true = (E .- x_true)*(E .- x_true)'/(m - 1)#(x_true - x_m)*(x_true - x_m)'
            innovation = y - H_model_prime*x_m
            P_e = innovation*innovation'
            b = ρ*innovation[:] + (1 - ρ)*biases[model]
            #E .+= b
            #P_e = (H_model_prime*E .- y)*(H_model_prime*E .- y)'/(m - 1)
            P_f = Symmetric(cov(E'))
            if fixed
                λ = tr(P_e - H_model_prime*P_f*H_model_prime' - R)/tr(H_model_prime*model_errs_fixed[model]*H_model_prime')
                Q_est = λ*model_errs_fixed[model]

                if λ < 0
                    #λ = 1
                    #println(λ)
                end
                #if λ > 0
                #Q = ρ*Q_est + (1 - ρ)*model_errs[model]
                #else
                #    Q = model_errs[model]
                #    print("negative")
                #end

                #b = ρ*innovation[:] + (1 - ρ)*biases[model]
            else
                Q_est = inv(H_model_prime)*(P_e - R - H_model_prime*P_f*H_model_prime')*inv(H_model_prime)'
                Q_true = P_true - P_f
                #Q_est = diagm(0=>diag(Q_est))
            end

            Q = ρ*Q_est + (1 - ρ)*model_errs[model]

            if !isposdef(Q)
                Q = make_psd(Q)
                println("not PSD")
            end

            Q_hist[model, cycle] = Q
            Q_true_hist[model, cycle] = Q_true
            model_errs[model] = Q
            bias_hist[model, cycle] = b
            biases[model] = b

            E += rand(MvNormal(model_errs[model]), ens_sizes[model])

            ensembles[model] = E
        end

        ensembles_new = similar(ensembles)
        # Iterative multi-model data assimilation
        if ~mmm
            for (i, order) in enumerate(orders)
                for model=2:n_models
                    # Posterior ensemble of the previous model is used as the prior
                    # ensemble for the next model
                    if model == 2
                        E = ensembles[order[model-1]]
                    else
                        E = ensembles_new[i]
                    end

                    E_model = ensembles[order[model]]

                    H_model = obs_ops[order[model]]

                    P_f = cov(E_model')
                    P_f_diag = Diagonal(diagm(0=>diag(P_f)))
                    P_f_inv = Symmetric(inv(P_f_diag))

                    # Assimilate the forecast of each ensemble member of the current
                    # model as if it were an observation
                    E = da_method(E=E, R=Symmetric(Matrix(P_f_diag)), R_inv=P_f_inv, H=H_model, y=mean(E_model, dims=2)[:, 1],
                                  ρ=localization)

                    ensembles_new[i] = E
                end
            end
        end

        if (n_models > 1) & (~mmm)
            ensembles = ensembles_new
        end

        errs_fcst[cycle, :] = mean(hcat(ensembles...), dims=2) - x_true
        #if mmm
        E_a = da_method(E=hcat(ensembles...), R=R, R_inv=R_inv, H=H, y=y, ρ=localization)
        #else
        #    E_a = ETKF.etkf(E=ensembles[n_models], R_inv=R_inv, H=H, y=y)
        #end
        if save_analyses
            analyses[cycle, :, :] = E_a
        end
        errs[cycle, :] = mean(E_a, dims=2) - x_true
        for model=1:n_models
            # Map from reference model space to the current model space
            #if ~mmm
            #    E = obs_ops[model]*E_a
            #else
            if fcst
                E = x_true .+ rand(MvNormal(ens_err), ens_sizes[model])
            elseif prev_analyses !== nothing
                E = prev_analyses[cycle, :, [0; cumsum(ens_sizes)][model]+1:[0; cumsum(ens_sizes)][model+1]]
            else
                E = obs_ops[model]*E_a[:, [0; cumsum(ens_sizes)][model]+1:[0; cumsum(ens_sizes)][model+1]]
            end
            #end
            x_m = mean(E, dims=2)

            #errs[model, cycle, :] = x_m .- x_true
            E_corr_array = xarray.DataArray(data=E, dims=["dim", "member"])
            crps[model, cycle] = xskillscore.crps_ensemble(x_true, E_corr_array).values[1]
            spread[model, cycle] = mean(std(E, dims=2))

            if model_errs_prescribed[model] === nothing
                pert = zeros(model_sizes[model])
            else
                pert = rand(MvNormal(model_errs_prescribed[model]))
            end
            Threads.@threads for i=1:ens_sizes[model]
                integration = integrator(models[model], E[:, i], t,
                                         t + window*outfreq*Δt, Δt, inplace=false)
                E[:, i] = integration[end, :] + pert
            end

            ensembles[model] = E
        end

        x_true = integrator(model_true, x_true, t, t + window*outfreq*Δt, Δt)

        t += window*outfreq*Δt
    end

    return Forecast_Info(errs, errs_fcst, crps, spread, Q_hist, Q_true_hist, bias_hist, analyses), ensembles, x_true
end

end
