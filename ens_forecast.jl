module ens_forecast

export model_err, init_ens, mmda

using Statistics
using LinearAlgebra
using Random

using Distributions
using PyCall

struct Forecast_Info
    errs
    errs_fcst
    crps
    crps_fcst
    spread
    spread_fcst
    Q_hist
    Q_true_hist
    bias_hist
    analyses
    model_errs
    inflation_hist
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
                obs_ops::AbstractVector{<:AbstractMatrix},
                mappings,
                model_errs::AbstractVector{<:Union{AbstractMatrix{float_type}, Nothing}},
                model_errs_prescribed,
                biases::AbstractVector{<:Union{AbstractVector{float_type}, Nothing}},
                integrator::Function, da_method::Function, localization,
                ens_sizes::AbstractVector{int_type},
                Δt::float_type, window::int_type, n_cycles::int_type,
                outfreq::int_type, model_sizes::AbstractVector{int_type},
                R::Symmetric{float_type}, ens_errs=false, ρ::float_type,
                ρ_all=0.01, all_orders::Bool=true,
                combine_forecasts::Bool=true, fcst=false, assimilate_obs=true, save_Q_hist=false,
                save_analyses::Bool=false, prev_analyses=nothing, leads=1,
                ref_model=1) where {float_type<:AbstractFloat, int_type<:Integer}
    n_models = length(models)
    obs_err_dist = MvNormal(R)
    R_inv = inv(R)

    x_true = x0

    errs = Array{float_type}(undef, n_cycles, model_sizes[1])
    errs_fcst = Array{float_type}(undef, n_cycles, model_sizes[1])
    crps = Array{float_type}(undef, n_cycles)
    crps_fcst = Array{float_type}(undef, n_cycles)
    Q_hist = Array{float_type}(undef, n_models, n_cycles)
    Q_true_hist = Array{float_type}(undef, n_models, n_cycles)
    if save_Q_hist
        Q_hist = Array{Matrix{float_type}}(undef, n_models, n_cycles)
        Q_true_hist = Array{Matrix{float_type}}(undef, n_models, n_cycles)
    end
    bias_hist = Array{Vector{float_type}}(undef, n_models, n_cycles)
    spread = Array{float_type}(undef, n_cycles)
    spread_fcst = Array{float_type}(undef, n_cycles)
    inflation_hist = Array{float_type}(undef, n_cycles)

    inflation_all = ones(leads)

    if save_analyses
        analyses = Array{float_type}(undef, n_cycles, model_sizes[1], sum(ens_sizes))
    else
        analyses = nothing
    end

    model_errs_leads = Array{AbstractMatrix{float_type}}(undef, n_models, leads)

    for model=1:n_models
        for lead=1:leads
            model_errs_leads[model, lead] = model_errs[model]*lead^2
        end
    end

    if all_orders
        orders = []
        for i=1:n_models
            order = Array(1:n_models)
            replace!(order, 1=>i, i=>1)
            append!(orders, [order])
        end
    else
        order = Array(1:n_models)
        replace!(order, 1=>ref_model, ref_model=>1)
        orders = [order]
    end

    t = 0.0

    for cycle=1:n_cycles
        println(cycle)

        y = obs_ops[ref_model]*x_true + rand(obs_err_dist)

        lead = mod(cycle, leads)

        for model=1:n_models
            E = ensembles[model]

            H_model_prime = obs_ops[model]

            x_m = mean(E, dims=2)
            m = ens_sizes[model]
            #P_true = (E .- x_true)*(E .- x_true)'/(m - 1)#(x_true - x_m)*(x_true - x_m)'
            innovation = y - H_model_prime*x_m
            P_e = innovation*innovation'
            #b = ρ*innovation[:] + (1 - ρ)*biases[model]
            #E .+= b
            #P_e = (H_model_prime*E .- y)*(H_model_prime*E .- y)'/(m - 1)
            P_f = Symmetric(cov(E'))

            Q_est = pinv(H_model_prime)*(P_e - R - H_model_prime*P_f*H_model_prime')*pinv(H_model_prime)'
            #Q_true = P_true - P_f
            #Q_est = diagm(0=>diag(Q_est))

            Q = ρ*Q_est + (1 - ρ)*model_errs_leads[model, lead + 1]

            if !isposdef(Q)
                Q = make_psd(Q)
                println("not PSD")
            end

            if save_Q_hist
                Q_hist[model, cycle] = Q
                #Q_true_hist[model, cycle] = Q_true
            else
                Q_hist[model, cycle] = tr(Q)
                #Q_true_hist[model, cycle] = tr(Q_true)
            end
            model_errs_leads[model, lead + 1] = Q
            #bias_hist[model, cycle] = b
            #biases[model] = b

            E += rand(MvNormal(model_errs_leads[model, lead + 1]), ens_sizes[model])

            ensembles[model] = E
        end

        ensembles_new = similar(ensembles)
        # Iterative multi-model data assimilation
        if combine_forecasts
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

                    H_model = mappings[order[1], order[model]]

                    P_f = cov(E_model')
                    # P_f_diag = Tridiagonal(diagm(0=>diag(P_f)))
	                P_f_inv = Symmetric(inv((mappings[ref_model, order[model]]*localization*mappings[ref_model, order[model]]').*P_f))

                    # Assimilate the forecast of each ensemble member of the current
                    # model as if it were an observation
                    E = da_method(E=E, R=Symmetric(Matrix(P_f_inv)), R_inv=P_f_inv, H=H_model, y=mean(E_model, dims=2)[:, 1],
                                  ρ=localization)

                    ensembles_new[i] = E
                end
            end
        end

        if (n_models > 1) & (combine_forecasts)
            if all_orders
                ensembles = ensembles_new
            else
                ensembles = [ensembles_new[ref_model]]
            end
        end

        if all_orders & (~all(model_sizes[1] .== model_sizes))
            error("Not implemented")
        end

        if (~all_orders) & (~all(ens_sizes[1] .== ens_sizes))
            error("Ensemble sizes must be the same")
        end

        if all_orders
            E_all = hcat([mappings[model, ref_model]*ensembles[model] for model=1:n_models]...)
        else
            E_all = ensembles
        end

	    if (combine_forecasts & (n_models > 1))
            H = obs_ops[ref_model]

            x_m = mean(E_all, dims=2)
            innovation = y - H*x_m

            P_e = innovation*innovation'
            P_f = Symmetric(cov(E_all'))
            λ = tr(P_e - R)/tr(H*P_f*H')

            inflation_all[lead + 1] = ρ_all*λ + (1 - ρ_all)*inflation_all[lead + 1]
	        inflation_hist[cycle] = inflation_all[lead + 1]

            E_all = x_m .+ sqrt(inflation_all[lead + 1])*(E_all .- x_m)
        end

        errs_fcst[cycle, :] = mean(E_all, dims=2) - x_true

	    E_corr_fcst_array = xarray.DataArray(data=E_all, dims=["dim", "member"])
        crps_fcst[cycle] = xskillscore.crps_ensemble(x_true, E_corr_fcst_array).values[1]
        spread_fcst[cycle] = mean(std(E_all, dims=2))

        if assimilate_obs
            E_a = da_method(E=E_all, R=R, R_inv=R_inv, H=obs_ops[ref_model],
                            y=y, ρ=localization)

            E_corr_array = xarray.DataArray(data=E_a, dims=["dim", "member"])
            crps[cycle] = xskillscore.crps_ensemble(x_true, E_corr_array).values[1]

            spread[cycle] = mean(std(E_a, dims=2))

            if save_analyses
                analyses[cycle, :, :] = E_a
            end
            errs[cycle, :] = mean(E_a, dims=2) - x_true
        else
            E_a = E_all
        end

        for model=1:n_models
            if fcst & (mod(cycle, leads) == 0)
                E = x_true .+ rand(MvNormal(ens_errs[model]), ens_sizes[model])
            elseif prev_analyses !== nothing
                E = prev_analyses[cycle, :, [0; cumsum(ens_sizes)][model]+1:[0; cumsum(ens_sizes)][model+1]]
            else
                if all_orders
                    E = mappings[ref_model, model]*E_a[:, [0; cumsum(ens_sizes)][model]+1:[0; cumsum(ens_sizes)][model+1]]
                else
                    E = mappings[ref_model, model]*E_a
                end
            end

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

    return Forecast_Info(errs, errs_fcst, crps, crps_fcst, spread, spread_fcst,
                         Q_hist, Q_true_hist, bias_hist, analyses,
                         model_errs_leads, inflation_hist), ensembles, x_true
end

end
