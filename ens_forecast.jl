module ens_forecast

export da_cycles

using Statistics
using LinearAlgebra
using Random

using Distributions
using ProgressMeter
using PyCall

struct Forecast_Info
    errs
    errs_fcst
    crps
    crps_fcst
    spread
    spread_fcst
    Q_hist
    P_hist
    analyses
    trues
    model_errs
    inflation_hist
end

xskillscore = pyimport("xskillscore")
xarray = pyimport("xarray")

function make_psd(A, tol=1e-6)
    L, Q = eigen(A)
    L[L .< 0] .= tol
    return Symmetric(Q*diagm(0=>L)*inv(Q))
end

function da_cycles(; x0::AbstractVector{float_type},
                     ensembles::AbstractVector{<:AbstractMatrix{float_type}},
                     models::AbstractVector{<:Function}, model_true::Function,
                     obs_ops::AbstractVector{<:AbstractMatrix}, H_true=I,
                     mappings::Union{Matrix{AbstractArray}, Nothing}=nothing,
                     model_errs::AbstractVector{<:Union{AbstractMatrix{float_type}, Nothing}},
                     model_errs_prescribed::AbstractVector{<:Union{AbstractMatrix{float_type}, Nothing}},
                     integrators::AbstractVector{<:Function}, integrator_true::Function,
                     da_method::Function, localization::AbstractMatrix{float_type},
                     ens_sizes::AbstractVector{int_type}, Δt::float_type, window::int_type,
                     n_cycles::int_type, outfreq::int_type,
                     model_sizes::AbstractVector{int_type}, R::Symmetric{float_type},
                     ens_errs::Union{AbstractVector{<:AbstractMatrix{float_type}}, Nothing}=nothing,
                     ρ::float_type,
                     Q_p::Union{AbstractVector{<:AbstractMatrix{float_type}}, Nothing}=nothing,
                     ρ_all::float_type=0.01, all_orders::Bool=true,
                     combine_forecasts::Bool=true, gen_ensembles::Bool=false,
                     assimilate_obs::Bool=true, save_Q_hist::Bool=false,
                     save_P_hist::Bool=false, save_analyses::Bool=false, save_trues::Bool=false,
                     prev_analyses::Union{AbstractArray{float_type}, Nothing}=nothing,
                     leads::int_type=1, ref_model::int_type=1) where {float_type<:AbstractFloat, int_type<:Integer}
    n_models = length(models)
    obs_err_dist = MvNormal(R)
    R_inv = inv(R)

    x_true = x0

    errs = Array{float_type}(undef, n_cycles, model_sizes[ref_model])
    errs_fcst = Array{float_type}(undef, n_cycles, model_sizes[ref_model])
    crps = Array{float_type}(undef, n_cycles)
    crps_fcst = Array{float_type}(undef, n_cycles)
    Q_hist = Array{float_type}(undef, n_models, n_cycles)
    if save_Q_hist
        Q_hist = Array{Matrix{float_type}}(undef, n_models, n_cycles)
    end
    if save_P_hist
        P_hist = Array{Matrix{float_type}}(undef, n_models, n_cycles)
    else
        P_hist = nothing
    end
    spread = Array{float_type}(undef, n_cycles)
    spread_fcst = Array{float_type}(undef, n_cycles)
    inflation_hist = Array{float_type}(undef, n_cycles)

    inflation_all = ones(leads)

    if save_analyses
        analyses = Array{float_type}(undef, n_cycles, model_sizes[ref_model],
                                     all_orders ? sum(ens_sizes) : ens_sizes[ref_model])
    else
        analyses = nothing
    end

    if save_trues
        trues = Array{float_type}(undef, n_cycles, model_sizes[ref_model])
    else
        trues = nothing
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

    if mappings === nothing
        if (~all(model_sizes[1] .== model_sizes))
            error("Must specify mappings")
        else
            mappings = Matrix{AbstractArray}(undef, n_models, n_models)
            for m1=1:n_models
                for m2=1:n_models
                    mappings[m1, m2] = I(model_sizes[1])
                end
            end
        end
    end

    t = 0.0

    @showprogress for cycle=1:n_cycles
        y = H_true*x_true + rand(obs_err_dist)

        lead = mod(cycle, leads)

        for model=1:n_models
            model_size = model_sizes[model]
            E = ensembles[model]

            H_m = obs_ops[model]

            x_m = mean(E, dims=2)
            innovation = y - H_m*x_m
            P_p = Symmetric(cov(E'))

            if save_P_hist
                P_hist[model, cycle] = P_p
            end

            C = innovation*innovation' - R - H_m*P_p*H_m'
            if rank(H_m) >= model_size
                Q_est = pinv(H_m)*C*pinv(H_m)'
            else
                A = Array{float_type}(undef, size(R)[1]^2, length(Q_p))
                for p=1:length(Q_p)
                    A[:, p] = vec(H_m*Q_p[p]*H_m')
                end
                q = A \ vec(C)
                Q_est = sum([q[p]*Q_p[p] for p=1:length(Q_p)])
            end

            Q = Symmetric(ρ*Q_est + (1 - ρ)*model_errs_leads[model, lead + 1])

            if !isposdef(Q)
                Q = make_psd(Q)
            end

            if save_Q_hist
                Q_hist[model, cycle] = Q
            else
                Q_hist[model, cycle] = tr(Q)
            end
            model_errs_leads[model, lead + 1] = Q

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
                    if localization !== nothing
                        P_f = (mappings[ref_model, order[model]]*localization*mappings[ref_model, order[model]]').*P_f
                    else
                        P_f = Diagonal(diagm(diag(P_f)))
                    end
                    P_f_inv = Symmetric(inv(P_f))

                    if localization !== nothing
                        localization_m = mappings[ref_model, order[model-1]]*localization*mappings[ref_model, order[model-1]]'
                    else
                        localization_m = nothing
                    end

                    # Assimilate the forecast of each ensemble member of the current
                    # model as if it were an observation
                    E = da_method(E=E, R=Symmetric(Matrix(P_f)), R_inv=P_f_inv, H=H_model,
                                  y=mean(E_model, dims=2)[:, 1],
                                  localization=localization_m)

                    ensembles_new[i] = E
                end
            end
        end

        if (n_models > 1) & (combine_forecasts)
            ensembles = ensembles_new
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
            E_all = ensembles[1]
        end

        H = obs_ops[ref_model]

        x_m = mean(E_all, dims=2)
        innovation = y - H*x_m

        P_e = innovation*innovation'
        P_f = Symmetric(cov(E_all'))
        λ = tr(P_e - R)/tr(H*P_f*H')
        λ = max(λ, 0)

        inflation_all[lead + 1] = ρ_all*λ + (1 - ρ_all)*inflation_all[lead + 1]
        inflation_hist[cycle] = inflation_all[lead + 1]

        E_all = x_m .+ sqrt(inflation_all[lead + 1])*(E_all .- x_m)

        errs_fcst[cycle, :] = mean(E_all, dims=2) - pinv(obs_ops[ref_model])*H_true*x_true

	    E_corr_fcst_array = xarray.DataArray(data=E_all, dims=["dim", "member"])
        #crps_fcst[cycle] = xskillscore.crps_ensemble(pinv(obs_ops[ref_model])*H_true*x_true, E_corr_fcst_array).values[1]
        spread_fcst[cycle] = mean(std(E_all, dims=2))

        if assimilate_obs & (mod(cycle, leads) == 0)
            E_a = da_method(E=E_all, R=R, R_inv=R_inv, H=obs_ops[ref_model],
                            y=y, localization=localization)

            E_corr_array = xarray.DataArray(data=E_a, dims=["dim", "member"])
            #crps[cycle] = xskillscore.crps_ensemble(pinv(obs_ops[ref_model])*H_true*x_true, E_corr_array).values[1]

            spread[cycle] = mean(std(E_a, dims=2))

            errs[cycle, :] = mean(E_a, dims=2) - pinv(obs_ops[ref_model])*H_true*x_true
        else
            E_a = E_all
        end

        if save_analyses
            analyses[cycle, :, :] = E_a
        end

        if save_trues
            trues[cycle, :] = x_true
        end

        for model=1:n_models
            if gen_ensembles & (mod(cycle, leads) == 0)
                E = mappings[ref_model, model]*pinv(obs_ops[ref_model])*H_true*x_true .+ rand(MvNormal(ens_errs[model]), ens_sizes[model])
            elseif (prev_analyses !== nothing) & (mod(cycle, leads) == 0)
                E = mappings[ref_model, model]*prev_analyses[cycle, :, [0; cumsum(ens_sizes)][model]+1:[0; cumsum(ens_sizes)][model+1]]
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
                integration = integrators[model](models[model], E[:, i], t,
                                                 t + window*outfreq*Δt, Δt, inplace=false)
                E[:, i] = integration[end, :] + pert
            end

            ensembles[model] = E
        end

        x_true = integrator_true(model_true, x_true, t, t + window*outfreq*Δt, Δt)

        t += window*outfreq*Δt
    end

    return Forecast_Info(errs, errs_fcst, crps, crps_fcst, spread, spread_fcst, Q_hist,
                         P_hist, analyses, trues, model_errs_leads, inflation_hist)
end

end
