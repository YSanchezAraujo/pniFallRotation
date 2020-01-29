using Distributions;

function cue_reward(n::Int64,
                    reinforce::Bool,
                    prob::Float64,
                    type_res::String)::Array{Int64}

    data = zeros(n, 2)
    if type_res == "one_to_one"
        for row_idx in 1:n
            bern_outcome = rand(Bernoulli(prob))
            # always reward when cue present
            if reinforce
                if bern_outcome
                    data[row_idx, :] = [1, 1]
                end
            else
            # never reward when cue is present
                if bern_outcome
                    data[row_idx, :] = [1, 0]
                end
            end
        end
    elseif type_res == "probabilistic"
        for row_idx in 1:n
            if rand(Bernoulli(prob))
                data[row_idx, :] = [1, 1]
            else
                data[row_idx, :] = [1, 0]
            end
        end
    end
    data
end


function make_data(nfeat::Int64, acq_rho::Array, ext_rho::Array,
                   acq_ntime::Int64, ext_ntime::Int64)::Dict

    ntime = acq_ntime + ext_ntime
    data = zeros(ntime, nfeat+1)
    acq_theta = rand(Dirichlet(acq_rho))
    ext_theta = rand(Dirichlet(ext_rho))

    c = 1
    for row_idx in 1:ntime
        if c <= ext_ntime
            data[row_idx, :] = [1; rand(Multinomial(1, acq_theta))]
        else
            data[row_idx, :] = [0; rand(Multinomial(1, ext_theta))]
        end
        c += 1
    end

    Dict(:data => data,
         :acqtheta => acq_theta,
         :exttheta => ext_theta)
end
