function transform_data_to_key(acq_trial::Array{Int64})::Int64
    parse(Int64, join(string.(acq_trial)))
end

function drop_singleton(arr::Array)::Array
    dropdims(arr, dims = (findall(size(arr) .== 1)...,))
end

function counter(arr::Array{Int64})::Dict
    Dict(val => count(x -> x == val, arr) for val in unique(arr))
end

function proportion(arr::Array{Int64})::Dict
    n = length(arr)
    Dict(val => count(x -> x == val, arr)/n for val in unique(arr))
end

function expand_array(arr::Array; t::Int64=1, dim::Int64=1)::Array
    mnpj= size(arr)
    if dim == 1
        return cat(arr, zeros(t, mnpj[2], mnpj[3]), dims=dim)
    elseif dim == 2
        return cat(arr, zeros(mnpj[1], t, mnpj[3], mnpj[4]), dims=dim)
    elseif dim == 3
        return cat(arr, zeros(mnpj[1], mnpj[2], t), dims=dim)
    end
end

function update_cause_probs(cause_vec::Array, t::Int64, alpha::Float64)::Array
    probs = zeros(length(cause_vec))
    nnonzero = Int64(sum(cause_vec .!= 0))
    for (idx, val) in enumerate(cause_vec)
        if (val == 0 && idx == nnonzero + 1)
            probs[idx] = alpha / (t - 1 + alpha)
        elseif val != 0
            probs[idx] = val / (t - 1 + alpha)
        else
            probs[idx] = 0
        end
    end
    return probs
end

function update_cause_probs(cause_mat::Array{Int64, 2}, t::Int64,
                            alpha::Float64)::Array{Float64, 2}
    N, P = size(cause_mat)
    probs = zeros(N, P)
    nonzeroN = [Int64(sum(cause_mat[:, col_idx])) for col_idx in 1:P]
    for col_idx in 1:P
        probs[:, col_idx] = update_cause_probs(cause_mat[:, col_idx], t, alpha)
    end
    probs
end

function col_mnz(x::Array)::Array
    N, P = size(x)
    nzmean = zeros(N)
    for di in 1:N
        nzmean[di] = mnz(x[di, :])
    end
    return nzmean
end

function row_mnz(x::Array)::Array
    N, P = size(x)
    nzmean = zeros(P)
    for di in 1:P
        nzmean[di] = mnz(x[:, di])
    end
    return nzmean
end

function mnz(x::Array)::Float64
    nzmean = 0
    c = 1
    for val in x
        if val != 0
            nzmean += 1/c * (val - nzmean)
            c += 1
        end
    end
    return nzmean
end

function prop_value(z::Array{Int64, 2})::Array{Float64, 2}
    N, P = size(z)
    result = zeros(Float64, N, length(unique(vcat(z...))))
    for t in 1:N
        propInfo = proportion(z[t, :])
        for k in collect(keys(propInfo))
            result[t, k] = propInfo[k]
        end
    end
    return result
end

function row_prod(x::Array{Float64, 2})::Array{Float64, 1}
    N, P = size(x)
    result = zeros(Float64, P)
    for idx in 1:P
        result[idx] = expsumlog(x[:, idx])
    end
    return result
end

function col_prod(x::Array{Float64, 2})::Array{Float64, 1}
    N, P = size(x)
    result = zeros(Float64, N)
    for idx in 1:N
        result[idx] = expsumlog(x[idx, :])
    end
    return result
end

function expsumlog(x::Array{Float64, 1})::Float64
    sl = 0
    for x_i in x
        sl += log(x_i)
    end
    return exp(sl)
end

function get_comps(x0::BitArray{1}, fa::Array{Float64, 3}, fb::Array{Float64, 3}, 
                   prior::Array{Float64, 2}, ccount::Array{Int64, 2})::NamedTuple
    C, F, P = size(fa)
    fCount = copy(fa)
    fCount[:, x0, :] .= fb[:, x0, :]
    lik = fCount ./ (fa .+ fb)
    likprod = hcat([col_prod(lik[:, :, p]) for p in 1:P]...)
    postCS = hcat([col_prod(lik[:, 2:F, p]) for p in 1:P]...)
    pNumer = likprod .* prior
    post = pNumer ./ sum(pNumer, dims=1)
    probUS = fa[:, 1, :] ./  (ccount .+ 2)
    postCS = postCS ./ sum(postCS, dims=1)
    v = sum(col_prod([vcat(probUS...) vcat(postCS...)]) )/ P
    return (lik=lik, likprod=likprod, post=post, 
            pUS=probUS, postCS=postCS, v=v)
end

function plot_results(r, save_prefix)
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(15, 8));
    ax[1].plot(r[1].post)
    ax[1].legend([string("cause ", x) for x in 1:size(r[1].post, 2)])
    ax[1].set_title("resampling on all trials from CRP prior")
    ax[1].set_ylabel("cause posterior probability")
    for axv in [20, 70]
        ax[1].axvline(axv, linestyle="--", color="black")
    end
    ax[2].plot(r[2].post)
    ax[2].set_title("resampling only when neff < n_particles/2 from the optimal proposal dist")
    ax[2].legend([string("cause ", x) for x in 1:size(r[2].post, 2)])
    ax[2].set_xlabel("trials")
    ax[2].set_ylabel("cause posterior probability")
    for axv in [20, 70]
        ax[2].axvline(axv, linestyle="--", color="black")
    end
    plt.savefig(string(save_prefix, "posteiorsCause.png"), dpi=300, bbox_inches="tight")
    plt.close()
    # same as above but bar plots
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(13, 8));
    ax[1,1].bar(
        height=r[1].post[21, :],
        x=[string("cause ", i) for i in 1:size(r[1].post, 2)]
    )
    ax[1, 1].set_ylabel("posteior resampling model")
    ax[1, 2].bar(
        height=r[1].post[71, :],
        x=[string("cause ", i) for i in 1:size(r[1].post, 2)]
    )
    ax[1, 2].set_title("posterior probability after context switch -- test phase")
    ax[1, 1].set_title("posterior probability after context switch -- extinction")
    ax[2,1].bar(
        height=r[2].post[21, :],
        x=[string("cause ", i) for i in 1:size(r[2].post, 2)]
    )
    ax[2, 1].set_ylabel("posetior importance weighted model")
    ax[2, 2].bar(
        height=r[2].post[71, :],
        x=[string("cause ", i) for i in 1:size(r[2].post, 2)]
    )
    plt.savefig(string(save_prefix, "barplotsCausePost.png"), dpi=300, bbox_inches="tight")
    plt.close()
    # plotting probability of US
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 5));
    ax[1].plot(reshape(mean(r[1].pus,dims=3),(90, size(r[1].post, 2))))
    ax[2].plot(reshape(mean(r[2].pus,dims=3),(90, size(r[2].post, 2))))
    fig.text(0.5, 0.04, "trials", ha="center")
    fig.text(0.04, 0.5, "probability of US", va="center", rotation="vertical")
    ax[1].set_title("resampling model")
    ax[2].set_title("importance weighted model")
    plt.savefig(string(save_prefix, "probabilityUS.png"), bbox_inches="tight", dpi=300)
    plt.close()
    # plotting value
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(12, 6));
    ax[1].plot(r[1].v)
    ax[1].set_title("resampling model")
    ax[2].plot(r[2].v)
    ax[2].set_title("importance weighted model")
    fig.text(0.5, 0.04, "trials", ha="center")
    fig.text(0.04, 0.5, "value", va="center", rotation="vertical")
    plt.savefig(string(save_prefix, "value.png"), bbox_inches="tight", dpi=300)
    plt.close()
    # posterior of CS
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(15, 8));
    ax[1].plot(reshape(mean(r[1].poscs, dims=3), (90, size(r[1].post, 2))))
    ax[1].legend([string("cause ", x) for x in 1:size(r[1].post, 2)])
    ax[1].set_title("resampling on all trials from CRP prior")
    ax[1].set_ylabel("CS posterior probability")
    for axv in [20, 70]
        ax[1].axvline(axv, linestyle="--", color="black")
    end
    ax[2].plot(reshape(mean(r[2].poscs, dims=3), (90, size(r[2].post, 2))))
    ax[2].set_title("resampling only when neff < n_particles/2 from the optimal proposal dist")
    ax[2].legend([string("cause ", x) for x in 1:size(r[2].post, 2)])
    ax[2].set_xlabel("trials")
    ax[2].set_ylabel("CS posterior probability")
    for axv in [20, 70]
        ax[2].axvline(axv, linestyle="--", color="black")
    end
    plt.savefig(string(save_prefix, "posteiorCS.png"), dpi=300, bbox_inches="tight")
    plt.close()

end