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
    N = length(cause_vec)
    probs = zeros(N)
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

