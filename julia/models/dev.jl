"""
p: prior probability for cause
c: cause count
a: feature by cause count (on)
b: feature by cause count (off)
l: cause by feature likelihood
z: array of cause sampled for each particle
po: cause by feature posterior
w: importance weight 
"""
mutable struct Particle
    p::Array{Float64, 1}
    c::Array{Int64, 1}
    a::Array{Float64, 2}
    b::Array{Float64, 2}
    l::Array{Float64, 2}
    z::Array{Float64, 1}
    po::Array{Float64, 2}
    w::Array{Float64, 2}
end

struct ParticleBunch
    p::Array{Particle, 1}
end

function initilize(n::Int64, f::Int64, p::Int64, X::Array{Float64, 1})::Particle
    prior = zeros(Float64, n)
    counts = zeros(Int64, n)
    onCount = ones(Float64, n, f)
    offCount = ones(Float64, n, f)
    lik = onCount ./ (onCount .+ offCount)
    prior[1] = 1.0
    counts[1] = 1
    z = ones(Int64, n)
    w = [1/p for i in 1:p]
    poN = col_prod(lik) .* prior
    po = poN / sum(poN, dims=1)
    return Particle(prior, counts, onCount, offCount, lik, z, po, w)
end

function particleLik(p::Particle, x0::BitArray{1})::Array{Float64, 2}
    onF = copy(p.a)
    onF[:, x0] = p.b[:, x0]
    return onF ./ (p.a .+ p.b)
end

function particlePost(p::Particle)::Array{Float64, 1}
    postNumer = col_prod(p.l) .* p.p
    return postNumer ./ sum(postNumer, dims=1)
end

include("../aux/utils.jl");
import Turing.Inference.resample_systematic;

function particle_filter_DEV(X::Array, 
                             n_particles::Int64, 
                             alpha::Float64; 
                             max_cause::Int64=10)::Array{Float64, 2}
    T, F = size(X)
    value = zeros(Float64, T)
    posterior = zeros(Float64, T, max_cause)
    # need to update this to align with the struct above
    pars = ParticleBunch([initialize(max_cause, F, n_particles, X[1, :])
                          for pnum in 1:n_particles])
    for t in 2:T
        x0_bit, x1_bit = X[t, :] .== 0, X[t, :] .== 1
        for pidx in 1:n_particles
            # update cause probabilities
            pars[pidx].p = update_cause_probs(pars[pidx].c, t, alpha)
            pars[pidx].p = pars[pidx].p / sum(pars[pidx].p, dims=1)
            # compute particle likelihood
            pars[pidx].l = particleLik(pars[pidx], x0_bit)
            # compute particle posterior
            pars[pidx].po = particlePost(pars[pidx])
            # sample cause for each particle based on the posterior
            pars[pidx].z[pidx] = rand(Categorical(vcat(mean(pars[pidx].po, dims=2)...)))
            # compute importance weights, not sure this is correct atm
            pars[pidx].w[t, pidx] = pars[pidx].po[pars[pidx.z[pidx]], pidx] * pars[pidx].w[t-1, pidx]
        end
        # normalize importance weights
        pars[pidx].w[t, :] = pars[pidx].w[t, :] ./ sum(pars[pidx].w[t, :])
        # check if resampling is needed
        if 1 ./ sum(pars[pidx].w[t, :].^2) < n_particles / 2
            println("resampling:    ", t)
            rspidx = resample_systematic(pars[pidx].w[t, :], n_particles)
            pars[pidx].w[t, :] .= 1/n_particles
        else
            rspidx = collect(1:n_particles)
        end
        # compute posterior based on resampling,
        #posterior[t, :] = cat([particles[pidx].po for pidx in 1:n_particles].., dims=3)
        # compute value
        #value[t] = 
        # update sufficient statistics
        pars = update_stats(findall(x0), findall(x1), pars, rspidx, n_particles)
    end
    return posterior
end
