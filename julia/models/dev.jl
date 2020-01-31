"""
p: prior probability for cause
c: cause count
a: feature by cause count (on)
b: feature by cause count (off)
l: cause by feature likelihood
po: cause by feature posterior
w: importance weight 
"""
mutable struct Particle
	p::Array{Float64, 1}
	c::Array{Int64, 1}
	a::Array{Float64, 2}
	b::Array{Float64, 2}
	l::Array{Float64, 2}
	po::Array{Float64, 2}
	w::Float64
end

struct ParticleBunch
	p::Array{Particle, 1}
end

function initPrior(n::Int64)::Array{Float64, 1}
    prior = zeros(Float64, n)
    prior[1] = 1.0
    return prior
end

function initCCount(n::Int64)::Array{Int64, 1}
    counts = zeros(Int64, n)
    counts[1] = 1
    return counts
end

function initOnCount(n::Int64, f::Int64, X::Array{Float64, 1})::Array{Float64, 2}
    onCount = ones(Float64, n, f)
    onCount[1, :] .+= X
    return onCount
end

function initOffCount(n::Int64, f::Int64, X::Array{Float64, 1})::Array{Float64, 2}
    offCount = ones(Float64, n, f)
    offCount[1, :] .+= (1 .- X)
    return offCount
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
	                         max_cause::Int64=10)::ParticleBunch
    T, F = size(X)
    impW = zeros(Float64, T, n_particles)
    z = zeros(Int64, T, n_particles)
    iOnC = initOnCont(max_cause, F, X[1, :])
    iOffC = initOffCount(max_cause, F, X[1, :])
    likInit = iOnC ./ (iOnC .+ iOffC)
    posterior = zeros(Float64, T, max_cause)

    particles = ParticleBunch([
        Particle(initPrior(max_cause),
                 initCCount(max_cause),
                 iOnC, iOffC, likInit,
                 1/n_particles)
        for pnum in 1:n_particles
    ])

    for t in 2:T
    	for pidx in 1:n_particles
            x0_bit, x1_bit = X[t, :] .== 0, X[t, :] .== 1
            particles[pidx].p = update_cause_probs(particles[pidx].c, t, alpha)
            particles[pidx].p = particles[pidx].p / sum(particles[pidx].p, dims=1)
            particles[pidx].l = particleLik(particles[pidx], x0_bit)
            particles[pidx].po = particlePost(particles[pidx])
    	end
        posterior[t, :] = cat([particles[pidx].po for pidx in 1:n_particles]..., dims=3
    end
    return particles
end
