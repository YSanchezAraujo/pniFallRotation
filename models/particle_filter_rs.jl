include("../aux/utils.jl");
import Turing.Inference.resample_systematic;

function particle_filter(X::Array, n_particles::Int64, alpha::Float64; max_cause::Int64=10)::NamedTuple
    T, F = size(X)
    fcountsA = ones(max_cause, F, n_particles)
    fcountsB = ones(max_cause, F, n_particles)
    liks = zeros(T, max_cause, F, n_particles)
    cause_post = zeros(T, max_cause, n_particles)
    cause_prior = zeros(T, max_cause, n_particles)
    cause_count = zeros(Int64, max_cause, n_particles)
    pUS = zeros(T, max_cause, n_particles)
    pUS[1, :, :] .= rand()
    posCS = zeros(T, max_cause, n_particles)
    posCS[1, :, :] .= rand()
    value = zeros(T)
    value[1] = rand()
    propDist = zeros(T, max_cause)
    z = zeros(Int64, T, n_particles)
    z[1, :] .= 1
    cause_count[1, :] .+= 1
    cause_prior[1, 1, :] .= 1.0
    liks[1, 1, :, :] = fcountsA[1, :, :] ./ (fcountsA[1, :, :] .+ fcountsB[1, :, :])
    likTime = hcat([col_prod(liks[1, :, :, p]) for p in 1:n_particles]...)
    impdistNumer = likTime .* cause_prior[1, :, :]
    cause_post[1, :, :] = impdistNumer ./ (sum(impdistNumer, dims=1) .+ eps())
    propDist[1, :] = vcat(mean(cause_post[1, :, :], dims=2)...)
    fcountsA[1, :, :] .+= X[1, :]
    fcountsB[1, :, :] .+= (1 .- X[1, :])
    # all is fine up until here
    K = 1
    for t in 2:T
        # update CRP prior
        priorProbs = update_cause_probs(cause_count, t, alpha) # (max_cause, n_particles)
        priorProbs = priorProbs ./ sum(priorProbs, dims=1)
        cause_prior[t, :, :] = priorProbs
        # forward sample causes for each particle
        for pidx in 1:n_particles
            z[t, pidx] = rand(Categorical(priorProbs[:, pidx]))
        end
        maxZ = maximum(z[t, :])
        maxZ == K ? K = K + 1 : (maxZ > K ? K = maxZ : nothing)
        x0_bit, x1_bit = X[t, :] .== 0, X[t, :] .== 1
        # get feature indices to update counts
        x0, x1 = findall(x0_bit), findall(x1_bit)
        # compute likelihood for these samples
        priorProbs = priorProbs[1:K, :] # (K, n_particles)
        fca, fcb = copy(fcountsA[1:K, :, :]), copy(fcountsB[1:K, :, :])
        fca[:, x0_bit, :] .= fcb[:, x0_bit, :]
        lik = fca ./ (fcountsA[1:K, :, :] .+ fcb) # (K, F, n_particles)
        liks[t, 1:K, :, :] = lik
        likprod = hcat([col_prod(lik[:, :, p]) for p in 1:n_particles]...) # (K, n_particles)
        postNumer = likprod .* priorProbs # (K, n_particles)
        post = postNumer ./ sum(postNumer, dims=1) # (K, n_particles)
        cause_post[t, 1:K, :] = post
        # resample the particles according to the importance weights
        impw = zeros(n_particles)
        for pidx in 1:n_particles
            z_particle = z[t, pidx]
            impw[pidx] = likprod[z_particle, pidx]
        end
        rspidx = resample_systematic(impw ./ sum(impw), n_particles)
        # update counts and reassign particles(if needed)
        zIndex = z[t, rspidx]
        z[t, :] = zIndex
        propDist[t, 1:K] = mean(post[:, rspidx], dims=2)
        cause_count[:, :] = cause_count[:, rspidx]
        fcountsA[:, :, :] = fcountsA[:, :, rspidx]
        fcountsB[:, :, :] = fcountsB[:, :, rspidx]
        for pidx in 1:n_particles
            cause_count[zIndex[pidx], pidx] = cause_count[zIndex[pidx], pidx] .+ 1
            fcountsA[zIndex[pidx], x1, pidx] = fcountsA[zIndex[pidx], x1, pidx] .+ 1
            fcountsB[zIndex[pidx], x0, pidx] = fcountsB[zIndex[pidx], x0, pidx] .+ 1
        end
    end
    ncu = maximum([sum(cause_count[:, pidx] .!= 0) for pidx in 1:n_particles])
    return (
        fca = fcountsA[1:ncu, :, :],
        fcb = fcountsB[1:ncu, :, :],
        lik = liks[:, 1:ncu, :, :],
        cprior = cause_prior[:, 1:ncu, :],
        ccount = cause_count[1:ncu, :],
        v = value,
        pus = pUS[:, 1:ncu, :],
        poscs = posCS[:, 1:ncu, :],
        z = z,
        post = propDist[:, 1:ncu]
    )    
end