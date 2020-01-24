include("../aux/utils.jl");
import Turing.Inference.resample_systematic;

function particle_filterIW(X::Array, n_particles::Int64, alpha::Float64; max_cause::Int64=10)::NamedTuple
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
    impW = ones(T, n_particles)
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
    for t in 2:T
        x0_bit, x1_bit = X[t, :] .== 0, X[t, :] .== 1
        priorProbs = update_cause_probs(cause_count, t, alpha) # (max_cause, n_particles)
        priorProbs = priorProbs ./ sum(priorProbs, dims=1)
        cause_prior[t, :, :] = priorProbs
        comps = get_comps(x0_bit, fcountsA, fcountsB, priorProbs, cause_count)
        liks[t, :, :, :] = comps.lik
        cause_post[t, :, :] = comps.post
        value[t] = comps.v
        pUS[t, :, :] = comps.pUS
        posCS[t, :, :] = comps.postCS       
        # forward sample causes for each particle
        z[t, :] = rand(Categorical(vcat(mean(comps.post, dims=2)...)), n_particles)
        # get feature indices to update counts
        x0, x1 = findall(x0_bit), findall(x1_bit)
        # resample the particles according to the importance weights
        impw = zeros(n_particles)
        for pidx in 1:n_particles
            z_particle = z[t, pidx]
            impw[pidx] = comps.post[z_particle, pidx] * impW[t-1, pidx]
        end
        impw = impw ./ sum(impw)
        if 1 ./ sum(impw.^2) < n_particles /2
            println("resampling:    ", t)
            rspidx = resample_systematic(impw, n_particles)
            impw = [1/n_particles for p in 1:n_particles]
        else
            rspidx = collect(1:n_particles)
        end
        impW[t, :] = impw
        # update counts and reassign particles(if needed)
        zIndex = z[t, rspidx]
        z[t, :] = zIndex
        propDist[t, :] = comps.post[:, rspidx] * impw
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
        impw = impW[:, 1:ncu],
        v = value,
        pus = pUS[:, 1:ncu, :],
        poscs = posCS[:, 1:ncu, :],
        z = z,
        post = propDist[:, 1:ncu]
    )    
end
