@model infiniteDMM(x, alpha, rho) = begin
    # Hyper-parameters, i.e. concentration parameter and parameters of H.
    M, N = size(x)
    # Define random measure, e.g. Dirichlet process.
    rpm = DirichletProcess(alpha)
    # Define the base distribution, i.e. expected value of the Dirichlet process.
    #H = Normal(μ0, σ0)
    H = Dirichlet(rho)
    # Latent assignment.
    z = tzeros(Int64, M)
    # Locations of the infinitely many clusters.
    theta = TArray{Float64}(0, N)
    for i in 1:M
        # Number of clusters.
        K = maximum(z)
        nk = Vector{Int}(map(k -> sum(z .== k), 1:K))
        # Draw the latent assignment.
        z[i] ~ ChineseRestaurantProcess(rpm, nk)
        # Create a new cluster?
        if z[i] > K
            theta = vcat(theta, TArray{Float64}(1, N))
            # Draw location of new cluster.
            theta[z[i], :] ~ H
        end
        # Draw observation.
        x[i, :] ~ Multinomial(1, theta[z[i], :])
    end
end

@model binomProduct(X, alpha, a, b) = begin
    N, P = size(X)
    randomMeasure = DirichletProcess(alpha)
    G0 = Beta(a, b)
    z = tzeros(Int64, N)
    theta = TArray{Float64}(0, P)
    for n in 1:N
        K = maximum(z)
        Nk = Vector{Int}(map(k -> sum(z .== k), 1:K))
        z[n] ~ ChineseRestaurantProcess(randomMeasure, Nk)
        if z[n] > K
            theta = vcat(theta, TArray{Float64}(1, P))
            for f in 1:P
                theta[z[n], f] ~ G0
            end
        end
        for f in 1:P
            X[n, f] ~ Binomial(1, theta[z[n], f])
        end
    end
end