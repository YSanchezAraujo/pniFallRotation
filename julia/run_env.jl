using Distributions, LinearAlgebra, DataFrames;
using PyPlot, Random, CSV;

#wpath = "/Users/yoelsanchezaraujo/Desktop/pniFallRotation";
wpath = "/home/yoel/Desktop/pniFallRotation/julia/";
include(joinpath(wpath, "aux/env.jl"));
include(joinpath(wpath, "models/particle_filter_iw.jl"));
include(joinpath(wpath, "models/particle_filter_rs.jl"));
cd(wpath);

# using gershman data
df = CSV.read("/home/yoel/Desktop/pniFallRotation/data/example_datABA.csv");
dfx = df[!, [7, 3, 4, 5, 6]];
X = Array(dfx);
# particle filter parameters
n_particles = 30;
# seed it for repro
Random.seed!(32343)
# now trying with a larger concentration parameter

multi_plot = false

if !multi_plot
    rs = particle_filterRS(X, n_particles, 2.0; max_cause=50);
    iw = particle_filterIW(X, n_particles, 2.0; max_cause=50);
else
	for alpha in [0.1, 0.5, 1.0, 3.0, 6.0]
		plot_results(
	        [particle_filterRS(X, n_particles, alpha; max_cause=50),
	         particle_filterIW(X, n_particles, alpha; max_cause=50)],
	         string("alpha", alpha)
		)
	end
end
