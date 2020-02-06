using Distributions, LinearAlgebra, DataFrames;
using PyPlot, Random, CSV;

wpath = "/Users/yoelsanchezaraujo/Desktop/pniFallRotation/";
#wpath = "/home/yoel/Desktop/pniFallRotation/julia/";
include(joinpath(wpath, "julia/aux/env.jl"));
include(joinpath(wpath, "julia/models/particle_filter_iw.jl"));
include(joinpath(wpath, "julia/models/particle_filter_rs.jl"));
include(joinpath(wpath, "julia/models/particle_filter_iwa.jl"));
cd(wpath);

# using gershman data
df = CSV.read(joinpath(wpath, "data/renewal.csv"));
dfx = df[!, [7, 3, 4, 5, 6]];
X = Array(dfx);
# particle filter parameters
n_particles = 300;
# seed it for repro
Random.seed!(32343)
# now trying with a larger concentration parameter

multi_plot = false

# need to add in extra flexibility such that modeling post pre-training
# situations at t=0 is possible

iw = particle_filterIW(X, n_particles, 2.0; max_cause=50);
plotcv(iw, "alpha2.0IW");
plotbar(iw, "alpha2.0IW")

alphIWA=0.2
iwa = particle_filterIWA(X, n_particles, alphIWA; max_cause=50);
plotcv(iwa, string("alpha",alphIWA,"IWA"));
plotbar(iwa, string("alpha",alphIWA,"IWA"));

if !multi_plot
    iw = particle_filterIW(X, n_particles, 2.0; max_cause=50);
    iwa = particle_filterIWA(X, n_particles, 0.1; max_cause=50);
else
	for alpha in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
		plot_results(
	        [particle_filterIW(X, n_particles, alpha; max_cause=50),
	         particle_filterIWA(X, n_particles, alpha; max_cause=50)],
	         string("alpha", alpha)
		)
	end
end

