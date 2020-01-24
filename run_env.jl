using Distributions, LinearAlgebra, DataFrames;
using PyPlot, Random, CSV;

#wpath = "/Users/yoelsanchezaraujo/Desktop/pniFallRotation";
wpath = "/home/yoel/Desktop/pniFallRotation";
include(joinpath(wpath, "aux/env.jl"));
include(joinpath(wpath, "models/particle_filter_iw.jl"));
include(joinpath(wpath, "models/particle_filter_rs.jl"));
cd(wpath);

# using gershman data
df = CSV.read("data/example_datABA.csv");
dfx = df[!, [7, 3, 4, 5, 6]];
X = Array(dfx);
# creating data
# acq_rho = [20, 3, 2, 4];
# ext_rho = [3, 20, 2, 4];
# nfeat = 4;
# acqn = 50;
# extn = 50;
# data_params = make_data(nfeat, acq_rho, ext_rho, acqn, extn);
# data = data_params[:data];
# particle filter parameters
n_particles = 3000;
# seed it for repro
Random.seed!(32343)
# now trying with a larger concentration parameter

plot_results(
	[particle_filterRS(X, n_particles, 0.1; max_cause=50), 
	 particle_filterIW(X, n_particles, 0.1; max_cause=50)]
    , "alpha0.1"
)

plot_results(
	[particle_filterRS(X, n_particles, 0.5; max_cause=50), 
	 particle_filterIW(X, n_particles, 0.5; max_cause=50)]
    , "alpha0.5"
)

plot_results(
	[particle_filterRS(X, n_particles, 1.0; max_cause=50), 
	 particle_filterIW(X, n_particles, 1.0; max_cause=50)]
    , "alpha1.0"
)

plot_results(
	[particle_filterRS(X, n_particles, 3.0; max_cause=50), 
	 particle_filterIW(X, n_particles, 3.0; max_cause=50)]
    , "alpha3.0"
)


plot_results(
	[particle_filterRS(X, n_particles, 6.0; max_cause=50), 
	 particle_filterIW(X, n_particles, 6.0; max_cause=50)]
    , "alpha6.0"
)