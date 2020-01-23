using Distributions, LinearAlgebra, DataFrames;
using PyPlot, Random, CSV;

wpath = "/Users/yoelsanchezaraujo/Desktop/pniFallRotation";
include(joinpath(wpath, "aux/env.jl"));
include(joinpath(wpath, "models/particle_filter_iw.jl"));
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
n_particles = 300;
crpAlpha = 0.1;

# trial 71 is when the test phase begins
res = particle_filter(X, n_particles, crpAlpha; max_cause=100);
used_cause_count = size(res.ccount, 1)
postdf = DataFrame(res.post);
likdf = DataFrame(reshape(mean(prod(res.lik, dims=3), dims=4), (90, used_cause_count)));
propCause = DataFrame(prop_value(res.z));


