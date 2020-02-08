using Distributions, LinearAlgebra, DataFrames;
using PyPlot, Random, CSV;

wpath = "/Users/yoelsanchezaraujo/Desktop/pniFallRotation/";
#wpath = "/home/yoel/Desktop/pniFallRotation/julia/";
include(joinpath(wpath, "julia/aux/env.jl"));
include(joinpath(wpath, "julia/models/particle_filter_iw.jl"));
include(joinpath(wpath, "julia/models/particle_filter_rs.jl"));
include(joinpath(wpath, "julia/models/particle_filter_iwa.jl"));
cd(wpath);
# load data
df = CSV.read(joinpath(wpath, "data/condinhib.csv"));
X = Array(df);
n_particles = 3000;
# seed it for repro
Random.seed!(32343);
# models
iw = particle_filterIW(X, n_particles, 2.0; max_cause=50);
alphIWA = 0.1;
iwa = particle_filterIWA(X, n_particles, alphIWA; max_cause=50);
