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
dfpartial = CSV.read(joinpath(wpath, "data/partialreinf.csv"));
Xpartial = Array(dfpartial);
n_particles = 3000;
# seed it for repro
Random.seed!(32343);
# models
iwpartial = particle_filterIW(Xpartial, n_particles, 2.0; max_cause=50);
alphIWA = 0.5;
iwapartial = particle_filterIWA(Xpartial, n_particles, alphIWA; max_cause=50);

dffull = CSV.read(joinpath(wpath, "data/fullreinf.csv"));
Xfull = Array(dffull);
iwfull = particle_filterIW(Xfull, n_particles, 2.0; max_cause=50);
alphIWA = 0.5;
iwafull = particle_filterIWA(Xfull, n_particles, alphIWA; max_cause=50);

