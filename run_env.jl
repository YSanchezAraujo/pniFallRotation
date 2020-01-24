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
crpAlpha = 0.1;

# seed it for repro
Random.seed!(32343)
# trial 71 is when the test phase begins
rs = particle_filterRS(X, n_particles, crpAlpha; max_cause=20);
iw = particle_filterIW(X, n_particles, crpAlpha; max_cause=20);

# plotting the posterior of the causes
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(15, 8));
ax[1].plot(rs.post)
ax[1].legend([string("cause ", x) for x in 1:size(rs.post, 2)])
ax[1].set_title("resampling on all trials from CRP prior")
ax[1].set_ylabel("cause posterior probability")
for axv in [20, 70]
	ax[1].axvline(axv, linestyle="--", color="black")
end
ax[2].plot(iw.post)
ax[2].set_title("resampling only when neff < n_particles/2 from the optimal proposal dist")
ax[2].legend([string("cause ", x) for x in 1:size(iw.post, 2)])
ax[2].set_xlabel("trials")
ax[2].set_ylabel("cause posterior probability")
for axv in [20, 70]
	ax[2].axvline(axv, linestyle="--", color="black")
end
plt.savefig("posteiorsCause.png", dpi=300, bbox_inches="tight")
plt.close()

# same thing as above but only on the test trials, using bar plots
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(13, 8));
ax[1,1].bar(
    height=rs.post[21, :],
    x=[string("cause ", i) for i in 1:size(rs.post, 2)]
)
ax[1, 1].set_ylabel("posteior resampling model")
ax[1, 2].bar(
    height=rs.post[71, :],
    x=[string("cause ", i) for i in 1:size(rs.post, 2)]
)
ax[1, 2].set_title("posterior probability after context switch -- test phase")
ax[1, 1].set_title("posterior probability after context switch -- extinction")
ax[2,1].bar(
    height=iw.post[21, :],
    x=[string("cause ", i) for i in 1:size(iw.post, 2)]
)
ax[2, 1].set_ylabel("posetior importance weighted model")
ax[2, 2].bar(
    height=iw.post[71, :],
    x=[string("cause ", i) for i in 1:size(iw.post, 2)]
)
plt.savefig("barplotsCausePost.png", dpi=300, bbox_inches="tight")
plt.close()

# plotting probability of US
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 5));
ax[1].plot(reshape(mean(rs.pus,dims=3),(90, size(rs.post, 2))))
ax[2].plot(reshape(mean(iw.pus,dims=3),(90, size(iw.post, 2))))
fig.text(0.5, 0.04, "trials", ha="center")
fig.text(0.04, 0.5, "probability of US", va="center", rotation="vertical")
ax[1].set_title("resampling model")
ax[2].set_title("importance weighted model")
plt.savefig("probabilityUS.png", bbox_inches="tight", dpi=300)
plt.close()

# plotting value
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(12, 6));
ax[1].plot(rs.v)
ax[1].set_title("resampling model")
ax[2].plot(iw.v)
ax[2].set_title("importance weighted model")
fig.text(0.5, 0.04, "trials", ha="center")
fig.text(0.04, 0.5, "value", va="center", rotation="vertical")
plt.savefig("value.png", bbox_inches="tight", dpi=300)


# posterior of CS
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(15, 8));
ax[1].plot(reshape(mean(rs.poscs, dims=3), (90, size(rs.post, 2))))
ax[1].legend([string("cause ", x) for x in 1:size(rs.post, 2)])
ax[1].set_title("resampling on all trials from CRP prior")
ax[1].set_ylabel("CS posterior probability")
for axv in [20, 70]
	ax[1].axvline(axv, linestyle="--", color="black")
end
ax[2].plot(reshape(mean(iw.poscs, dims=3), (90, size(iw.post, 2))))
ax[2].set_title("resampling only when neff < n_particles/2 from the optimal proposal dist")
ax[2].legend([string("cause ", x) for x in 1:size(iw.post, 2)])
ax[2].set_xlabel("trials")
ax[2].set_ylabel("CS posterior probability")
for axv in [20, 70]
	ax[2].axvline(axv, linestyle="--", color="black")
end
plt.savefig("posteiorCS.png", dpi=300, bbox_inches="tight")
plt.close()



