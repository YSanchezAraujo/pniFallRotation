using Distributions, LinearAlgebra, DataFrames;
using PyPlot, Random, CSV;

wpath = "/Users/yoelsanchezaraujo/Desktop/nlr";
include(joinpath(wpath, "env.jl"));
include(joinpath(wpath, "particle_filter_iw.jl"));
cd(wpath);

# using gershman data
df = CSV.read("example_datABA.csv");
X = df[!, [7, 3, 4, 5, 6]];
# creating data
# acq_rho = [20, 3, 2, 4];
# ext_rho = [3, 20, 2, 4];
# nfeat = 4;
# acqn = 50;
# extn = 50;
# data_params = make_data(nfeat, acq_rho, ext_rho, acqn, extn);
# data = data_params[:data];
# particle filter parameters
n_particles = 40;
crpAlpha = 0.1;

# trial 71 is when the test phase begins
res = particle_filter(Array(X), n_particles, crpAlpha; max_cause=100);

used_cause_count = size(res.ccount, 1)
postdf = DataFrame(res.ap);
likdf = DataFrame(reshape(mean(prod(res.lik, dims=3), dims=4), (90, used_cause_count)));
propCause = DataFrame(prop_value(res.z));

clist = ["red", "blue", "black", "green", "orange", "purple"]
set_default_plot_size(20cm, 18cm)
vstack(
    plot(propCause,
        [layer(y=string("x", i), Geom.point, 
            Geom.vline, xintercept=[21, 71, 90], 
            Theme(default_color=color(clist[i])))
        for i in 1:size(propCause, 2)]...,
        Guide.xlabel("trial number"),
        Guide.ylabel("proportion of cause sampled"),
    ),
    plot(x=resRs.z[21, :], Geom.histogram, Theme(bar_spacing=1mm)),
    plot(x=resRs.z[71, :], Geom.histogram, Theme(bar_spacing=1mm)),
    plot(x=resRs.z[90, :], Geom.histogram, Theme(bar_spacing=1mm))
)

set_default_plot_size(20cm, 18cm)
vstack(
    plot(x=resRs.z[21, :], Geom.histogram, Theme(bar_spacing=1mm)),
    plot(x=resRs.z[71, :], Geom.histogram, Theme(bar_spacing=1mm)),
    plot(x=resRs.z[90, :], Geom.histogram, Theme(bar_spacing=1mm))
)

plot(propCause,
	[layer(y=string("x", i), Geom.point, Geom.line, 
		for i in 1:size(propCause, 2)]...,
    Guide.xlabel("trial number"),
	Guide.ylabel("proportion of cause sampled"),
	Theme(key_position=:none)
)

plot(likdf,
	[layer(y=string("x", i), Geom.point, Geom.line) for i in 1:size(likdf, 2)]...,
	Guide.XLabel("trial number"),
    Guide.YLabel("likelihood")
)

plot(postdf, 
	[layer(y=string("x", i), Geom.point, Geom.line) for i in 1:size(post, 2)]...,
	Guide.XLabel("trial number"),
	Guide.YLabel("posterior probability")
)


# this plots a matrix heatmap like object
set_default_plot_size(40cm, 40cm)
spy(z)

plot(y=propCause.x1, Geom.point, Geom.line)
plot(x=z[3, :], Geom.histogram, Theme(bar_spacing=1mm))


# this somehow works!
# plt.close()
# gcf()
# i0=reshape(prod(liks, dims=3), (90, used_cause_count, n_particles));
# i1=reshape(mean(cprior .* i0, dims=3), (90, used_cause_count));
# i2 = i1 ./ sum(i1, dims=1);
# plt.plot(i2 ./ sum(i2, dims=2));
# plt.plot(mean(i2, dims=2))
# gcf()
