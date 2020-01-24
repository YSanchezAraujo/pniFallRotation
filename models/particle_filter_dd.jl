include("../aux/utils.jl");
import Turing.Inference.resample_systematic;

function particle_filterDD(X::Array, n_particles::Int64, alpha::Float64; max_cause::Int64=10)::NamedTuple
    T, F = size(X)
    # TODO: everything -- function for inference using distance dependent CRP
end