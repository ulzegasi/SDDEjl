"""
SABC inference with solar dynamo model
"""

##### Cluster or Local? #####
run_on_cluster = 1  # 0 is local, 1 is cluster
if run_on_cluster ∉ [0, 1]
	error("Set 'run_on_cluster' to 0 or 1")
end

if run_on_cluster == 0
	import Pkg
	Pkg.activate(@__DIR__)   # run scripts in ~/SABC
	Pkg.instantiate()
end

#= Current environment is visible in Pkg REPL (type']' to activate Pkg REPL)
In Pkg REPL (activated with ']') do:
'add /Users/ulzg/SABC/SimulatedAnnealingABC.jl' 
to have the local package installed
OR (after loading Revise)
'dev /Users/ulzg/SABC/SimulatedAnnealingABC.jl' to develop package =#

#= To add dependencies:
pkg> activate /Users/ulzg/SABC/SimulatedAnnealingABC.jl
pkg> add PkgName =#
                   
using Random
using Distributions
using Statistics
using SimulatedAnnealingABC
using Plots
using BenchmarkTools
using CSV 
using DataFrames
using FFTW
using Distances
using DifferentialEquations
using StochasticDelayDiffEq
using SpecialFunctions
using DelimitedFiles
using Serialization

include("./SDDESolarDynamo.jl")
using .SDDESolarDynamo

if run_on_cluster == 0
	datadir = "/Users/ulzg/SABC/"
	outdir = "/Users/ulzg/SABC/output/"
elseif run_on_cluster == 1
	datadir = "/cfs/earth/scratch/ulzg/julia/SABC/"
	outdir = "/cfs/earth/scratch/ulzg/julia/SABC/output/"
end

println("---------------------------------------------------")
println("Running with $(Threads.nthreads()) Julia threads.")
println("RUN_ID = ", get(ENV, "RUN_ID", "not set"))
println("---------------------------------------------------")

#########################################################################
#### Load C14-reconstructed SN data (Usoskin et al., A&A, 2021) #####
#########################################################################

SNdata_temp = parse.(Float64, Matrix(CSV.read(datadir * "SN_Usoskin.csv", DataFrame, header=false))[3:end,2])
SNyrs_temp = parse.(Float64, Matrix(CSV.read(datadir * "SN_Usoskin.csv", DataFrame, header=false))[3:end,1])
Tobs_temp = size(SNdata_temp, 1)
SNdata = reshape(SNdata_temp, Tobs_temp)
SNyrs = reshape(SNyrs_temp, Tobs_temp)

# IMPORTANT: Tobs/nj = period of the nj-th component of the Fourier spectrum
# Jupiter period, Tj = 11.86
# -> Tobs/Tj = nj
# We want nj as close as possible to an integer. Therefore, we remove the first 4 data points, so that:
# (929-4)/11.86 = 77.99
# THE 78TH COMPONENT OF THE FOURIER SPECTRUM IS JUPITER 

deleteat!(SNdata,1:4)
deleteat!(SNyrs,1:4)
# nj = 78
# N.B.: there is in principle no need to reduce the dataset 
# when Jupiter is not in the model, but
# since we delete only 4 points, we do it in any case

Tobs_without_warmup = size(SNdata, 1)
#########################################################################
#########################################################################

ss_obs = summary_statistics(SNdata)
num_sumstats = size(ss_obs,1)

##### Distance #####
function f_dist(θ)
	data = sn(θ; Twarmup = 200, Tobs = Tobs_without_warmup)
	ss = summary_statistics(data)
	rho = [abs(ss[ix] - ss_obs[ix]) for ix in 1:num_sumstats]
	return rho
end

##### Prior ranges #####
τ_min  = 0.1;   τ_max  = 10.0
T_min  = 0.1;   T_max  = 10.0
Nd_min = 1.0;   Nd_max = 15.0
σ_min  = 0.01;  σ_max  = 0.3
Bmax_min = 1.0; Bmax_max = 15.0

prior = product_distribution(Uniform(τ_min, τ_max), Uniform(T_min, T_max), Uniform(Nd_min, Nd_max), 
			Uniform(σ_min, σ_max), Uniform(Bmax_min, Bmax_max))

# --- Run ---
exet = @elapsed out = sabc(f_dist, prior; n_particles = 1000, n_simulation = 12_000_000, 
			show_checkpoint = 500, v = 1.0, 
			algorithm = :single_eps, proposal = DifferentialEvolution(n_para=length(prior)))

println("---------------------------------------------------")
println("Benchmark results for $(Threads.nthreads()) threads:")
println("Elapsed time = ", exet, " seconds")
println("---------------------------------------------------")

outfile = outdir * "benchmark_earth5_C14sing_$(Threads.nthreads())_$(ENV["RUN_ID"]).csv"

open(outfile, "w") do io
    println(io, "run_id,n_threads,elapsed_time")
    println(io, "$(ENV["RUN_ID"]),$(Threads.nthreads()),$exet")
end