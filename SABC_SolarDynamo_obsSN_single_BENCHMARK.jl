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
#---- Load observed SN dataset (yearly resolution) ----#
#-------- https://www.sidc.be/SILSO/datafiles --------#
#########################################################################

SNdata_temp = Matrix(CSV.read(datadir * "SN_y_tot_V2.0.csv", DataFrame, header=false))[:,2]
SNyrs_temp = Matrix(CSV.read(datadir * "SN_y_tot_V2.0.csv", DataFrame, header=false))[:,1]
# ------------------------------------------------------
# IMPORTANT 1: 
# a) We don't need years before 1749 (unreliable)
#    Cycle 1 begins 1755/02 (Hathaway, Living Reviews in Solar Physics,12, 2015)
# b) We don't need the beginning of cycle 25. We delete at the end 2020-2024 (5 points)

SNyrs = SNyrs_temp[50:end-5] # 1749-2019
SNdata = SNdata_temp[50:end-5]

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
exet = @elapsed out = sabc(f_dist, prior; n_particles = 1000, n_simulation = 25_000_000, 
			show_checkpoint = 500, v = 1.0, 
			algorithm = :single_eps, proposal = DifferentialEvolution(n_para=length(prior)))

println("---------------------------------------------------")
println("Benchmark results for $(Threads.nthreads()) threads:")
println("Elapsed time = ", exet, " seconds")
println("---------------------------------------------------")

outfile = outdir * "benchmark_earth3_SNsing_$(Threads.nthreads())_$(ENV["RUN_ID"]).csv"

open(outfile, "w") do io
    println(io, "run_id,n_threads,elapsed_time")
    println(io, "$(ENV["RUN_ID"]),$(Threads.nthreads()),$exet")
end

