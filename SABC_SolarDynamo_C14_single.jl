"""
SABC inference with solar dynamo model
"""

##### Cluster or Local? #####
run_on_cluster = 1  # 0 is local, 1 is cluster
if run_on_cluster ∉ [0, 1]
	error("Set 'run_on_cluster' to 0 or 1")
end
##### First or udpate run? #####
from_previous = 1  # set 0 for first run, 1 for updating previous run
if from_previous ∉ [0, 1]
	error("Set 'from_previous' to 0 or 1")
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

# --- Instead of deleting the first 4 data points, delete the first 5 ---
# --- This makes the first data point (year 976) a minimum. Compatible with history values that have derivative dB(t=0) ~ 0 
# deleteat!(SNdata,1:5)
# deleteat!(SNyrs,1:5)

Tobs_without_warmup = size(SNdata, 1)
#########################################################################
#########################################################################

##### Define summary stats for observations #####
#################################################
### FULL SET (1:6:120) OR II-BASED SELECTION ###

# FULL SET is the default:
# summary_statistics_ii(data)

##### DEFINITION OF II-BASED SUMMARY STATS ##### 
# We consider the first 120 Fourier components, Re and Im parts (-> 240 components).
# Indices 0-119 are Re parts, indices 120-239 are the Im parts

# ---- Observed SN (yearly resolution), Tobs = 271:
# fourier_range = [1, 129, 68, 197, 7, 75, 76, 206, 79, 210, 211, 56, 57, 127] .+ 1

# ---- C-14 data (Usoskin et al., A&A, 2021), Tobs = 925:
# fourier_range = [65, 1, 2, 142, 145, 113, 28, 29] .+ 1
#################################################

# ------ --------------------------------------------- ------ #
# ------ If you want to add Jupiter line to the stats: ------ #
# ------ --------------------------------------------- ------ #
# fourier_range = [ix for ix in 1:6:120]
# if !(nj in fourier_range)
# 	append!(fourier_range, nj)
# end 
# sort!(fourier_range)
# ------ --------------------------------------------- ------ #
# ------ --------------------------------------------- ------ #

# --> N.B: IMPORTANT! Use summary_statistics OR summary_statistics_ii <--

ss_obs = summary_statistics(SNdata)

num_sumstats = size(ss_obs,1)

##### Distance #####
function f_dist(θ)
	data = sn(θ; Twarmup = 200, Tobs = Tobs_without_warmup)
	ss = summary_statistics(data)
	# Distance --- alternatives: sqeuclidean, cityblock (cityblock(x, y):=sum(abs(x - y)))
	rho = [abs(ss[ix] - ss_obs[ix]) for ix in 1:num_sumstats]
	return rho
end

##### Prior ranges #####
τ_min  = 0.1;   τ_max  = 10.0
T_min  = 0.1;   T_max  = 10.0
Nd_min = 1.0;   Nd_max = 15.0
σ_min  = 0.01;  σ_max  = 0.3
Bmax_min = 1.0; Bmax_max = 15.0

# prior = product_distribution(Uniform(τ_min, τ_max), Uniform(T_min, T_max), Uniform(Nd_min, Nd_max), 
# 							Uniform(σ_min, σ_max), Uniform(Bmax_min, Bmax_max),  Uniform(ϵ_min, ϵ_max), 
# 							Uniform(ϕ_min, ϕ_max))

prior = product_distribution(Uniform(τ_min, τ_max), Uniform(T_min, T_max), Uniform(Nd_min, Nd_max), 
			Uniform(σ_min, σ_max), Uniform(Bmax_min, Bmax_max))

if from_previous == 0
	# --- Output file name ---
	fname = "C14_single_77_10ka"
	# --- Run ---
	out = sabc(f_dist, prior; n_particles = 10_000, n_simulation = 2_500_000_000, 
				show_checkpoint = 500, v = 1.0,
				algorithm = :single_eps, proposal = DifferentialEvolution(n_para=length(prior)))
	display(out); flush(stdout)
elseif from_previous == 1
	# --- Output file name ---
	fname = "C14_single_77_10kd"
	# --- Continue from... ---
	fname_previous = "C14_single_77_10kc"
	# --- Run ---
	fasa = deserialize(outdir * "SABCresult_" * fname_previous)
	out = update_population!(fasa, f_dist, prior; n_simulation = 2_500_000_000, 
							show_checkpoint = 500, v = 1.0,
							proposal = DifferentialEvolution(n_para=length(prior)))
	display(out); flush(stdout)
end


pop = hcat(out.population...)
eps = hcat(out.state.ϵ_history...)
rhos = hcat(out.state.ρ_history...)
us = hcat(out.state.u_history...)

serialize(outdir * "SABCresult_" * fname, out)
writedlm(outdir * "post_population_" * fname * ".csv", pop, ',') 
writedlm(outdir * "epsilon_history_" * fname * ".csv", eps, ',')
writedlm(outdir * "rho_history_" * fname * ".csv", rhos, ',')
writedlm(outdir * "u_history_" * fname * ".csv", us, ',')
