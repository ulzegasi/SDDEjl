"""
SABC inference with solar dynamo model
"""

##### Cluster or Local? #####
run_on_cluster = 1  # 0 is local, 1 is cluster
if run_on_cluster ∉ [0, 1]
	error("Set 'run_on_cluster' to 0 or 1")
end
##### First or udpate run? #####
from_previous = 0  # set 0 for first run, 1 for updating previous run
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

# *****************************************************************
# MONTHLY RESOLUTION: USED TO PRODUCE OUTPUTS IN "SABC_obsSN_2.nb"
# *****************************************************************
#
# #########################################################################
# #### Load observed SN dataset #####
# #########################################################################

# SNdata_temp = Matrix(CSV.read(datadir * "SN_monthly_mean_total_withYrs.csv", DataFrame, header=false))[:,2]
# SNyrs_temp = Matrix(CSV.read(datadir * "SN_monthly_mean_total_withYrs.csv", DataFrame, header=false))[:,1]

# Tobs_temp = size(SNdata_temp, 1)
# SNdata = reshape(SNdata_temp, Tobs_temp)
# SNyrs = reshape(SNyrs_temp, Tobs_temp)

# # IMPORTANT: Tobs/nj = period of the nj-th component of the Fourier spectrum
# # Jupiter period, Tj = (11.86*12) (time unit is MONTH here!)
# # -> Tobs/Tj = nj
# # We want nj as close as possible to an integer. 
# # Therefore, we remove the first 120 data points (10 years), so that:
# # (3251-120)/(11.86*12) = 3131/(11.86*12) = 21.9997
# # THE 22nd COMPONENT OF THE FOURIER SPECTRUM IS JUPITER 

# # deleteat!(SNdata,1:120)  # when using Jupietr driver
# # deleteat!(SNyrs,1:120)   # when using Jupiter driver
# # nj = 22

# Tobs_without_warmup = size(SNdata, 1)
# #########################################################################
# #########################################################################

#########################################################################
#### Load observed SN dataset #####
#########################################################################

#= SNdata_temp = Matrix(CSV.read(datadir * "SN_monthly_mean_total_withYrs.csv", DataFrame, header=false))[:,2]
SNmonths_temp = Matrix(CSV.read(datadir * "SN_monthly_mean_total_withYrs.csv", DataFrame, header=false))[:,1]

Tobs_temp = size(SNdata_temp, 1)
SNdata_monthly = reshape(SNdata_temp, Tobs_temp)
SNmonths = reshape(SNmonths_temp, Tobs_temp)

# ------------------------------------------------------
# IMPORTANT 1: We don't need monthly resolution
# Switch to yearly resolution
SNyrs = collect(1749.5:1:2019.5)
SNdata = map(x -> mean(collect(x)), Iterators.partition(SNdata_monthly, 12)) =#

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

# ------------------------------------------------------
# IMPORTANT 2: Tobs/nj = period of the nj-th component of the Fourier spectrum
# Jupiter period, Tj = (11.86*12) (time unit is MONTH here!)
# -> Tobs/Tj = nj
# We want nj as close as possible to an integer. 
# Therefore, we remove the first 120 data points (10 years), so that:
# (3251-120)/(11.86*12) = 3131/(11.86*12) = 21.9997
# THE 22nd COMPONENT OF THE FOURIER SPECTRUM IS JUPITER 
# ---> IF WE USE YEARLY RESOLUTION, WE DELETE THE FIRST 10 POINTS (10 YEARS)
# The 22nd componet is still Jupiter.

# deleteat!(SNdata_monthly,1:120)  # when using Jupiter driver
# deleteat!(SNmonths,1:120)   # when using Jupiter driver

# ------ --------------------------------------------- ------ #
# ------ If you want to add Jupiter line to the stats: ------ #
# ------ --------------------------------------------- ------ #
# deleteat!(SNdata,1:10)  
# deleteat!(SNyrs,1:10)
# nj = 22
# ------ --------------------------------------------- ------ #
# ------ --------------------------------------------- ------ #

# ------ --------------------------------------------- ------ #
# ------ If you want to exclude cycles from the inference: ------ #
# ------ --------------------------------------------- ------ #
# To exclude cycle 24 (use only 1 to 23):
# resize!(SNyrs, length(SNyrs) - 11)
# resize!(SNdata, length(SNdata) - 11)
# To exclude cycle 23 (use only 1 to 22): 
# resize!(SNyrs, length(SNyrs) - 23)
# resize!(SNdata, length(SNdata) - 23)
# To exclude cycle 22 (use only 1 to 21):
# resize!(SNyrs, length(SNyrs) - 33)
# resize!(SNdata, length(SNdata) - 33)
# ------ --------------------------------------------- ------ #
# ------ --------------------------------------------- ------ #

# --- First 18 points of the SN record can be used as history (1749 - 1766) ---
# --- Delete the first 17 data points. SN data begins at 1766.
# --- Last point of history is = to first data point (1766)

# deleteat!(SNyrs, 1:17)
# deleteat!(SNdata, 1:17)

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
# [65, 1, 2, 142, 145, 113, 28, 29]
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
# ss_obs = summary_statistics_ii(SNdata; fourier_range=fourier_range)

num_sumstats = size(ss_obs,1)

##### Distance #####
function f_dist(θ)
	data = sn(θ; Twarmup = 200, Tobs = Tobs_without_warmup)
	ss = summary_statistics(data)
	# ss = summary_statistics_ii(data; fourier_range=fourier_range)
	# Distance
	# rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:num_sumstats]
	rho = [abs(ss[ix] - ss_obs[ix]) for ix in 1:num_sumstats]
	# --- alternatives: sqeuclidean, cityblock (cityblock(x, y):=sum(abs(x - y)))
	return rho
end

# *****************************************************************
# MONTHLY RESOLUTION: USED TO PRODUCE OUTPUTS IN "SABC_obsSN_2.nb"
# *****************************************************************
#
# ##### Prior ranges #####
# τ_min  = 1.0;   τ_max  = 72.0
# T_min  = 1.0;   T_max  = 180.0
# Nd_min = 1.0;   Nd_max = 15.0
# σ_min  = 0.01;  σ_max  = 0.3
# Bmax_min = 1.0; Bmax_max = 15.0

# *****************************************************************
# YEARLY RESOLUTION: USED TO PRODUCE OUTPUTS IN "SABC_obsSN_3.nb"
# *****************************************************************
#
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
	# --- Output file names ---
	fname = "obsSN_multi_77"
	# --- Run ---
	out = sabc(f_dist, prior; n_particles = 1000, n_simulation = 1_000_000_000, 
				show_checkpoint = 500, v = 1.0, 
				algorithm = :multi_eps, proposal = DifferentialEvolution(n_para=length(prior)))
	display(out); flush(stdout)
elseif from_previous == 1
	# --- Output file name ---
	fname = "obsSN_multi_70b"
	# --- Continue from... ---
	fname_previous = "obsSN_multi_70"
	# --- Run ---
	fasa = deserialize(outdir * "SABCresult_" * fname_previous)
	out = update_population!(fasa, f_dist, prior; n_simulation = 600_000_000, 
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
