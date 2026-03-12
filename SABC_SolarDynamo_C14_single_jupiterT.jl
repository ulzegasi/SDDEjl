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

include("./SDDESolarDynamo_jupiter.jl")
using .SDDESolarDynamo_jupiter

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
nj = 78
# N.B.: there is in principle no need to reduce the dataset 
# when Jupiter is not in the model, but
# since we delete only 4 points, we do it in any case

Tobs_without_warmup = size(SNdata, 1)
#########################################################################
#########################################################################

##### Window (needed because stats are Fourier components) #####
# The window function is zero at t = 1 and t = Tobs_without_warmup, and is 1 in the middle
window = [(cos(π*((t-1)- (Tobs_without_warmup-1)/2)/(Tobs_without_warmup-1)))^2 for t in 1:Tobs_without_warmup]
# Cos(π/2) gives 3.7e-33. Change it into zero
window[abs.(window) .< 1e-15] .= 0

SNhat = ifft(SNdata)
if run_on_cluster == 0
	display(plot(SNyrs, [SNdata window.*SNdata], title="Data + Window", xlabel="t", ylabel="SN", linewidth=2, legend = false))
	display(plot(SNyrs, window, title="Window", xlabel="t", ylabel="SN", linewidth=2, legend = false))
	display(plot(abs.(SNhat[1:200]), title="abs(fft(data))", linewidth=2, legend = false))
end

##### Choose Fourier components used as stats #####
Nk = 120
thinstep = 6
fourier_range = [ix for ix in 1:thinstep:Nk]

# Add Jupiter component when using dynamo model with Jupiter driver
#= for ix in nj-2:nj+2
    if !(ix in fourier_range)
        append!(fourier_range, ix)
    end 
end
sort!(fourier_range) =#

num_sumstats = size(fourier_range,1)

##### Define summary stats #####
function sum_stats(data)
	windowed_data = window.*data
	fs = ifft(windowed_data)
	ss = abs.(fs[fourier_range])
	return ss
end

##### Define summary stats for observations #####
ss_obs = sum_stats(SNdata)

##### Distance #####
function f_dist(θ)
	data = sn(θ; Twarmup = 200, Tobs = Tobs_without_warmup, mod = :T)
	ss = sum_stats(data)
	# Distance --- alternatives: sqeuclidean, cityblock (cityblock(x, y):=sum(abs(x - y)))
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:num_sumstats]
	return rho
end

##### Prior ranges #####
τ_min  = 0.1;   τ_max  = 15.0
T_min  = 0.1;   T_max  = 15.0
Nd_min = 1.0;   Nd_max = 15.0
σ_min  = 0.01;  σ_max  = 0.3
Bmax_min = 1.0; Bmax_max = 15.0
ϵ_min = 0.01; ϵ_max = 0.3
ϕ_min = 0.0; ϕ_max = 2.0 * π

prior = product_distribution(Uniform(τ_min, τ_max), Uniform(T_min, T_max), Uniform(Nd_min, Nd_max), 
				Uniform(σ_min, σ_max), Uniform(Bmax_min, Bmax_max),
				Uniform(ϵ_min, ϵ_max), Uniform(ϕ_min, ϕ_max))

if from_previous == 0
	# --- Output file name ---
	fname = "C14_single_JT_1"
	# --- Run ---
	out = sabc(f_dist, prior; n_particles = 1000, n_simulation = 100_000_000, checkpoint_display = 500, v = 1.0, type = "single")
	display(out); flush(stdout)
elseif from_previous == 1
	# --- Output file name ---
	fname = "C14_single_JT_2"
	# --- Continue from... ---
	fname_previous = "C14_single_JT_1"
	# --- Run ---
	fasa = deserialize(outdir * "SABCresult_" * fname_previous)
	out = update_population!(fasa, f_dist, prior; n_simulation = 100_000_000, checkpoint_display = 500, v = 1.0, type = "single")
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