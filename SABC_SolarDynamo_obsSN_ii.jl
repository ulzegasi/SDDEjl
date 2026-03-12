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
#### Load observed SN dataset #####
#########################################################################

SNdata_temp = Matrix(CSV.read(datadir * "SN_monthly_mean_total_withYrs.csv", DataFrame, header=false))[:,2]
SNyrs_temp = Matrix(CSV.read(datadir * "SN_monthly_mean_total_withYrs.csv", DataFrame, header=false))[:,1]

Tobs_temp = size(SNdata_temp, 1)
SNdata = reshape(SNdata_temp, Tobs_temp)
SNyrs = reshape(SNyrs_temp, Tobs_temp)

# IMPORTANT: Tobs/nj = period of the nj-th component of the Fourier spectrum
# Jupiter period, Tj = (11.86*12) (time unit is MONTH here!)
# -> Tobs/Tj = nj
# We want nj as close as possible to an integer. 
# Therefore, we remove the first 120 data points (10 years), so that:
# (3251-120)/(11.86*12) = 3131/(11.86*12) = 21.9997
# THE 22nd COMPONENT OF THE FOURIER SPECTRUM IS JUPITER 

# deleteat!(SNdata,1:120)  # when using Jupietr driver
# deleteat!(SNyrs,1:120)   # when using Jupiter driver
# nj = 22

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
# --------------------------
fourier_range = [128, 66, 185, 43, 171, 108, 179, 212, 215, 153, 58] .+ 1  # without Jupiter
# --------------------------
num_sumstats = size(fourier_range, 1)

##### Define reference data (and their stats) #####
fs_obs = ifft(window.*SNdata)[1:120]
ss_obs = [real.(fs_obs); imag.(fs_obs)][fourier_range]

##### Single distance #####
function f_dist_singeps(θ)
		data = sn(θ; Twarmup = 200, Tobs = Tobs_without_warmup)
		windowed_data = window.*data
        fs = ifft(windowed_data)[1:120]
        ss = [real.(fs); imag.(fs)][fourier_range]
        # Distance
        rho = [euclidean(ss, ss_obs), [euclidean(ss[ix], ss_obs[ix]) for ix in 1:num_sumstats]]  # --- alternatives: sqeuclidean, cityblock (cityblock(x, y):=sum(abs(x - y)))
        return rho
end

##### Multiple distances #####
function f_dist_multeps(θ)
	data = sn(θ; Twarmup = 200, Tobs = Tobs_without_warmup)
	windowed_data = window.*data
	fs = ifft(windowed_data)[1:120]
	ss = [real.(fs); imag.(fs)][fourier_range]
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:num_sumstats]
	return rho
end

##### Prior ranges #####
τ_min  = 1.0;   τ_max  = 72.0
T_min  = 1.0;   T_max  = 180.0
Nd_min = 1.0;   Nd_max = 15.0
σ_min  = 0.01;  σ_max  = 0.6
Bmax_min = 1.0; Bmax_max = 15.0

# prior = product_distribution(Uniform(τ_min, τ_max), Uniform(T_min, T_max), Uniform(Nd_min, Nd_max), 
# 							Uniform(σ_min, σ_max), Uniform(Bmax_min, Bmax_max),  Uniform(ϵ_min, ϵ_max), 
# 							Uniform(ϕ_min, ϕ_max))

prior = product_distribution(Uniform(τ_min, τ_max), Uniform(T_min, T_max), Uniform(Nd_min, Nd_max), 
 				Uniform(σ_min, σ_max), Uniform(Bmax_min, Bmax_max))

# --- Output file names ---
fname_singeps = "obsSN_singeps_ii_noJ_10"
fname_multeps = "obsSN_multeps_ii_noJ_10"

# --- Run for single distance ---
out_singeps_1 = sabc(f_dist_singeps, prior; n_particles = 1000, n_simulation = 6_000_000, checkpoint_display = 100)
display(out_singeps_1); flush(stdout)

pop_singeps_1 = hcat(out_singeps_1.population...)
eps_singeps_1 = hcat(out_singeps_1.state.ϵ_history...)
rho_singeps_1 = hcat(out_singeps_1.state.ρ_history...)
u_singeps_1 = hcat(out_singeps_1.state.u_history...)

writedlm(outdir * "post_population_" * fname_singeps * ".csv", pop_singeps_1, ',') 
writedlm(outdir * "epsilon_history_" * fname_singeps * ".csv", eps_singeps_1, ',')
writedlm(outdir * "rho_history_" * fname_singeps * ".csv", rho_singeps_1, ',')
writedlm(outdir * "u_history_" * fname_singeps * ".csv", u_singeps_1, ',')

# --- Re-run for single distance ---
fname_singeps = "obsSN_singeps_ii_noJ_11"
out_singeps_2 = sabc(f_dist_singeps, prior; n_particles = 1000, n_simulation = 6_000_000, checkpoint_display = 100)
display(out_singeps_2); flush(stdout)

pop_singeps_2 = hcat(out_singeps_2.population...)
eps_singeps_2 = hcat(out_singeps_2.state.ϵ_history...)
rho_singeps_2 = hcat(out_singeps_2.state.ρ_history...)
u_singeps_2 = hcat(out_singeps_2.state.u_history...)

writedlm(outdir * "post_population_" * fname_singeps * ".csv", pop_singeps_2, ',') 
writedlm(outdir * "epsilon_history_" * fname_singeps * ".csv", eps_singeps_2, ',')
writedlm(outdir * "rho_history_" * fname_singeps * ".csv", rho_singeps_2, ',')
writedlm(outdir * "u_history_" * fname_singeps * ".csv", u_singeps_2, ',')

# --- Run for multiple distances ---
# out_multeps_1 = sabc(f_dist_multeps, prior; n_particles = 1000, n_simulation = 10_000, v = 84, checkpoint_display = 100)
# display(out_multeps_1); flush(stdout)

# pop_multeps_1 = hcat(out_multeps_1.population...)
# eps_multeps_1 = hcat(out_multeps_1.state.ϵ_history...)
# rho_multeps_1 = hcat(out_multeps_1.state.ρ_history...)
# u_multeps_1 = hcat(out_multeps_1.state.u_history...)

# writedlm(outdir * "post_population_" * fname_multeps * ".csv", pop_multeps_1, ',') 
# writedlm(outdir * "epsilon_history_" * fname_multeps * ".csv", eps_multeps_1, ',')
# writedlm(outdir * "rho_history_" * fname_multeps * ".csv", rho_multeps_1, ',')
# writedlm(outdir * "u_history_" * fname_multeps * ".csv", u_multeps_1, ',')

if run_on_cluster == 0
	# --- Plot histograms ---
	# -----------------------
	P_hist_tau = histogram(title = "τ")
	histogram!(P_hist_tau, pop_singeps_1[1,:], bins=range(τ_min, τ_max, length=16), 
	fillcolor = :skyblue1, fillalpha=0.5, label="single eps")
	histogram!(P_hist_tau, pop_multeps_1[1,:], bins=range(τ_min, τ_max, length=16), 
	fillcolor = :coral, fillalpha=0.5, label="multi eps")
	display(P_hist_tau)
	# -----------------------
	P_hist_T = histogram(title = "T")
	histogram!(P_hist_T, pop_singeps_1[2,:], bins=range(T_min, T_max, length=16), 
	fillcolor = :skyblue1, fillalpha=0.5, label="single eps")
	histogram!(P_hist_T, pop_multeps_1[2,:], bins=range(T_min, T_max, length=16), 
	fillcolor = :coral, fillalpha=0.5, label="multi eps")
	display(P_hist_T)
	# -----------------------
	P_hist_Nd = histogram(title = "Nd")
	histogram!(P_hist_Nd, pop_singeps_1[3,:], bins=range(Nd_min, Nd_max, length=16), 
	fillcolor = :skyblue1, fillalpha=0.5, label="single eps")
	histogram!(P_hist_Nd, pop_multeps_1[3,:], bins=range(Nd_min, Nd_max, length=16), 
	fillcolor = :coral, fillalpha=0.5, label="multi eps")
	display(P_hist_Nd)
	# -----------------------
	P_hist_sig = histogram(title = "σ")
	histogram!(P_hist_sig, pop_singeps_1[4,:], bins=range(σ_min, σ_max, length=16), 
	fillcolor = :skyblue1, fillalpha=0.5, label="single eps")
	histogram!(P_hist_sig, pop_multeps_1[4,:], bins=range(σ_min, σ_max, length=16), 
	fillcolor = :coral, fillalpha=0.5, label="multi eps")
	display(P_hist_sig)
	# -----------------------
	P_hist_Bmax = histogram(title = "Bmax")
	histogram!(P_hist_Bmax, pop_singeps_1[5,:], bins=range(Bmax_min, Bmax_max, length=16), 
	fillcolor = :skyblue1, fillalpha=0.5, label="single eps")
	histogram!(P_hist_Bmax, pop_multeps_1[5,:], bins=range(Bmax_min, Bmax_max, length=16), 
	fillcolor = :coral, fillalpha=0.5, label="multi eps")
	display(P_hist_Bmax)
	# -----------------------
	P_hist_eps = histogram(title = "ϵ")
	histogram!(P_hist_eps, pop_singeps_1[6,:], bins=range(ϵ_min, ϵ_max, length=16), 
	fillcolor = :skyblue1, fillalpha=0.5, label="single eps")
	histogram!(P_hist_eps, pop_multeps_1[6,:], bins=range(ϵ_min, ϵ_max, length=16), 
	fillcolor = :coral, fillalpha=0.5, label="multi eps")
	display(P_hist_eps)
	# -----------------------
	P_hist_phi = histogram(title = "ϕ")
	histogram!(P_hist_phi, pop_singeps_1[7,:], bins=range(ϕ_min, ϕ_max, length=16), 
	fillcolor = :skyblue1, fillalpha=0.5, label="single eps")
	histogram!(P_hist_phi, pop_multeps_1[7,:], bins=range(ϕ_min, ϕ_max, length=16), 
	fillcolor = :coral, fillalpha=0.5, label="multi eps")
	display(P_hist_phi)

	# --- Plot epsilons ---
	P_eps = plot(title="SABC epsilon", legend = :bottomleft)
	plot!(P_eps, eps_singeps_1[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
	linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
	plot!(P_eps, eps_multeps_1[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - τ", 
	linecolor = :coral, linewidth=3, thickness_scaling = 1)
	plot!(P_eps, eps_multeps_1[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - T", 
	linecolor = :green, linewidth=3, thickness_scaling = 1)
	plot!(P_eps, eps_multeps_1[3,1:end], xaxis=:log, yaxis=:log, label="multi eps - Nd", 
	linecolor = :orange, linewidth=3, thickness_scaling = 1)
	plot!(P_eps, eps_multeps_1[4,1:end], xaxis=:log, yaxis=:log, label="multi eps - σ", 
	linecolor = :yellow, linewidth=3, thickness_scaling = 1)
	plot!(P_eps, eps_multeps_1[5,1:end], xaxis=:log, yaxis=:log, label="multi eps - Bmax", 
	linecolor = :blue, linewidth=3, thickness_scaling = 1)
	plot!(P_eps, eps_multeps_1[6,1:end], xaxis=:log, yaxis=:log, label="multi eps - ϵ", 
	linecolor = :purple, linewidth=3, thickness_scaling = 1)
	plot!(P_eps, eps_multeps_1[7,1:end], xaxis=:log, yaxis=:log, label="multi eps - ϕ", 
	linecolor = :grey53, linewidth=3, thickness_scaling = 1)
	display(P_eps)
end
	