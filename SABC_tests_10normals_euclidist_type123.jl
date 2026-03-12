import Pkg
# Activate environment. 
Pkg.activate("SABC") 

#= Current environment is visible in Pkg REPL (type']' to activate Pkg REPL)
In Pkg REPL (activated with ']') do:
'add ...path to.../SimulatedAnnealingABC.jl' 
to have the local package installed
OR (after loading Revise)
'dev ...path to.../SimulatedAnnealingABC.jl' to develop package =#

#= To add dependencies:
pkg> activate /Users/ulzg/SABC/SimulatedAnnealingABC.jl
pkg> add PkgName =#
                   
using Revise
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
using Dates
using LinearAlgebra
using Serialization

include("./AffineInvMCMC.jl")
using .AffineInvMCMC

println(" ")
println("---- ------------------------------ ----")
println("---- 10 independent normals - Infer only 1 mean ----")
println("---- ------------------------------ ----")

"""
-----------------------------------------------------------------
--- Data
--- Prior
--- True posterior
-----------------------------------------------------------------
"""
# --- Data ---
# Random.seed!(1822)
Random.seed!()
# Mean (to be inferred)
true_μ = 0 
σ = 0.1

# --- Prior ---
μ_min = -σ*10; μ_max = σ*10
prior = Uniform(μ_min, μ_max)


#= function posterior_sample(n_samples=1000)
    rand(Normal(true_μ, σ), n_samples)
end  =#


# --- True posterior ---
llhood = theta -> begin
	m = theta[1];
	return - 0.5 * (true_μ - m)^2 /(σ^2)
end

lprior = theta -> begin
	m = theta[1];
	if (μ_min <= m <= μ_max)
		return 0.0
	else
		return -Inf
	end
end

lprob = theta -> begin
	lp = lprior(theta)
	if isinf(lp) 
		return -Inf
	else
		return lp + llhood(theta)
	end
end

num_samples_true_post = 1000
numdims = 1
numwalkers = 10
thinning = 10
numsamples_perwalker = num_samples_true_post
burnin = 1000;

rng = MersenneTwister(11);
theta0 = Array{Float64}(undef, numdims, numwalkers);
theta0[1,:] = rand(rng, Uniform(μ_min, μ_max), numwalkers);  # mu

chain, llhoodvals = runMCMCsample(lprob, numwalkers, theta0, burnin, 1);
chain, llhoodvals = runMCMCsample(lprob, numwalkers, chain[:, :, end], numsamples_perwalker, thinning);
flatchain, flatllhoodvals = flattenMCMCarray(chain, llhoodvals)

#= P_true = histogram(title = "Normal - mean")
histogram!(P_true, flatchain[1,:], bins=range(μ_min/1, μ_max/1, length=16), 
			fillcolor = :yellow, fillalpha=0.5, label="true post")
display(P_true) =#
########################################################################################

########################################################################################
### Reset RNG seed for the inference
### Use Random.seed!() for random inference outputs  
# Random.seed!(1113)
Random.seed!()

"""
-----------------------------------------------------------------
--- Infer mean value of 1d normal distribution
--- Statistics: 10 stats
-----------------------------------------------------------------
"""
σ_noninf = 1.0
Σ = Diagonal([σ^2; repeat([σ_noninf], 9).^2])

function model(θ)
	μ = [θ; zeros(9)]
	return rand(MvNormal(μ, Σ))
end

# not needed, it's all zeros
# yobs = [[true_μ]; repeat([0],9)]

# --- Model + distance functions ---
function f_dist(θ)
	y = model(θ)	
	# rho = [euclidean(y[ix], yobs[ix]) for ix in 1:size(y,1)]
	dist1 = abs(y[1])
	dist2 = abs.(y[2:end])
	# dist3 = [dist1; dist2]
	# return sqrt(sum(dist3.^2))
	return [dist1; dist2]
end

function f_dist_old(θ)
	y = model(θ)	
	# rho = [euclidean(y[ix], yobs[ix]) for ix in 1:size(y,1)]
	dist1 = abs(y[1])/0.501106
	dist2 = abs.(y[2:end]./0.8)
	dist3 = [dist1; dist2]
	return sqrt(sum(dist3.^2))
end

##################################################################
### Run!
#################################################################
nsim = 2_000_000
vmulti = 10.0
# --- Run for single distance ---
out_singeps = sabc(f_dist, prior; n_particles = 1000, v = 1.0, n_simulation = nsim, algorithm = :single_eps)
display(out_singeps)
# --- Run for multiple distances ---
out_multeps = sabc(f_dist, prior; n_particles = 1000, v = vmulti, n_simulation = nsim, algorithm = :multi_eps)
display(out_multeps)
# --- Run for old distance ---
out_old = sabc(f_dist_old, prior; n_particles = 1000, v = 1.0, n_simulation = nsim, algorithm = :single_eps)
display(out_old)

# # --- Extract populations and epsilons ---
# pop_singeps = hcat(out_singeps.population...)
# eps_singeps = hcat(out_singeps.state.ϵ_history...)
# rho_singeps = hcat(out_singeps.state.ρ_history...)
# u_singeps = hcat(out_singeps.state.u_history...)

# pop_multeps = hcat(out_multeps.population...)
# eps_multeps = hcat(out_multeps.state.ϵ_history...)
# rho_multeps = hcat(out_multeps.state.ρ_history...)
# u_multeps = hcat(out_multeps.state.u_history...)

# serialize("/Users/ulzg/SABC/output/SABCresult_test10stats_s0p01_pr10s_100s_singeps_5M", out_singeps)
# serialize("/Users/ulzg/SABC/output/SABCresult_test10stats_s0p01_pr10s_100s_multeps_5M", out_multeps)
# serialize("/Users/ulzg/SABC/output/SABCresult_test10stats_s0p01_pr10s_100s_hybreps_5M", out_hybreps)

# ------------------------------------------------------------------ #
# to continue from previous run just do:
# fasa = deserialize("/Users/ulzg/SABC/output/SABCresult_test10stats_s0p01_pr10s_100s_singeps_5M")
# out_singeps_2 = update_population!(fasa, f_dist_euclidean, prior; v = 1.0, n_simulation = nsim, type = 1)
# ------------------------------------------------------------------ #

# --- Run for single distance ---
# out_singeps_2 = update_population!(out_singeps, f_dist_euclidean, prior; v = 1.0, n_simulation = nsim, type = 1)
# display(out_singeps_2)
# # --- Run for multiple distances ---
# out_multeps_2 = update_population!(out_multeps, f_dist_euclidean, prior; v = 10.0, n_simulation = nsim, type = 2)
# display(out_multeps_2)
# # --- Run for multi-u-single-epsilon ---
# out_hybreps_2 = update_population!(out_hybreps, f_dist_euclidean, prior; v = 1.0, n_simulation = nsim, type = 3)
# display(out_hybreps_2)

# --- Run for single distance ---
update_population!(out_singeps, f_dist, prior; v = 1.0, n_simulation = nsim)
display(out_singeps)
# --- Run for multiple distances ---
update_population!(out_multeps, f_dist, prior; v = vmulti, n_simulation = nsim)
display(out_multeps)
# --- Run for old distances ---
update_population!(out_old, f_dist_old, prior; v = 1.0, n_simulation = nsim)
display(out_old)


# --- Extract populations and epsilons ---
# pop_singeps = hcat(out_singeps_2.population...)
# eps_singeps = hcat(out_singeps_2.state.ϵ_history...)
# rho_singeps = hcat(out_singeps_2.state.ρ_history...)
# u_singeps = hcat(out_singeps_2.state.u_history...)

# pop_multeps = hcat(out_multeps_2.population...)
# eps_multeps = hcat(out_multeps_2.state.ϵ_history...)
# rho_multeps = hcat(out_multeps_2.state.ρ_history...)
# u_multeps = hcat(out_multeps_2.state.u_history...)

# pop_hybreps = hcat(out_hybreps_2.population...)
# eps_hybreps = hcat(out_hybreps_2.state.ϵ_history...)
# rho_hybreps = hcat(out_hybreps_2.state.ρ_history...)
# u_hybreps = hcat(out_hybreps_2.state.u_history...)

# pop_singeps = hcat(out_singeps.population...)
# eps_singeps = hcat(out_singeps.state.ϵ_history...)
# rho_singeps = hcat(out_singeps.state.ρ_history...)
# u_singeps = hcat(out_singeps.state.u_history...)

# pop_multeps = hcat(out_multeps.population...)
# eps_multeps = hcat(out_multeps.state.ϵ_history...)
# rho_multeps = hcat(out_multeps.state.ρ_history...)
# u_multeps = hcat(out_multeps.state.u_history...)


# serialize("/Users/ulzg/SABC/output//SABCresult_test10stats_s0p01_pr10s_100s_singeps_10M", out_singeps_2)
# serialize("/Users/ulzg/SABC/output//SABCresult_test10stats_s0p01_pr10s_100s_multeps_10M", out_multeps_2)
# serialize("/Users/ulzg/SABC/output//SABCresult_test10stats_s0p01_pr10s_100s_hybreps_10M", out_hybreps_2)

# ------------------------------------------------------------------ #
# to continue from previous run just do:
# fasa = deserialize("/Users/ulzg/SABC/output/SABCresult_test10stats_s0p01_pr10s_100s_singeps_10M")
# out_singeps_3 = update_population!(fasa, f_dist_euclidean, prior; v = 1.0, n_simulation = nsim, type = 1)
# ------------------------------------------------------------------ #

# --- Run for single distance ---
update_population!(out_singeps, f_dist, prior; v = 1.0, n_simulation = nsim)
display(out_singeps)
# --- Run for multiple distances ---
update_population!(out_multeps, f_dist, prior; v = vmulti, n_simulation = nsim)
display(out_multeps)
# --- Run for old distances ---
update_population!(out_old, f_dist_old, prior; v = 1.0, n_simulation = nsim)
display(out_old)

# --- Extract populations and epsilons ---
pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)
rho_singeps = hcat(out_singeps.state.ρ_history...)
u_singeps = hcat(out_singeps.state.u_history...)

pop_multeps = hcat(out_multeps.population...)
eps_multeps = hcat(out_multeps.state.ϵ_history...)
rho_multeps = hcat(out_multeps.state.ρ_history...)
u_multeps = hcat(out_multeps.state.u_history...)

pop_old = hcat(out_old.population...)
eps_old = hcat(out_old.state.ϵ_history...)
rho_old = hcat(out_old.state.ρ_history...)
u_old = hcat(out_old.state.u_history...)

# serialize("/Users/ulzg/SABC/output//SABCresult_test10stats_s0p01_pr10s_100s_singeps_15M", out_singeps_3)
# serialize("/Users/ulzg/SABC/output//SABCresult_test10stats_s0p01_pr10s_100s_multeps_15M", out_multeps_3)
# serialize("/Users/ulzg/SABC/output//SABCresult_test10stats_s0p01_pr10s_100s_hybreps_15M", out_hybreps_3)

# ------------------------------------------------------------------ #
# to continue from previous run just do:
# fasa = deserialize("/Users/ulzg/SABC/output/SABCresult_test10stats_s0p01_pr10s_100s_singeps_15M")
# out_singeps_4 = update_population!(fasa, f_dist_euclidean, prior; v = 1.0, n_simulation = nsim, type = 1)
# ------------------------------------------------------------------ #


# --- Plot histograms ---
bins_range_factor = 1
bins_range_length = 30
ylim = 300
P_post_1 = histogram(title = "Normal - mean 1 - 10 stats", 
				bins=range(μ_min/bins_range_factor, μ_max/bins_range_factor, length=bins_range_length), ylims=(0,ylim))
histogram!(P_post_1, pop_singeps[1,:], fillcolor = :green2, fillalpha=0.5, 
			bins=range(μ_min/bins_range_factor, μ_max/bins_range_factor, length=bins_range_length), label="single eps")
histogram!(P_post_1, flatchain[1,:], fillcolor = :yellow, fillalpha=0.5, 
			bins=range(μ_min/bins_range_factor, μ_max/bins_range_factor, length=bins_range_length), label="true post")
P_post_2 = histogram(bins=range(μ_min/bins_range_factor, μ_max/bins_range_factor, length=bins_range_length), ylims=(0,ylim))
histogram!(P_post_2, pop_multeps[1,:], fillcolor = :coral, fillalpha=0.5, 
			bins=range(μ_min/bins_range_factor, μ_max/bins_range_factor, length=bins_range_length), label="multi eps")
histogram!(P_post_2, flatchain[1,:], fillcolor = :yellow, fillalpha=0.5, 
			bins=range(μ_min/bins_range_factor, μ_max/bins_range_factor, length=bins_range_length), label="true post")
P_post_3 = histogram(bins=range(μ_min/bins_range_factor, μ_max/bins_range_factor, length=bins_range_length), ylims=(0,ylim))
histogram!(P_post_3, pop_old[1,:], fillcolor = :skyblue1, fillalpha=0.5, 
			bins=range(μ_min/bins_range_factor, μ_max/bins_range_factor, length=bins_range_length), label="old")
histogram!(P_post_3, flatchain[1,:], fillcolor = :yellow, fillalpha=0.5, 
			bins=range(μ_min/bins_range_factor, μ_max/bins_range_factor, length=bins_range_length), label="true post")
display(plot(P_post_1, P_post_2, P_post_3, layout = (1, 3), size=(1200,400)))

# --- Plot rho ---
P_r = plot(title="rho - 10 stats", legend = :bottomleft)
for ix in 3:10
	plot!(P_r, rho_singeps[ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :green4, linewidth=2, thickness_scaling = 1)
	plot!(P_r, rho_multeps[ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
end
plot!(P_r, rho_singeps[2,1:end], xaxis=:log, yaxis=:log, label="single eps - other stats", 
		linecolor = :green4, linewidth=2, thickness_scaling = 1)
plot!(P_r, rho_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - other stats", 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
plot!(P_r, rho_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean 1", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_old[1,1:end], xaxis=:log, yaxis=:log, label="old", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
display(P_r)

# --- Plot u ---
P_u = plot(title="u - 10 stats", legend = :bottomleft)
for ix in 3:10
	plot!(P_u, u_singeps[ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :green4, linewidth=2, thickness_scaling = 1)
	plot!(P_u, u_multeps[ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
end
plot!(P_u, u_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - other stats", 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
plot!(P_u, u_singeps[2,1:end], xaxis=:log, yaxis=:log, label="single eps - other stats", 
		linecolor = :green4, linewidth=2, thickness_scaling = 1)
plot!(P_u, u_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_old[1,1:end], xaxis=:log, yaxis=:log, label="old", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
display(P_u)

# --- Plot eps ---
P_eps = plot(title="epsilon - 10 stats", legend = :bottomleft)
for ix in 3:10
	plot!(P_eps, eps_multeps[ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
end
plot!(P_eps, eps_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - other stats", 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_old[1,1:end], xaxis=:log, yaxis=:log, label="old", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
display(P_eps)

# --- Plot eps (linear scale) ---
P_eps = plot(title="epsilon - 10 stats", legend = :topright, ylims=[-0.1,5])
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[1,1:end], xaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_old[1,1:end], xaxis=:log, label="old", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
display(P_eps)


##################################################################
##################################################################
### Kullback - Leibler divergence 
#################################################################
##################################################################
using KernelDensity
using QuadGK

true_samples = flatchain[1,:]  
sampled1_samples = pop_singeps[1,:]
sampled2_samples = pop_multeps[1,:]

# Estimate PDFs using Kernel Density Estimation (KDE)
kde_q = kde(true_samples)
kde_p1 = kde(sampled1_samples)
kde_p2 = kde(sampled2_samples)

# Define the function to integrate for KL divergence
function kl_integrand(x, kde_p, kde_q)
    p_x = pdf(kde_p, x)
    q_x = pdf(kde_q, x)
	if p_x < 1e-7
		p_x = 0.0
	elseif q_x < 1e-7
		q_x = 0.0
	end
    if p_x == 0.0 || q_x == 0.0
        return 0.0
    else
        return p_x * log(p_x / q_x)
    end
end

# Choose a range for integration
x_min = minimum([minimum(true_samples), minimum(sampled1_samples), minimum(sampled2_samples)])
x_max = maximum([maximum(true_samples), maximum(sampled1_samples), maximum(sampled2_samples)])

# Compute KL divergence for P1 and Q
kl_divergence_p1_q, error_estimate_p1_q = quadgk(x -> kl_integrand(x, kde_p1, kde_q), x_min, x_max)

# Compute KL divergence for P2 and Q
kl_divergence_p2_q, error_estimate_p2_q = quadgk(x -> kl_integrand(x, kde_p2, kde_q), x_min, x_max)

println("KL Divergence P1 || Q: ", kl_divergence_p1_q)
println("Estimated Absolute Error P1 || Q: ", error_estimate_p1_q)
println("KL Divergence P2 || Q: ", kl_divergence_p2_q)
println("Estimated Absolute Error P2 || Q: ", error_estimate_p2_q)

if kl_divergence_p1_q < kl_divergence_p2_q
    println("Sampled posterior P1 is closer to the true posterior Q.")
else
    println("Sampled posterior P2 is closer to the true posterior Q.")
end

x_range = LinRange(x_min, x_max, 1000)
plot(x_range, pdf(kde_q, x_range), label="True Posterior Q", lw=2)
plot!(x_range, pdf(kde_p1, x_range), label="Sampled Posterior P1", lw=2)
plot!(x_range, pdf(kde_p2, x_range), label="Sampled Posterior P2", lw=2)
title!("KDE of True and Sampled Posteriors")
xlabel!("x")
ylabel!("Density")

#= 
##################################################################
##################################################################
### Run a series
#################################################################
##################################################################
length_series = 10

pop_singeps = Vector{Matrix{Float64}}(undef,length_series)
eps_singeps = Vector{Matrix{Float64}}(undef,length_series)
rho_singeps = Vector{Matrix{Float64}}(undef,length_series)
u_singeps = Vector{Matrix{Float64}}(undef,length_series)

pop_multeps = Vector{Matrix{Float64}}(undef,length_series)
eps_multeps = Vector{Matrix{Float64}}(undef,length_series)
rho_multeps = Vector{Matrix{Float64}}(undef,length_series)
u_multeps = Vector{Matrix{Float64}}(undef,length_series)

pop_hybreps = Vector{Matrix{Float64}}(undef,length_series)
eps_hybreps = Vector{Matrix{Float64}}(undef,length_series)
rho_hybreps = Vector{Matrix{Float64}}(undef,length_series)
u_hybreps = Vector{Matrix{Float64}}(undef,length_series)

# --- Reset hyperparameters --- #
nsim = 5_000_000

for irun in 1:length_series
	# --- Run for single distance ---
	out_singeps = sabc(f_dist_euclidean, prior; n_particles = 1000, v = 1.0, n_simulation = nsim, type = 1)
	display(out_singeps)
	# --- Run for multiple distances ---
	out_multeps = sabc(f_dist_euclidean, prior; n_particles = 1000, v = 10.0, n_simulation = nsim, type = 2)
	display(out_multeps)
	# --- Run for multi-u-single-epsilon ---
	out_hybreps = sabc(f_dist_euclidean, prior; n_particles = 1000, v = 1.0, n_simulation = nsim, type = 3)
	display(out_hybreps)

	pop_singeps[irun] = hcat(out_singeps.population...)
	eps_singeps[irun] = hcat(out_singeps.state.ϵ_history...)
	rho_singeps[irun] = hcat(out_singeps.state.ρ_history...)
	u_singeps[irun] = hcat(out_singeps.state.u_history...)

	pop_multeps[irun] = hcat(out_multeps.population...)
	eps_multeps[irun] = hcat(out_multeps.state.ϵ_history...)
	rho_multeps[irun] = hcat(out_multeps.state.ρ_history...)
	u_multeps[irun] = hcat(out_multeps.state.u_history...)

	pop_hybreps[irun] = hcat(out_hybreps.population...)
	eps_hybreps[irun] = hcat(out_hybreps.state.ϵ_history...)
	rho_hybreps[irun] = hcat(out_hybreps.state.ρ_history...)
	u_hybreps[irun] = hcat(out_hybreps.state.u_history...)

end

# --- Rhos ---
P_r = plot(title="Rhos - 10 stats", legend = :bottomright, yticks = [1,10])
plot!(P_r, rho_singeps[1][2,1:end], xaxis=:log, yaxis=:log, label="single eps - other stats", 
		linecolor = :green4, linewidth=2, thickness_scaling = 1)
plot!(P_r, rho_multeps[1][2,1:end], xaxis=:log, yaxis=:log, label="multi eps - other stats", 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
plot!(P_r, rho_hybreps[1][2,1:end], xaxis=:log, yaxis=:log, label="hybrid - other stats", 
		linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
for ix in 3:10
	plot!(P_r, rho_singeps[1][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :green4, linewidth=2, thickness_scaling = 1)
	plot!(P_r, rho_multeps[1][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
	plot!(P_r, rho_hybreps[1][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
end
for irun in 2:length_series
	for ix in 2:10
		plot!(P_r, rho_singeps[irun][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :green4, linewidth=2, thickness_scaling = 1)
		plot!(P_r, rho_multeps[irun][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :red, linewidth=2, thickness_scaling = 1)
		plot!(P_r, rho_hybreps[irun][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
	end
end
plot!(P_r, rho_singeps[1][1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean 1", 
		linecolor = :green2, linewidth=2, thickness_scaling = 1)
plot!(P_r, rho_multeps[1][1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
plot!(P_r, rho_hybreps[1][1,1:end], xaxis=:log, yaxis=:log, label="hybrid - mean 1", 
		linecolor = :skyblue1, linewidth=2, thickness_scaling = 1)
for irun in 2:length_series
	plot!(P_r, rho_singeps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :green2, linewidth=2, thickness_scaling = 1)
	plot!(P_r, rho_multeps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
	plot!(P_r, rho_hybreps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :skyblue1, linewidth=2, thickness_scaling = 1)
end
display(P_r)

# ----------------------------
# To plot only one rho from the series
asa = 6
P_r = plot(title="rho - 10 stats", legend = :bottomright)
for ix in 3:10
	plot!(P_r, rho_singeps[asa][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :green4, linewidth=2, thickness_scaling = 1)
	plot!(P_r, rho_multeps[asa][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
	plot!(P_r, rho_hybreps[asa][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
end
plot!(P_r, rho_singeps[asa][2,1:end], xaxis=:log, yaxis=:log, label="single eps - other stats", 
		linecolor = :green4, linewidth=2, thickness_scaling = 1)
plot!(P_r, rho_multeps[asa][2,1:end], xaxis=:log, yaxis=:log, label="multi eps - other stats", 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
plot!(P_r, rho_hybreps[asa][2,1:end], xaxis=:log, yaxis=:log, label="hybrid - other stats", 
		linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
plot!(P_r, rho_singeps[asa][1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean 1", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_multeps[asa][1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_hybreps[asa][1,1:end], xaxis=:log, yaxis=:log, label="hybrid - mean 1", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
display(P_r)
# ----------------------------

# --- Epsilon ---
P_eps = plot(title="Epsilon - 10 stats", legend = :topright, yticks = [1,10])
plot!(P_eps, eps_multeps[1][2,1:end], xaxis=:log, yaxis=:log, label="multi eps - other stats", 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
for ix in 3:10
	plot!(P_eps, eps_multeps[1][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
end
for irun in 2:length_series
	for ix in 2:10
		plot!(P_eps, eps_multeps[irun][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :red, linewidth=2, thickness_scaling = 1)
	end
end
plot!(P_eps, eps_singeps[1][1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=2, thickness_scaling = 1)
plot!(P_eps, eps_multeps[1][1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
plot!(P_eps, eps_hybreps[1][1,1:end], xaxis=:log, yaxis=:log, label="hybrid", 
		linecolor = :skyblue1, linewidth=2, thickness_scaling = 1)
for irun in 2:length_series
	plot!(P_eps, eps_singeps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :green2, linewidth=2, thickness_scaling = 1)
	plot!(P_eps, eps_multeps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
	plot!(P_eps, eps_hybreps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :skyblue1, linewidth=2, thickness_scaling = 1)
end
display(P_eps)

# ----------------------------
# To plot only one epsilon from the series
asa = 6
P_eps = plot(title="epsilon - 10 stats", legend = :bottomright)
for ix in 3:10
	plot!(P_eps, rho_multeps[asa][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
end
plot!(P_eps, rho_multeps[asa][2,1:end], xaxis=:log, yaxis=:log, label="multi eps - other stats", 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
plot!(P_eps, rho_singeps[asa][1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
plot!(P_eps, rho_multeps[asa][1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_eps, rho_hybreps[asa][1,1:end], xaxis=:log, yaxis=:log, label="hybrid", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
display(P_eps)
# ----------------------------


# --- u's ---
P_u = plot(title="u's - 10 stats", legend = :bottomright, yticks = [1,10])
plot!(P_u, u_multeps[1][2,1:end], xaxis=:log, yaxis=:log, label="multi eps - other stats", 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
plot!(P_u, u_hybreps[1][2,1:end], xaxis=:log, yaxis=:log, label="hybrid - other stats", 
		linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
for ix in 3:10
	plot!(P_u, u_multeps[1][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
	plot!(P_u, u_hybreps[1][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
end
for irun in 2:length_series
	for ix in 2:10
		plot!(P_u, u_multeps[irun][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :red, linewidth=2, thickness_scaling = 1)
		plot!(P_u, u_hybreps[irun][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
	end
end
plot!(P_u, u_singeps[1][1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=2, thickness_scaling = 1)
plot!(P_u, u_multeps[1][1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
plot!(P_u, u_hybreps[1][1,1:end], xaxis=:log, yaxis=:log, label="hybrid - mean 1", 
		linecolor = :skyblue1, linewidth=2, thickness_scaling = 1)
for irun in 2:length_series
	plot!(P_u, u_singeps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :green2, linewidth=2, thickness_scaling = 1)
	plot!(P_u, u_multeps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
	plot!(P_u, u_hybreps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :skyblue1, linewidth=2, thickness_scaling = 1)
end
display(P_u)

# ----------------------------
# To plot only one u from the series
asa = 5
P_u = plot(title="u's - 10 stats", legend = :bottomright)
for ix in 3:10
	plot!(P_u, u_multeps[asa][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
	plot!(P_u, u_hybreps[asa][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
end
plot!(P_u, u_multeps[asa][2,1:end], xaxis=:log, yaxis=:log, label="multi eps - other stats", 
		linecolor = :red, linewidth=2, thickness_scaling = 1)
plot!(P_u, u_hybreps[asa][2,1:end], xaxis=:log, yaxis=:log, label="hybrid - other stats", 
		linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
plot!(P_u, u_singeps[asa][1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :green2, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[asa][1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_hybreps[asa][1,1:end], xaxis=:log, yaxis=:log, label="hybrid - mean 1", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
display(P_u)
# ----------------------------

# --- Posteriors ---
for irun in 1:length_series
	P_post_1 = histogram(title = "Normal - mean 1 - 10 stats", bins=range(μ_min, μ_max, length=11), ylims=(0,600))
	histogram!(P_post_1, pop_singeps[irun][1,:], fillcolor = :green2, fillalpha=0.5, 
			bins=range(μ_min, μ_max, length=11), label="single eps")
	histogram!(P_post_1, flatchain[1,:], fillcolor = :yellow, fillalpha=0.5, 
			bins=range(μ_min, μ_max, length=11), label="true post")
	P_post_2 = histogram(bins=range(μ_min, μ_max, length=21), ylims=(0,600))
	histogram!(P_post_2, pop_multeps[irun][1,:], fillcolor = :coral, fillalpha=0.5, 
			bins=range(μ_min, μ_max, length=11), label="multi eps")
	histogram!(P_post_2, flatchain[1,:], fillcolor = :yellow, fillalpha=0.5, 
			bins=range(μ_min, μ_max, length=11), label="true post")
	P_post_3 = histogram(bins=range(μ_min, μ_max, length=21), ylims=(0,600))
	histogram!(P_post_3, pop_hybreps[irun][1,:], fillcolor = :skyblue1, fillalpha=0.5, 
			bins=range(μ_min, μ_max, length=11), label="hybrid")
	histogram!(P_post_3, flatchain[1,:], fillcolor = :yellow, fillalpha=0.5, 
			bins=range(μ_min, μ_max, length=11), label="true post")
	display(plot(P_post_1, P_post_2, P_post_3, layout = (1, 3)))
end
 =#

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
#= 
println(" ")
println("---- ------------------------------ ----")
println("---- Two independent normals - Infer mean values ----")
println("---- ------------------------------ ----")

"""
-----------------------------------------------------------------
--- Generate 2d normally distributed data
--- Prior
--- True posterior
-----------------------------------------------------------------
"""
# --- Data ---
# Random.seed!(1822)
Random.seed!()

# true_mean_1 = 0
# true_mean_2 = 0
# sigma_1 = 1
# sigma_2 = 1000
# num_samples = 100
# yobs = hcat([[rand(Normal(true_mean_1, sigma_1)),rand(Normal(true_mean_2, sigma_2))] for ix in 1:num_samples]...)  # generate data
# scatter(yobs[1,:], yobs[2,:], markercolor = :skyblue1, title = "Independent Normals - Data", xlabel = "dim 1", ylabel = "dim 2", xlims = (-5,5), ylims = (-5000,5000))

μ = [0,0]        # Mean (to be inferred)
σ = [1, 1000]  # Std
ρ = 0.0          # Correlation, to construct covariance matrix

σ2 = σ.^2
Σ = [ [σ2[1] ρ*σ2[1]*σ2[2]]; [ρ*σ2[1]*σ2[2] σ2[2]] ]

mvn = MvNormal(μ, Σ)

num_samples = 1
yobs = hcat([0,0]) 

# yobs = rand(mvn, num_samples)
# scatter(yobs[1,:], yobs[2,:], markercolor = :skyblue1, title = "Independent Normals - Data", xlabel = "dim 1", ylabel = "dim 2", xlims = (-5,5), ylims = (-40_000,40_000))
# To plot
# function f(x,y)
# 	pdf(mvn,[x,y])
# end
# x=range(start = -10, stop=10, step=0.1)
# y=range(start = -400, stop=400, step=0.1)
# display(surface(x,y,f))

# --- Prior ---
μ1_min = -5000
μ1_max = 5000
μ2_min = -5000
μ2_max = 5000
prior = product_distribution(Uniform(μ1_min, μ1_max), Uniform(μ2_min, μ2_max))

# --- True posterior ---
llhood = theta -> begin
	m1, m2  = theta;
	return - 0.5 * (((yobs[1] - m1)^2/(σ[1]^2)) + ((yobs[2] - m2)^2/(σ[2]^2)))
end

lprior = theta -> begin
	m1, m2 = theta;
	if (μ1_min <= m1 <= μ1_max) && (μ2_min <= m2 <= μ2_max)
		return 0.0
	else
		return -Inf
	end
end

lprob = theta -> begin
	m1,m2 = theta;
	lp = lprior(theta)
	if isinf(lp) 
		return -Inf
	else
		return lp + llhood(theta)
	end
end

# We want num_samples posterior samples
num_samples_true_post = 1000
numdims = 2
numwalkers = 10
thinning = 10
numsamples_perwalker = num_samples_true_post
burnin = 1000;

rng = MersenneTwister(11);
theta0 = Array{Float64}(undef, numdims, numwalkers);
theta0[1, :] = rand(rng, Uniform(μ1_min, μ1_max), numwalkers);  # mu1
theta0[2, :] = rand(rng, Uniform(μ2_min, μ2_max), numwalkers);  # mu2

chain, llhoodvals = runMCMCsample(lprob, numwalkers, theta0, burnin, 1);
chain, llhoodvals = runMCMCsample(lprob, numwalkers, chain[:, :, end], numsamples_perwalker, thinning);
flatchain, flatllhoodvals = flattenMCMCarray(chain, llhoodvals)

########################################################################################
### Wanna plot the true posterior ?
### Run this!
########################################################################################
μ1_min_plot = -5
μ1_max_plot = 5
μ2_min_plot = μ2_min
μ2_max_plot = μ2_max
P_scatter_1 = scatter(xlims = (μ1_min_plot,μ1_max_plot), ylims = (μ2_min_plot,μ2_max_plot), title = "2d Normal - True posterior (Mean values)",
					xlabel = "μ1", ylabel = "μ2")
scatter!(P_scatter_1, flatchain[1,:], flatchain[2,:], markercolor = :yellow, label="true post")
# display(plot(P_scatter_1, P_scatter_2, layout = (1, 2), aspect_ratio = :equal))
display(plot(P_scatter_1))
########################################################################################

########################################################################################
### Reset RNG seed for the inference
### Use Random.seed!() for random inference outputs  
# Random.seed!(1113)
Random.seed!()

"""
-----------------------------------------------------------------
--- Infer mean values for 2d normal distribution (independent normals)
--- Statistics: empirical means
-----------------------------------------------------------------
"""

# --- Model + distance functions ---
function f_dist_euclidean(θ)
	y = rand(MvNormal(θ, Σ), num_samples)	
	rho = [euclidean(y[ix], yobs[ix]) for ix in 1:size(y,1)]
	return rho
end

##################################################################
### Run!
#################################################################
nsim = 5_000_000
# --- Run for single distance ---
out_singeps = sabc(f_dist_euclidean, prior; n_particles = 1000, v = 1.0, n_simulation = nsim, single = true)
display(out_singeps)
# --- Run for multiple distances ---
out_multeps = sabc(f_dist_euclidean, prior; n_particles = 1000, v = 10.0, n_simulation = nsim, single = false)
display(out_multeps)

pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)
rho_singeps = hcat(out_singeps.state.ρ_history...)
u_singeps = hcat(out_singeps.state.u_history...)
r_singeps = hcat(out_singeps.state.ρ_history...)

pop_multeps = hcat(out_multeps.population...)
eps_multeps = hcat(out_multeps.state.ϵ_history...)
rho_multeps = hcat(out_multeps.state.ρ_history...)
u_multeps = hcat(out_multeps.state.u_history...)
r_multeps = hcat(out_multeps.state.ρ_history...)

# --- Plot histograms ---
P_hist_mu1 = histogram(title = "2d Normal - mean 1 - 2 stats")
histogram!(P_hist_mu1, pop_singeps[1,:], bins=range(-250, 250, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="mean 1 - single eps")
histogram!(P_hist_mu1, pop_multeps[1,:], bins=range(-250, 250, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="mean 1 - multi eps")
display(P_hist_mu1)
P_hist_mu2 = histogram(title = "2d Normal - mean 2 - 2 stats")
histogram!(P_hist_mu2, pop_singeps[2,:], bins=range(-100_000, 100_000, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="mean 2 - single eps")
histogram!(P_hist_mu2, pop_multeps[2,:], bins=range(-100_000, 100_000, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="mean 2 - multi eps")
display(P_hist_mu2)
# --- Scatterplot ---
μ1_min_plot = -50
μ1_max_plot = 50
μ2_min_plot = μ2_min
μ2_max_plot = μ2_max
P_scatter_1 = scatter(title = "2d Normal - 2 stats", xlims = (μ1_min_plot,μ1_max_plot), ylims = (μ2_min_plot,μ2_max_plot), 
					xlabel = "mean 1", ylabel = "mean 2")
scatter!(P_scatter_1, pop_singeps[1,:], pop_singeps[2,:], markercolor = :skyblue1, label="single eps")
scatter!(P_scatter_1, flatchain[1,:], flatchain[2,:], markercolor = :yellow, label="true post")
P_scatter_2 = scatter(xlims = (μ1_min_plot,μ1_max_plot), ylims = (μ2_min_plot,μ2_max_plot), xlabel = "mean 1")
scatter!(P_scatter_2, pop_multeps[1,:], pop_multeps[2,:], markercolor = :coral, label="multi eps")
scatter!(P_scatter_2, flatchain[1,:], flatchain[2,:], markercolor = :yellow, label="true post")
# display(plot(P_scatter_1, P_scatter_2, layout = (1, 2), aspect_ratio = :equal))
display(plot(P_scatter_1, P_scatter_2, layout = (1, 2)))
# --- Plot epsilons ---
P_eps = plot(title="2d Normal - epsilon - 2 stats", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 2", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
display(P_eps)
# --- Plot u ---
P_u = plot(title="2d Normal - u - 2 stats", legend = :bottomleft)
plot!(P_u, u_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 2", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
display(P_u)
# --- Plot rho ---
P_r_sing = plot(title="2d Normal - rho singeps - 2 stats", legend = :bottomleft)
plot!(P_r_sing, r_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean 1", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_r_sing, r_singeps[2,1:end], xaxis=:log, yaxis=:log, label="single eps - mean 2", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
display(P_r_sing)
P_r_multi = plot(title="2d Normal - rho multeps - 2 stats", legend = :bottomleft)
plot!(P_r_multi, r_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_r_multi, r_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 2", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
display(P_r_multi)

P_r = plot(title="2d Normal - rho - 2 stats", legend = :topright)
plot!(P_r, r_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean 1", 
		linecolor = :blue, linewidth=3, thickness_scaling = 1)
plot!(P_r, r_singeps[2,1:end], xaxis=:log, yaxis=:log, label="single eps - mean 2", 
		linecolor = :purple, linewidth=3, thickness_scaling = 1)
plot!(P_r, r_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_r, r_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 2", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
display(P_r)

# update_population!(out_multeps, f_dist_euclidean_multeps, prior; n_simulation = 1000, checkpoint_display = 100)


##################################################################
##################################################################
### Run a series
#################################################################
##################################################################
length_series = 10

pop_singeps = Vector{Matrix{Float64}}(undef,length_series)
eps_singeps = Vector{Matrix{Float64}}(undef,length_series)
rho_singeps = Vector{Matrix{Float64}}(undef,length_series)
u_singeps = Vector{Matrix{Float64}}(undef,length_series)
r_singeps = Vector{Matrix{Float64}}(undef,length_series)

pop_multeps = Vector{Matrix{Float64}}(undef,length_series)
eps_multeps = Vector{Matrix{Float64}}(undef,length_series)
rho_multeps = Vector{Matrix{Float64}}(undef,length_series)
u_multeps = Vector{Matrix{Float64}}(undef,length_series)
r_multeps = Vector{Matrix{Float64}}(undef,length_series)

# --- Reset hyperparameters --- #
nsim = 5_000_000

for irun in 1:length_series
	# --- Run for single distance ---
	out_singeps = sabc(f_dist_euclidean, prior; n_particles = 1000, v = 1.0, n_simulation = nsim, checkpoint_display = 500, single = true)
	display(out_singeps)
	# --- Run for multiple distances ---
	out_multeps = sabc(f_dist_euclidean, prior; n_particles = 1000, v = 10.0, n_simulation = nsim, checkpoint_display = 500, single = false)
	display(out_multeps)

	pop_singeps[irun] = hcat(out_singeps.population...)
	eps_singeps[irun] = hcat(out_singeps.state.ϵ_history...)
	rho_singeps[irun] = hcat(out_singeps.state.ρ_history...)
	u_singeps[irun] = hcat(out_singeps.state.u_history...)
	r_singeps[irun] = hcat(out_singeps.state.ρ_history...)

	pop_multeps[irun] = hcat(out_multeps.population...)
	eps_multeps[irun] = hcat(out_multeps.state.ϵ_history...)
	rho_multeps[irun] = hcat(out_multeps.state.ρ_history...)
	u_multeps[irun] = hcat(out_multeps.state.u_history...)
	r_multeps[irun] = hcat(out_multeps.state.ρ_history...)
end


P_r_multi = plot(title="2d Normal - rho multeps - 2 stats", legend = :bottomleft)
plot!(P_r_multi, r_multeps[1][1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
plot!(P_r_multi, r_multeps[1][2,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 2", 
		linecolor = :green, linewidth=2, thickness_scaling = 1)
for irun in 2:length_series
	plot!(P_r_multi, r_multeps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false,
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
	plot!(P_r_multi, r_multeps[irun][2,1:end], xaxis=:log, yaxis=:log, label=false,
		linecolor = :green, linewidth=2, thickness_scaling = 1)
end
display(P_r_multi)

P_r = plot(title="2d Normal - rho singeps vs multeps - 2 stats", legend = :bottomleft)
plot!(P_r, r_singeps[1][1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean 1", 
		linecolor = :blue, linewidth=2, thickness_scaling = 1)
plot!(P_r, r_singeps[1][2,1:end], xaxis=:log, yaxis=:log, label="single eps - mean 2", 
		linecolor = :purple, linewidth=2, thickness_scaling = 1)
plot!(P_r, r_multeps[1][1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
plot!(P_r, r_multeps[1][2,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 2", 
		linecolor = :green, linewidth=2, thickness_scaling = 1)
for irun in 2:length_series
	plot!(P_r, r_multeps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false,
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
	plot!(P_r, r_multeps[irun][2,1:end], xaxis=:log, yaxis=:log, label=false,
		linecolor = :green, linewidth=2, thickness_scaling = 1)
	plot!(P_r, r_singeps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :blue, linewidth=2, thickness_scaling = 1)
	plot!(P_r, r_singeps[irun][2,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :purple, linewidth=2, thickness_scaling = 1)
end
display(P_r)

##################################################################
##################################################################
##################################################################
asa = 9
P_r_multi = plot(title="2d Normal - rho multeps - 2 stats", legend = :bottomleft)
plot!(P_r_multi, r_multeps[asa][1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
plot!(P_r_multi, r_multeps[asa][2,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 2", 
		linecolor = :green, linewidth=2, thickness_scaling = 1)
display(P_r_multi)

# --- Scatterplot ---
μ1_min_plot = -20
μ1_max_plot = 20
μ2_min_plot = μ2_min
μ2_max_plot = μ2_max
for asa in 1:length_series
	P_scatter_1 = scatter(title = "2d Normal - 2 stats", xlims = (μ1_min_plot,μ1_max_plot), ylims = (μ2_min_plot,μ2_max_plot), 
					xlabel = "mean 1", ylabel = "mean 2")
	scatter!(P_scatter_1, pop_singeps[asa][1,:], pop_singeps[asa][2,:], markercolor = :skyblue1, label="single eps")
	scatter!(P_scatter_1, flatchain[1,:], flatchain[2,:], markercolor = :yellow, label="true post")
	P_scatter_2 = scatter(xlims = (μ1_min_plot,μ1_max_plot), ylims = (μ2_min_plot,μ2_max_plot), xlabel = "mean 1")
	scatter!(P_scatter_2, pop_multeps[asa][1,:], pop_multeps[asa][2,:], markercolor = :coral, label="multi eps")
	scatter!(P_scatter_2, flatchain[1,:], flatchain[2,:], markercolor = :yellow, label="true post")
	# display(plot(P_scatter_1, P_scatter_2, layout = (1, 2), aspect_ratio = :equal))
	display(plot(P_scatter_1, P_scatter_2, layout = (1, 2)))
end

asa = 5
P_u = plot(title="2d Normal - u - 2 stats", legend = :bottomleft)
plot!(P_u, u_multeps[asa][1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[asa][2,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 2", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
display(P_u)

 =#
