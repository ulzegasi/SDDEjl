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

include("./AffineInvMCMC.jl")
using .AffineInvMCMC

"""
--------------------------------
Set hyper-parameters
--------------------------------
"""
nsim = 2_000_000  # total number of particle updates 
v = 1.0           # annealing speed 


println(" ")
println("---- ------------------------------ ----")
println("---- 1d Normal - Infer mean and std ----")
println("---- ------------------------------ ----")

"""
-----------------------------------------------------------------
--- Generate normally distributed data
--- Prior
--- True posterior
-----------------------------------------------------------------
"""
# --- Data ---
# Random.seed!(1822)
Random.seed!()
true_mean = 3
true_sigma = 15
num_samples = 100

yobs = rand(Normal(true_mean, true_sigma), num_samples)  # generate data
# display(histogram(yobs, bins=20, title = "The data"))  # display it (if you want)

# --- Prior ---
s1_min = -10
s1_max = 20
s2_min = 0
s2_max = 25
prior = product_distribution(Uniform(s1_min, s1_max), Uniform(s2_min, s2_max))

# --- True posterior ---
llhood = theta -> begin
	m, s  = theta;
	return -length(yobs)*log(s) - sum((yobs.-m).^2)/(2*s^2)
end

lprior = theta -> begin
	m, s = theta;
	if (s1_min <= m <= s1_max) && (s2_min <= s <= s2_max)
		return 0.0
	else
		return -Inf
	end
end

lprob = theta -> begin
	m,s = theta;
	lp = lprior(theta)
	if isinf(lp) 
		return -Inf
	else
		return lp + llhood(theta)
	end
end

# We want num_samples posterior samples
numdims = 2
numwalkers = 10
thinning = 10
numsamples_perwalker = num_samples
burnin = 1000;

rng = MersenneTwister(11);
theta0 = Array{Float64}(undef, numdims, numwalkers);
theta0[1, :] = rand(rng, Uniform(s1_min, s1_max), numwalkers);  # mu
theta0[2, :] = rand(rng, Uniform(s2_min, s2_max), numwalkers);  # sigma

chain, llhoodvals = runMCMCsample(lprob, numwalkers, theta0, burnin, 1);
chain, llhoodvals = runMCMCsample(lprob, numwalkers, chain[:, :, end], numsamples_perwalker, thinning);
flatchain, flatllhoodvals = flattenMCMCarray(chain, llhoodvals)

########################################################################################
### Wanna plot the true posterior ?
### Run this!
########################################################################################
P_scatter_1 = scatter(xlims = (s2_min,s2_max), ylims = (s1_min,s1_max), title = "1d Normal - True posterior",
					xlabel = "std", ylabel = "mean")
scatter!(P_scatter_1, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
P_scatter_2 = scatter(xlims = (s2_min,s2_max), ylims = (s1_min,s1_max), xlabel = "std")
scatter!(P_scatter_2, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
# display(plot(P_scatter_1, P_scatter_2, layout = (1, 2), aspect_ratio = :equal))
display(plot(P_scatter_1, P_scatter_2, layout = (1, 2)))
########################################################################################

########################################################################################
### Reset RNG seed for the inference
### Use Random.seed!() for random inference outputs  
# Random.seed!(1113)
Random.seed!()
########################################################################################

"""
-----------------------------------------------------------------
--- Infer mean and std for 1d normal distribution
--- Statistics: mean, std
-----------------------------------------------------------------
"""

sleep(0.5)
println(" ")
println("---- ---------- 2 stats ---------- ----")

# --- Summary stats ---
s1obs = mean(yobs)
s2obs = std(yobs) 
ss_obs = [s1obs, s2obs]

# --- Model + distance functions ---
function f_dist_euclidean_singeps(θ)
	# Data-generating model
	y = rand(Normal(θ[1],θ[2]), num_samples)
	# Summary stats
	s1 = mean(y); s2 = std(y)
	ss = [s1, s2]
	# Distance
	rho = [euclidean(ss, ss_obs), [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]]
	return rho
end

function f_dist_euclidean_multeps(θ)
	# Data-generating model
	y = rand(Normal(θ[1],θ[2]), num_samples)
	# Summary stats
	s1 = mean(y); s2 = std(y)
	ss = [s1, s2]
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end

##################################################################

##################################################################
### Wanna run multiple trajectories ?
### Run this!
#################################################################
#= 
num_traj = 200
# --- Run for single distance ---
out_singeps = Array{Any}(undef, num_traj)
for id in 1:num_traj
	out_singeps[id] = sabc(f_dist_euclidean_singeps, prior; n_particles = 1000, n_simulation = nsim, v = 1.2)
end
# --- Run for multiple distances ---
out_multeps = Array{Any}(undef, num_traj)
for id in 1:num_traj
	out_multeps[id] = sabc(f_dist_euclidean_multeps, prior; n_particles = 1000, n_simulation = nsim, v = 1.2)
end

# --- Extract populations and epsilons ---
pop_singeps = Array{Any}(undef, num_traj)
eps_singeps = Array{Any}(undef, num_traj)
rho_singeps = Array{Any}(undef, num_traj)
u_singeps = Array{Any}(undef, num_traj)
for ix in 1:num_traj
	pop_singeps[ix] = hcat(out_singeps[ix].population...)
	eps_singeps[ix] = hcat(out_singeps[ix].state.ϵ_history...)
	rho_singeps[ix] = hcat(out_singeps[ix].state.ρ_history...)
	u_singeps[ix] = hcat(out_singeps[ix].state.u_history...)
end

pop_multeps = Array{Any}(undef, num_traj)
eps_multeps = Array{Any}(undef, num_traj)
rho_multeps = Array{Any}(undef, num_traj)
u_multeps = Array{Any}(undef, num_traj)
for ix in 1:num_traj
	pop_multeps[ix] = hcat(out_multeps[ix].population...)
	eps_multeps[ix] = hcat(out_multeps[ix].state.ϵ_history...)
	rho_multeps[ix] = hcat(out_multeps[ix].state.ρ_history...)
	u_multeps[ix] = hcat(out_multeps[ix].state.u_history...)
end

fname_singeps = "test_normal_2stats_singeps_100samples_seed1113"
fname_multeps = "test_normal_2stats_multeps_100samples_seed1113"

for ix in 1:num_traj
	writedlm("/Users/ulzg/SABC/output/multi_trajectory_test/post_population_" * fname_singeps * "_" * string(ix) * ".csv", pop_singeps[ix], ',') 
	writedlm("/Users/ulzg/SABC/output/multi_trajectory_test/epsilon_history_" * fname_singeps * "_" * string(ix) * ".csv", eps_singeps[ix], ',')
	writedlm("/Users/ulzg/SABC/output/multi_trajectory_test/rho_history_" * fname_singeps * "_" * string(ix) * ".csv", rho_singeps[ix], ',')
	writedlm("/Users/ulzg/SABC/output/multi_trajectory_test/u_history_" * fname_singeps * "_" * string(ix) * ".csv", u_singeps[ix], ',')

	writedlm("/Users/ulzg/SABC/output/multi_trajectory_test/post_population_" * fname_multeps * "_" * string(ix) * ".csv", pop_multeps[ix], ',') 
	writedlm("/Users/ulzg/SABC/output/multi_trajectory_test/epsilon_history_" * fname_multeps * "_" * string(ix) * ".csv", eps_multeps[ix], ',')
	writedlm("/Users/ulzg/SABC/output/multi_trajectory_test/rho_history_" * fname_multeps * "_" * string(ix) * ".csv", rho_multeps[ix], ',')
	writedlm("/Users/ulzg/SABC/output/multi_trajectory_test/u_history_" * fname_multeps * "_" * string(ix) * ".csv", u_multeps[ix], ',')
end
 =#
##################################################################

##################################################################
### Single trajectory?
### Run this!
#################################################################
# --- Run for single distance ---
out_singeps = sabc(f_dist_euclidean_singeps, prior; n_particles = 1000, v = v, n_simulation = nsim)
display(out_singeps)
# --- Run for multiple distances ---
out_multeps = sabc(f_dist_euclidean_multeps, prior; n_particles = 1000, v = v, n_simulation = nsim)
display(out_multeps)

pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)
rho_singeps = hcat(out_singeps.state.ρ_history...)
u_singeps = hcat(out_singeps.state.u_history...)

pop_multeps = hcat(out_multeps.population...)
eps_multeps = hcat(out_multeps.state.ϵ_history...)
rho_multeps = hcat(out_multeps.state.ρ_history...)
u_multeps = hcat(out_multeps.state.u_history...)

##################################################################
### Wanna save outputs ?
### Run this!
#################################################################
#= 
fname_singeps = "test_normal_2stats_singeps_100samples_seed1822"
fname_multeps = "test_normal_2stats_multeps_100samples_seed1822"

writedlm("/Users/ulzg/SABC/output/post_population_" * fname_singeps * ".csv", pop_singeps, ',') 
writedlm("/Users/ulzg/SABC/output/epsilon_history_" * fname_singeps * ".csv", eps_singeps, ',')
writedlm("/Users/ulzg/SABC/output/rho_history_" * fname_singeps * ".csv", rho_singeps, ',')
writedlm("/Users/ulzg/SABC/output/u_history_" * fname_singeps * ".csv", u_singeps, ',')

writedlm("/Users/ulzg/SABC/output/post_population_" * fname_multeps * ".csv", pop_multeps, ',') 
writedlm("/Users/ulzg/SABC/output/epsilon_history_" * fname_multeps * ".csv", eps_multeps, ',')
writedlm("/Users/ulzg/SABC/output/rho_history_" * fname_multeps * ".csv", rho_multeps, ',')
writedlm("/Users/ulzg/SABC/output/u_history_" * fname_multeps * ".csv", u_multeps, ',')
 =#

# --- Plot histograms ---
P_hist_mu = histogram(title = "1d Normal - mean - 2 stats")
histogram!(P_hist_mu, pop_singeps[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="mean - single eps")
histogram!(P_hist_mu, pop_multeps[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="mean - multi eps")
display(P_hist_mu)
P_hist_sd = histogram(title = "1d Normal - std - 2 stats")
histogram!(P_hist_sd, pop_singeps[2,:], bins=range(0, 30, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="std - single eps")
histogram!(P_hist_sd, pop_multeps[2,:], bins=range(0, 30, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="std - multi eps")
display(P_hist_sd)

# --- Scatterplot ---
P_scatter_1 = scatter(xlims = (s2_min,s2_max), ylims = (s1_min,s1_max), title = "1d Normal - 2 stats",
					xlabel = "std", ylabel = "mean")
scatter!(P_scatter_1, pop_singeps[2,:], pop_singeps[1,:], markercolor = :skyblue1, label="single eps")
scatter!(P_scatter_1, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
P_scatter_2 = scatter(xlims = (s2_min,s2_max), ylims = (s1_min,s1_max), xlabel = "std")
scatter!(P_scatter_2, pop_multeps[2,:], pop_multeps[1,:], markercolor = :coral, label="multi eps")
scatter!(P_scatter_2, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
# display(plot(P_scatter_1, P_scatter_2, layout = (1, 2), aspect_ratio = :equal))
display(plot(P_scatter_1, P_scatter_2, layout = (1, 2)))

# --- Plot epsilons ---
P_eps = plot(title="1d Normal - epsilon - 2 stats", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - std", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
display(P_eps)

# --- Plot u ---
P_u = plot(title="1d Normal - u - 2 stats", legend = :bottomleft)
plot!(P_u, u_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - std", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
display(P_u)

# --- Plot rho ---
P_r = plot(title="1d Normal - rho - 2 stats", legend = :bottomleft)
plot!(P_r, rho_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_singeps[2,1:end], xaxis=:log, yaxis=:log, label="single eps - std", 
		linecolor = :purple, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_r, rho_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - std", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
display(P_r)
##################################################################

"""
-----------------------------------------------------------------
--- Infer mean and std for 1d normal distribution
--- Statistics: mean, std, median
-----------------------------------------------------------------
"""

sleep(0.5)
println(" ")
println("---- ---------- 3 stats (median) ---------- ----")

# --- Summary stats ---
s1obs = mean(yobs)
s2obs = std(yobs) 
s3obs = median(yobs)
ss_obs = [s1obs, s2obs, s3obs]

# --- Model + distance functions ---
function f_dist_euclidean_singeps_withmedian(θ)
        # Data-generating model
        y = rand(Normal(θ[1],θ[2]),num_samples)
        # Summary stats
        s1 = mean(y); s2 = std(y); s3 = median(y)
		ss = [s1, s2, s3]
        # Distance
        rho = [euclidean(ss, ss_obs), [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]]
        return rho
end

function f_dist_euclidean_multeps_withmedian(θ)
	# Data-generating model
	y = rand(Normal(θ[1],θ[2]),num_samples)
	# Summary stats
	s1 = mean(y); s2 = std(y); s3 = median(y)
	ss = [s1, s2, s3]
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end


# --- Run for single distance ---
out_singeps = sabc(f_dist_euclidean_singeps_withmedian, prior; n_particles = 1000, v = v, n_simulation = nsim)
display(out_singeps)
# --- Run for multiple distances ---
out_multeps = sabc(f_dist_euclidean_multeps_withmedian, prior; n_particles = 1000, v = v, n_simulation = nsim)
display(out_multeps)

# --- Extract populations and epsilons ---
pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)
r_singeps = hcat(out_singeps.state.ρ_history...)
u_singeps = hcat(out_singeps.state.u_history...)

pop_multeps = hcat(out_multeps.population...)
eps_multeps = hcat(out_multeps.state.ϵ_history...)
r_multeps = hcat(out_multeps.state.ρ_history...)
u_multeps = hcat(out_multeps.state.u_history...)

writedlm("/Users/ulzg/SABC/output/post_population_norm_median_singeps.csv", pop_singeps, ',') 
writedlm("/Users/ulzg/SABC/output/epsilon_history_norm_median_singeps.csv", eps_singeps, ',')
writedlm("/Users/ulzg/SABC/output/rho_history_norm_median_singeps.csv", r_singeps, ',')
writedlm("/Users/ulzg/SABC/output/u_history_norm_median_singeps.csv", u_singeps, ',')

writedlm("/Users/ulzg/SABC/output/post_population_norm_median_multeps.csv", pop_multeps, ',') 
writedlm("/Users/ulzg/SABC/output/epsilon_history_norm_median_multeps.csv", eps_multeps, ',')
writedlm("/Users/ulzg/SABC/output/rho_history_norm_median_multeps.csv", r_multeps, ',')
writedlm("/Users/ulzg/SABC/output/u_history_norm_median_multeps.csv", u_multeps, ',')

# --- Plot histograms ---
P_hist_mu = histogram(title = "1d Normal - mean - 3 stats with median")
histogram!(P_hist_mu, pop_singeps[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="mean - single eps")
histogram!(P_hist_mu, pop_multeps[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="mean - multi eps")
display(P_hist_mu)
P_hist_sd = histogram(title = "1d Normal - std - 3 stats with median")
histogram!(P_hist_sd, pop_singeps[2,:], bins=range(0, 30, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="std - single eps")
histogram!(P_hist_sd, pop_multeps[2,:], bins=range(0, 30, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="std - multi eps")
display(P_hist_sd)
# --- Scatterplot ---
P_scatter_1 = scatter(xlims = (s2_min,s2_max), ylims = (s1_min,s1_max), 
					xlabel = "std", ylabel = "mean", title = "1d Normal - 3 stats with median")
scatter!(P_scatter_1, pop_singeps[2,:], pop_singeps[1,:], markercolor = :skyblue1, label="single eps")
scatter!(P_scatter_1, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
P_scatter_2 = scatter(xlims = (s2_min,s2_max), ylims = (s1_min,s1_max), xlabel = "std")
scatter!(P_scatter_2, pop_multeps[2,:], pop_multeps[1,:], markercolor = :coral, label="multi eps")
scatter!(P_scatter_2, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
display(plot(P_scatter_1, P_scatter_2, layout = (1, 2), aspect_ratio = :equal))
# --- Plot epsilons ---
P_eps = plot(title="1d Normal - epsilon - 3 stats with median", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - std", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[3,1:end], xaxis=:log, yaxis=:log, label="multi eps - median",
		linecolor = :purple1, linewidth=3, thickness_scaling = 1)
display(P_eps)
# --- Plot u ---
P_u = plot(title="1d Normal - u - 3 stats with median", legend = :bottomleft)
plot!(P_u, u_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - std", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[3,1:end], xaxis=:log, yaxis=:log, label="multi eps - median",
		linecolor = :purple1, linewidth=3, thickness_scaling = 1)
display(P_u)


"""
-----------------------------------------------------------------
--- Infer mean and std for 1d normal distribution
--- Statistics: mean, std, some noise
-----------------------------------------------------------------
"""

sleep(0.5)
println(" ")
println("---- ---------- 3 stats (noise) ---------- ----")

# --- Summary stats ---
s1obs = mean(yobs)
s2obs = std(yobs) 
s3obs = randn()
# s3obs = rand(Normal(0, 5))
ss_obs = [s1obs, s2obs, s3obs]

# --- Model + distance functions ---
function f_dist_euclidean_singeps_withnoise(θ)
        # Data-generating model
        y = rand(Normal(θ[1],θ[2]),num_samples)
        # Summary stats
        s1 = mean(y); s2 = std(y); s3 = randn() # rand(Normal(0, 5)) 
		ss = [s1, s2, s3]
        # Distance
        rho = [euclidean(ss, ss_obs), [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]]
        return rho
end

function f_dist_euclidean_multeps_withnoise(θ)
	# Data-generating model
	y = rand(Normal(θ[1],θ[2]),num_samples)
	# Summary stats
	s1 = mean(y); s2 = std(y); s3 = randn() # rand(Normal(0, 5)) 
	ss = [s1, s2, s3]
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end

# --- Run for single distance ---
out_singeps = sabc(f_dist_euclidean_singeps_withnoise, prior; n_particles = 1000, v = 1.0, n_simulation = nsim)
display(out_singeps)
# --- Run for multiple distances ---
out_multeps = sabc(f_dist_euclidean_multeps_withnoise, prior; n_particles = 1000, v = 1.0, n_simulation = nsim)
display(out_multeps)

# --- Extract populations and epsilons ---
pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)
u_singeps = hcat(out_singeps.state.u_history...)
r_singeps = hcat(out_singeps.state.ρ_history...)

pop_multeps = hcat(out_multeps.population...)
eps_multeps = hcat(out_multeps.state.ϵ_history...)
u_multeps = hcat(out_multeps.state.u_history...)
r_multeps = hcat(out_multeps.state.ρ_history...)

writedlm("/Users/ulzg/SABC/output/post_population_norm_noise_singeps.csv", pop_singeps, ',') 
writedlm("/Users/ulzg/SABC/output/epsilon_history_norm_noise_singeps.csv", eps_singeps, ',')
writedlm("/Users/ulzg/SABC/output/rho_history_norm_noise_singeps.csv", r_singeps, ',')
writedlm("/Users/ulzg/SABC/output/u_history_norm_noise_singeps.csv", u_singeps, ',')

writedlm("/Users/ulzg/SABC/output/post_population_norm_noise_multeps.csv", pop_multeps, ',') 
writedlm("/Users/ulzg/SABC/output/epsilon_history_norm_noise_multeps.csv", eps_multeps, ',')
writedlm("/Users/ulzg/SABC/output/rho_history_norm_noise_multeps.csv", r_multeps, ',')
writedlm("/Users/ulzg/SABC/output/u_history_norm_noise_multeps.csv", u_multeps, ',')

# --- Plot histograms ---
P_hist_mu = histogram(title = "1d Normal - mean - 3 stats with noise stat")
histogram!(P_hist_mu, pop_singeps[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="mean - single eps")
histogram!(P_hist_mu, pop_multeps[1,:], bins=range(-10, 20, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="mean - multi eps")
display(P_hist_mu)
P_hist_sd = histogram(title = "1d Normal - std - 3 stats with noise stat")
histogram!(P_hist_sd, pop_singeps[2,:], bins=range(0, 30, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="std - single eps")
histogram!(P_hist_sd, pop_multeps[2,:], bins=range(0, 30, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="std - multi eps")
display(P_hist_sd)
# --- Scatterplot ---
P_scatter_1 = scatter(title = "1d Normal - 3 stats with noise stat", xlims = (s2_min,s2_max), ylims = (s1_min,s1_max), 
					xlabel = "std", ylabel = "mean")
scatter!(P_scatter_1, pop_singeps[2,:], pop_singeps[1,:], markercolor = :skyblue1, label="single eps")
scatter!(P_scatter_1, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
P_scatter_2 = scatter(xlims = (s2_min,s2_max), ylims = (s1_min,s1_max), xlabel = "std")
scatter!(P_scatter_2, pop_multeps[2,:], pop_multeps[1,:], markercolor = :coral, label="multi eps")
scatter!(P_scatter_2, flatchain[2,:], flatchain[1,:], markercolor = :yellow, label="true post")
display(plot(P_scatter_1, P_scatter_2, layout = (1, 2), aspect_ratio = :equal))
# --- Plot epsilons ---
P_eps = plot(title="1d Normal - epsilon - 3 stats with noise stat", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - std", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[3,1:end], xaxis=:log, yaxis=:log, label="multi eps - noise stat",
		linecolor = :purple1, linewidth=3, thickness_scaling = 1)
display(P_eps)
# --- Plot u ---
P_u = plot(title="1d Normal - u - 3 stats with noise stat", legend = :bottomleft)
plot!(P_u, u_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - std", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[3,1:end], xaxis=:log, yaxis=:log, label="multi eps - noise",
		linecolor = :purple1, linewidth=3, thickness_scaling = 1)
display(P_u)
# --- Plot rho ---
P_r_sing = plot(title="1d Normal - rho singeps - 3 stats with noise stat ", legend = :bottomleft)
plot!(P_r_sing, r_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_r_sing, r_singeps[2,1:end], xaxis=:log, yaxis=:log, label="single eps - std", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
plot!(P_r_sing, r_singeps[3,1:end], xaxis=:log, yaxis=:log, label="single eps - noise",
		linecolor = :purple1, linewidth=3, thickness_scaling = 1)
display(P_r_sing)
P_r_multi = plot(title="1d Normal - rho multeps - 3 stats with noise stat ", legend = :bottomleft)
plot!(P_r_multi, r_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_r_multi, r_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - std", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
plot!(P_r_multi, r_multeps[3,1:end], xaxis=:log, yaxis=:log, label="multi eps - noise",
		linecolor = :purple1, linewidth=3, thickness_scaling = 1)
display(P_r_multi)

P_r = plot(title="1d Normal - rho multeps - 3 stats with noise stat", legend = :bottomleft)
plot!(P_r, r_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean", 
		linecolor = :blue, linewidth=2, thickness_scaling = 1)
plot!(P_r, r_singeps[2,1:end], xaxis=:log, yaxis=:log, label="single eps - std", 
		linecolor = :blue, linewidth=2, thickness_scaling = 1)
plot!(P_r, r_singeps[3,1:end], xaxis=:log, yaxis=:log, label="single eps - noise", 
		linecolor = :blue, linewidth=2, thickness_scaling = 1)
plot!(P_r, r_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean", 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
plot!(P_r, r_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - std", 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
plot!(P_r, r_multeps[3,1:end], xaxis=:log, yaxis=:log, label="multi eps - noise", 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
display(P_r)

println(" ")
println("---- ----------------------------- ----")
println("---- NLAR1 - Infer alpha and sigma ----")
println("---- ----------------------------- ----")

sleep(0.5)

"""
-----------------------------------------------------------------
--- NLAR1 model:
--- Data
--- Prior
--- True posterior
-----------------------------------------------------------------
"""

# --- Model ---
function nlf(x::Float64)
	return x^2*(1-x)
end

function nlar1(α, σ; x0 = 0.25, l = 200)
	x = [x0] 
	for ix in 2:l
		push!(x, α*nlf(last(x)) + σ*randn())
	end
	return x
end

# --- Data from Albert et al., SciPost Phys.Core 2022 ---
true_alpha = 5.3
true_sigma = 0.015

yobs = vec(readdlm("/Users/ulzg/SABC/test_data/dataset_c53_s0015_p0025_p200_values.dat"))
# 
display(plot(yobs, title = "NLAR1 data"))  # display it (if you want)
true_posterior = collect(readdlm("/Users/ulzg/SABC/test_data/truePosterior_c53_s0015_p0025_p200.dat")')

# --- Prior ---
a_min = 4.2
a_max = 5.8
s_min = 0.005
s_max = 0.025
prior = product_distribution(Uniform(a_min, a_max), Uniform(s_min, s_max))

"""
-----------------------------------------------------------------
--- Infer α and σ for NLAR1 model
--- Statistics: MLEs for α and σ, order parameter
--- Multi vs single epsilon 
--- Metric: Euclidean
-----------------------------------------------------------------
"""

sleep(0.5)
println(" ")
println("---- ---------- 3 stats ---------- ----")

# --- Summary stats: definition ---
function αhat(x::Vector{Float64})
	num = sum(x[2:end].* nlf.(x[1:end-1]))
	den = sum((nlf.(x)).^2)
	return num/den
end

function σhat(x::Vector{Float64})
	num = sum( (x[2:end] .- αhat(x) .* nlf.(x[1:end-1])).^2 )
	den = length(x)
	# return num/den        # gives larger posterior (in sigma dimension) with single eps
	return sqrt(num/den)  # both single and multi-eps give true post 
end

function order_par(x::Vector{Float64})
	num = sum((nlf.(x)).^2)
	den = length(x)
	return num/den
end


# --- Summary stats: data ---
s1obs = αhat(yobs)
s2obs = σhat(yobs) 
s3obs = order_par(yobs)
ss_obs = [s1obs, s2obs, s3obs]

# --- Model + distance functions ---
function f_dist_euclidean_singeps_3stats(θ)
    α, σ = θ
	# Data-generating model
    y = nlar1(α, σ)
    # Summary stats
    s1 = αhat(y); s2 = σhat(y); s3 = order_par(y)
	ss = [s1, s2, s3]
    # Distance
    rho = [euclidean(ss, ss_obs), [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]]
    return rho
end

function f_dist_euclidean_multeps_3stats(θ)
	α, σ = θ
	# Data-generating model
    y = nlar1(α, σ)
    # Summary stats
    s1 = αhat(y); s2 = σhat(y); s3 = order_par(y)
	ss = [s1, s2, s3]
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end

# --- Run for single distance ---
# "warm-up"
# out_singeps = sabc(f_dist_euclidean_singeps_3stats, prior; n_particles = 1000, v = v, n_simulation = nsim)
# out_multeps = sabc(f_dist_euclidean_multeps_3stats, prior; n_particles = 1000, v = v, n_simulation = nsim)

out_singeps = sabc(f_dist_euclidean_singeps_3stats, prior; n_particles = 1000, v = v, n_simulation = nsim)
display(out_singeps)
# --- Run for multiple distances ---
out_multeps = sabc(f_dist_euclidean_multeps_3stats, prior; n_particles = 1000, v = v, n_simulation = nsim)
display(out_multeps)

# --- Extract populations and epsilons ---
pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)
u_singeps = hcat(out_singeps.state.u_history...)
r_singeps = hcat(out_singeps.state.ρ_history...)

pop_multeps = hcat(out_multeps.population...)
eps_multeps = hcat(out_multeps.state.ϵ_history...)
u_multeps = hcat(out_multeps.state.u_history...)
r_multeps = hcat(out_multeps.state.ρ_history...)

writedlm("/Users/ulzg/SABC/output/post_population_NLAR1_3stats_singeps.csv", pop_singeps, ',') 
writedlm("/Users/ulzg/SABC/output/epsilon_history_NLAR1_3stats_singeps.csv", eps_singeps, ',')
writedlm("/Users/ulzg/SABC/output/rho_history_NLAR1_3stats_singeps.csv", r_singeps, ',')
writedlm("/Users/ulzg/SABC/output/u_history_NLAR1_3stats_singeps.csv", u_singeps, ',')

writedlm("/Users/ulzg/SABC/output/post_population_NLAR1_3stats_multeps.csv", pop_multeps, ',') 
writedlm("/Users/ulzg/SABC/output/epsilon_history_NLAR1_3stats_multeps.csv", eps_multeps, ',')
writedlm("/Users/ulzg/SABC/output/rho_history_NLAR1_3stats_multeps.csv", r_multeps, ',')
writedlm("/Users/ulzg/SABC/output/u_history_NLAR1_3stats_multeps.csv", u_multeps, ',')


# --- Plot histograms ---
P_hist_a = histogram(title = "NLAR1 - alpha - 3 stats")
histogram!(P_hist_a, pop_singeps[1,:], bins=range(a_min, a_max, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="alpha - single eps")
histogram!(P_hist_a, pop_multeps[1,:], bins=range(a_min, a_max, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="alpha - multi eps")
display(P_hist_a)
P_hist_s = histogram(title = "NLAR1 - sigma - 3 stats")
histogram!(P_hist_s, pop_singeps[2,:], bins=range(s_min, s_max, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="sigma - single eps")
histogram!(P_hist_s, pop_multeps[2,:], bins=range(s_min, s_max, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="sigma - multi eps")
display(P_hist_s)
# --- Scatterplot ---
P_scatter_1 = scatter(title = "NLAR1 - 3 stats", xlims = (s_min,s_max), ylims = (a_min,a_max), 
					xlabel = "σ", ylabel = "α")
scatter!(P_scatter_1, pop_singeps[2,:], pop_singeps[1,:], markercolor = :skyblue1, label="single eps")
scatter!(P_scatter_1, true_posterior[2,:], true_posterior[1,:], markercolor = :yellow, label="true post")
P_scatter_2 = scatter(xlims = (s_min,s_max), ylims = (a_min,a_max), xlabel = "σ")
scatter!(P_scatter_2, pop_multeps[2,:], pop_multeps[1,:], markercolor = :coral, label="multi eps")
scatter!(P_scatter_2, true_posterior[2,:], true_posterior[1,:], markercolor = :yellow, label="true post")
display(plot(P_scatter_1, P_scatter_2, layout = (1, 2)))
# --- Plot epsilons ---
P_eps = plot(title="NLAR1 - epsilon - 3 stats", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - α", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - σ", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[3,1:end], xaxis=:log, yaxis=:log, label="multi eps - order par",
		linecolor = :purple1, linewidth=3, thickness_scaling = 1)
display(P_eps)
# --- Plot u ---
P_u = plot(title="NLAR1 - u - 3 stats", legend = :bottomleft)
plot!(P_u, u_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - std", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[3,1:end], xaxis=:log, yaxis=:log, label="multi eps - median",
		linecolor = :purple1, linewidth=3, thickness_scaling = 1)
display(P_u)

"""
-----------------------------------------------------------------
--- Infer α and σ for NLAR1 model
--- Statistics: MLEs for α and σ
-----------------------------------------------------------------
"""

sleep(0.5)
println(" ")
println("---- ---------- 2 stats ---------- ----")

# --- Summary stats: definition ---
function αhat(x::Vector{Float64})
	num = sum(x[2:end].* nlf.(x[1:end-1]))
	den = sum((nlf.(x)).^2)
	return num/den
end

function σhat(x::Vector{Float64})
	num = sum( (x[2:end] .- αhat(x) .* nlf.(x[1:end-1])).^2 )
	den = length(x)
	return num/den
end

function order_par(x::Vector{Float64})
	num = sum((nlf.(x)).^2)
	den = length(x)
	return num/den
end


# --- Summary stats: data ---
s1obs = αhat(yobs)
s2obs = σhat(yobs) 
ss_obs = [s1obs, s2obs]

# --- Model + distance functions ---
function f_dist_euclidean_singeps_2stats(θ)
    α, σ = θ
	# Data-generating model
    y = nlar1(α, σ)
    # Summary stats
    s1 = αhat(y); s2 = σhat(y)
	ss = [s1, s2]
    # Distance
    rho = [euclidean(ss, ss_obs), [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]]
    return rho
end

function f_dist_euclidean_multeps_2stats(θ)
	α, σ = θ
	# Data-generating model
    y = nlar1(α, σ)
    # Summary stats
    s1 = αhat(y); s2 = σhat(y);
	ss = [s1, s2]
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end

# --- Run for single distance ---
out_singeps = sabc(f_dist_euclidean_singeps_2stats, prior; n_particles = 1000, n_simulation = nsim)
display(out_singeps)
# --- Run for multiple distances ---
out_multeps = sabc(f_dist_euclidean_multeps_2stats, prior; n_particles = 1000, n_simulation = nsim)
display(out_multeps)

# --- Extract populations and epsilons ---
pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)
u_singeps = hcat(out_singeps.state.u_history...)
r_singeps = hcat(out_singeps.state.ρ_history...)

pop_multeps = hcat(out_multeps.population...)
eps_multeps = hcat(out_multeps.state.ϵ_history...)
u_multeps = hcat(out_multeps.state.u_history...)
r_multeps = hcat(out_multeps.state.ρ_history...)

writedlm("/Users/ulzg/SABC/output/post_population_NLAR1_2stats_singeps.csv", pop_singeps, ',') 
writedlm("/Users/ulzg/SABC/output/epsilon_history_NLAR1_2stats_singeps.csv", eps_singeps, ',')
writedlm("/Users/ulzg/SABC/output/rho_history_NLAR1_2stats_singeps.csv", r_singeps, ',')
writedlm("/Users/ulzg/SABC/output/u_history_NLAR1_2stats_singeps.csv", u_singeps, ',')

writedlm("/Users/ulzg/SABC/output/post_population_NLAR1_2stats_multeps.csv", pop_multeps, ',') 
writedlm("/Users/ulzg/SABC/output/epsilon_history_NLAR1_2stats_multeps.csv", eps_multeps, ',')
writedlm("/Users/ulzg/SABC/output/rho_history_NLAR1_2stats_multeps.csv", r_multeps, ',')
writedlm("/Users/ulzg/SABC/output/u_history_NLAR1_2stats_multeps.csv", u_multeps, ',')

# --- Plot histograms ---
P_hist_a = histogram(title = "NLAR1 - alpha - 2 stats")
histogram!(P_hist_a, pop_singeps[1,:], bins=range(a_min, a_max, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="alpha - single eps")
histogram!(P_hist_a, pop_multeps[1,:], bins=range(a_min, a_max, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="alpha - multi eps")
display(P_hist_a)
P_hist_s = histogram(title = "NLAR1 - sigma - 2 stats")
histogram!(P_hist_s, pop_singeps[2,:], bins=range(s_min, s_max, length=31), 
			fillcolor = :skyblue1, fillalpha=0.5, label="sigma - single eps")
histogram!(P_hist_s, pop_multeps[2,:], bins=range(s_min, s_max, length=31), 
			fillcolor = :coral, fillalpha=0.5, label="sigma - multi eps")
display(P_hist_s)
# --- Scatterplot ---
P_scatter_1 = scatter(title = "NLAR1 - 2 stats", xlims = (s_min,s_max), ylims = (a_min,a_max), 
					xlabel = "σ", ylabel = "α")
scatter!(P_scatter_1, pop_singeps[2,:], pop_singeps[1,:], markercolor = :skyblue1, label="single eps")
scatter!(P_scatter_1, true_posterior[2,:], true_posterior[1,:], markercolor = :yellow, label="true post")
P_scatter_2 = scatter(xlims = (s_min,s_max), ylims = (a_min,a_max), xlabel = "σ")
scatter!(P_scatter_2, pop_multeps[2,:], pop_multeps[1,:], markercolor = :coral, label="multi eps")
scatter!(P_scatter_2, true_posterior[2,:], true_posterior[1,:], markercolor = :yellow, label="true post")
display(plot(P_scatter_1, P_scatter_2, layout = (1, 2)))
# --- Plot epsilons ---
P_eps = plot(title="NLAR1 - epsilon - 2 stats", legend = :bottomleft)
plot!(P_eps, eps_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - α", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_eps, eps_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - σ", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
display(P_eps)
# --- Plot u ---
P_u = plot(title="NLAR1 - u - 2 stats", legend = :bottomleft)
plot!(P_u, u_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps", 
		linecolor = :skyblue1, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_u, u_multeps[2,1:end], xaxis=:log, yaxis=:log, label="multi eps - std", 
		linecolor = :green, linewidth=3, thickness_scaling = 1)
display(P_u)

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

println(" ")
println("---- ------------------------------ ----")
println("---- Independent normals - Infer mean values ----")
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
function f_dist_euclidean_singeps(θ)
	y = rand(MvNormal(θ, Σ), num_samples)	
	rho = [euclidean(y, yobs), [euclidean(y[ix], yobs[ix]) for ix in 1:size(y,1)]]
	return rho
end

function f_dist_euclidean_multeps(θ)
	y = rand(MvNormal(θ, Σ), num_samples)
	rho = [euclidean(y[ix], yobs[ix]) for ix in 1:size(y,1)]
	return rho
end

##################################################################
### Run!
#################################################################
nsim = 5_000_000
# --- Run for single distance ---
out_singeps = sabc(f_dist_euclidean_singeps, prior; n_particles = 1000, v = 1.0, n_simulation = nsim)
display(out_singeps)
# --- Run for multiple distances ---
out_multeps = sabc(f_dist_euclidean_multeps, prior; n_particles = 1000, v = 1.0, n_simulation = nsim)
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
μ1_min_plot = -1000
μ1_max_plot = 1000
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

P_r = plot(title="2d Normal - rho - 2 stats", ylims=(0,3000),legend = :topright)
plot!(P_r, r_singeps[1,100:end], label="single eps - mean 1", 
		linecolor = :blue, linewidth=3, thickness_scaling = 1)
plot!(P_r, r_singeps[2,100:end], label="single eps - mean 2", 
		linecolor = :purple, linewidth=3, thickness_scaling = 1)
plot!(P_r, r_multeps[1,100:end], label="multi eps - mean 1", 
		linecolor = :coral, linewidth=3, thickness_scaling = 1)
plot!(P_r, r_multeps[2,100:end], label="multi eps - mean 2", 
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
	out_singeps = sabc(f_dist_euclidean_singeps, prior; n_particles = 1000, v = 1.0, n_simulation = nsim, checkpoint_display = 500)
	display(out_singeps)
	# --- Run for multiple distances ---
	out_multeps = sabc(f_dist_euclidean_multeps, prior; n_particles = 1000, v = 1.0, n_simulation = nsim, checkpoint_display = 500)
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



# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

println(" ")
println("---- ------------------------------ ----")
println("---- 10 independent normals - Infer only 1 mean value ----")
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
μ_min = -0.5
μ_max = 0.5
prior = Uniform(μ_min, μ_max)

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

# We want num_samples posterior samples
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

P_true = histogram(title = "Normal - mean")
histogram!(P_true, flatchain[1,:], bins=range(-1, 1, length=31), 
			fillcolor = :yellow, fillalpha=0.5, label="true post")
display(P_true)
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
Σ = Diagonal([σ^2; repeat([10], 9)].^2)

function model(θ)
	μ = [θ; zeros(9)]
	return rand(MvNormal(μ, Σ))
end

yobs = [[40*true_μ]; rand(Normal(0,10),9)]

# --- Model + distance functions ---
function f_dist_euclidean_singeps(θ)
	y = model(θ)	
	y[1] *= 40
	rho = [euclidean(y, yobs), [euclidean(y[ix], yobs[ix]) for ix in 1:size(y,1)]]
	return rho
end

function f_dist_euclidean_multeps(θ)
	y = model(θ)
	y[1] *= 40
	rho = [euclidean(y[ix], yobs[ix]) for ix in 1:size(y,1)]
	return rho
end

##################################################################
### Run!
#################################################################
nsim = 1_000_000
# --- Run for single distance ---
out_singeps = sabc(f_dist_euclidean_singeps, prior; n_particles = 1000, v = 1.0, n_simulation = nsim)
display(out_singeps)
# --- Run for multiple distances ---
out_multeps = sabc(f_dist_euclidean_multeps, prior; n_particles = 1000, v = 10.0, n_simulation = nsim)
display(out_multeps)

pop_singeps = hcat(out_singeps.population...)
eps_singeps = hcat(out_singeps.state.ϵ_history...)
rho_singeps = hcat(out_singeps.state.ρ_history...)
u_singeps = hcat(out_singeps.state.u_history...)

pop_multeps = hcat(out_multeps.population...)
eps_multeps = hcat(out_multeps.state.ϵ_history...)
rho_multeps = hcat(out_multeps.state.ρ_history...)
u_multeps = hcat(out_multeps.state.u_history...)

# --- Plot histograms ---
P_post_1 = histogram(title = "Normal - mean 1 - 10 stats", bins=range(μ_min, μ_max, length=21))
histogram!(P_post_1, pop_singeps[1,:], fillcolor = :skyblue1, fillalpha=0.5, 
			bins=range(μ_min, μ_max, length=21), label="single eps")
histogram!(P_post_1, flatchain[1,:], fillcolor = :yellow, fillalpha=0.5, 
			bins=range(μ_min, μ_max, length=21), label="true post")
P_post_2 = histogram(bins=range(μ_min, μ_max, length=21))
histogram!(P_post_2, pop_multeps[1,:], fillcolor = :coral, fillalpha=0.5, 
			bins=range(μ_min, μ_max, length=21), label="multi eps")
histogram!(P_post_2, flatchain[1,:], fillcolor = :yellow, fillalpha=0.5, 
			bins=range(μ_min, μ_max, length=21), label="true post")
display(plot(P_post_1, P_post_2, layout = (1, 2)))

# --- Plot rho ---
P_r_sing = plot(title="rho singeps - 10 stats", legend = :topright)
plot!(P_r_sing, rho_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean 1", 
		linecolor = :skyblue1, linewidth=2, thickness_scaling = 1)
for ix in 2:10
	plot!(P_r_sing, rho_singeps[ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
end
display(P_r_sing)
P_r_multi = plot(title="rho multeps - 10 stats", legend = :topright)
plot!(P_r_multi, rho_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
for ix in 2:10
	plot!(P_r_multi, rho_multeps[ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :green, linewidth=2, thickness_scaling = 1)
end
display(P_r_multi)

P_r = plot(title="rho - 10 stats", legend = :topright)
plot!(P_r, rho_singeps[1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean 1", 
		linecolor = :skyblue1, linewidth=2, thickness_scaling = 1)
plot!(P_r, rho_multeps[1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
for ix in 2:10
	plot!(P_r, rho_singeps[ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
	plot!(P_r, rho_multeps[ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :green1, linewidth=2, thickness_scaling = 1)
end
display(P_r)


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

# --- Reset hyperparameters --- #
nsim = 5_000_000

for irun in 1:length_series
	# --- Run for single distance ---
	out_singeps = sabc(f_dist_euclidean_singeps, prior; n_particles = 1000, v = 1.0, n_simulation = nsim, checkpoint_display = 500)
	display(out_singeps)
	# --- Run for multiple distances ---
	out_multeps = sabc(f_dist_euclidean_multeps, prior; n_particles = 1000, v = 10.0, n_simulation = nsim, checkpoint_display = 500)
	display(out_multeps)

	pop_singeps[irun] = hcat(out_singeps.population...)
	eps_singeps[irun] = hcat(out_singeps.state.ϵ_history...)
	rho_singeps[irun] = hcat(out_singeps.state.ρ_history...)
	u_singeps[irun] = hcat(out_singeps.state.u_history...)

	pop_multeps[irun] = hcat(out_multeps.population...)
	eps_multeps[irun] = hcat(out_multeps.state.ϵ_history...)
	rho_multeps[irun] = hcat(out_multeps.state.ρ_history...)
	u_multeps[irun] = hcat(out_multeps.state.u_history...)
end

# --- Rhos ---
P_r_multi = plot(title="Rho multeps - 10 stats", legend = :bottomleft)
plot!(P_r_multi, rho_multeps[1][1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
plot!(P_r_multi, rho_multeps[2][1,1:end], xaxis=:log, yaxis=:log, label="multi eps - other stats", 
		linecolor = :green, linewidth=2, thickness_scaling = 1)
for ix in 3:10
	plot!(P_r_multi, rho_multeps[1][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :green, linewidth=2, thickness_scaling = 1)
end
for irun in 2:length_series
	plot!(P_r_multi, rho_multeps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
	for ix in 2:10
		plot!(P_r_multi, rho_multeps[irun][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :green, linewidth=2, thickness_scaling = 1)
	end
end
display(P_r_multi)


P_r = plot(title="Rhos - 10 stats", legend = :bottomleft)
plot!(P_r, rho_multeps[1][2,1:end], xaxis=:log, yaxis=:log, label="multi eps - other stats", 
		linecolor = :green, linewidth=2, thickness_scaling = 1)
plot!(P_r, rho_singeps[1][2,1:end], xaxis=:log, yaxis=:log, label="single eps - other stats", 
		linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
for ix in 3:10
	plot!(P_r, rho_multeps[1][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :green, linewidth=2, thickness_scaling = 1)
	plot!(P_r, rho_singeps[1][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
end
for irun in 2:length_series
	for ix in 2:10
		plot!(P_r, rho_multeps[irun][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :green, linewidth=2, thickness_scaling = 1)
		plot!(P_r, rho_singeps[irun][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
	end
end
plot!(P_r, rho_multeps[1][1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
plot!(P_r, rho_singeps[1][1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean 1", 
		linecolor = :skyblue1, linewidth=2, thickness_scaling = 1)
for irun in 2:length_series
	plot!(P_r, rho_multeps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :coral, linewidth=2, thickness_scaling = 1)
	plot!(P_r, rho_singeps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :skyblue1, linewidth=2, thickness_scaling = 1)
end
display(P_r)

P_r = plot(title="Rhos - 10 stats", legend = :topright, ylim = [0,16], yticks = [1,10])
plot!(P_r, rho_multeps[1][2,1:end], xaxis=:log, yaxis=:log, label="multi eps - other stats", 
		linecolor = :green, linewidth=2, thickness_scaling = 1)
plot!(P_r, rho_singeps[1][2,1:end], xaxis=:log, yaxis=:log, label="single eps - other stats", 
		linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
for ix in 3:10
	plot!(P_r, rho_multeps[1][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :green, linewidth=2, thickness_scaling = 1)
	plot!(P_r, rho_singeps[1][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
end
for irun in 2:length_series
	for ix in 2:10
		plot!(P_r, rho_multeps[irun][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :green, linewidth=2, thickness_scaling = 1)
		plot!(P_r, rho_singeps[irun][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
	end
end
plot!(P_r, rho_multeps[1][1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
plot!(P_r, rho_singeps[1][1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean 1", 
		linecolor = :skyblue1, linewidth=2, thickness_scaling = 1)
for irun in 2:length_series
	plot!(P_r, rho_multeps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :coral, linewidth=2, thickness_scaling = 1)
	plot!(P_r, rho_singeps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :skyblue1, linewidth=2, thickness_scaling = 1)
end
display(P_r)

# ----------------------------
asa = 6
P_r = plot(title="Rhos - 10 stats", legend = :topright)
plot!(P_r, rho_multeps[asa][1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
plot!(P_r, rho_singeps[asa][1,1:end], xaxis=:log, yaxis=:log, label="single eps - mean 1", 
		linecolor = :skyblue1, linewidth=2, thickness_scaling = 1)
for ix in 2:10
	plot!(P_r, rho_multeps[asa][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :green, linewidth=2, thickness_scaling = 1)
	plot!(P_r, rho_singeps[asa][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :royalblue1, linewidth=2, thickness_scaling = 1)
end
display(P_r)
# ----------------------------


# --- Posteriors ---
for irun in 1:length_series
	P_post_1 = histogram(title = "Normal - mean 1 - 10 stats", bins=range(μ_min, μ_max, length=11), ylims=(0,600))
	histogram!(P_post_1, pop_singeps[irun][1,:], fillcolor = :skyblue1, fillalpha=0.5, 
			bins=range(μ_min, μ_max, length=11), label="single eps")
	histogram!(P_post_1, flatchain[1,:], fillcolor = :yellow, fillalpha=0.5, 
			bins=range(μ_min, μ_max, length=11), label="true post")
	P_post_2 = histogram(bins=range(μ_min, μ_max, length=21), ylims=(0,600))
	histogram!(P_post_2, pop_multeps[irun][1,:], fillcolor = :coral, fillalpha=0.5, 
			bins=range(μ_min, μ_max, length=11), label="multi eps")
	histogram!(P_post_2, flatchain[1,:], fillcolor = :yellow, fillalpha=0.5, 
			bins=range(μ_min, μ_max, length=11), label="true post")
	display(plot(P_post_1, P_post_2, layout = (1, 2)))
end


# --- u's, multeps ---
P_u = plot(title="u's - 10 stats", legend = :topright)
plot!(P_u, u_multeps[1][2,1:end], xaxis=:log, yaxis=:log, label="multi eps - other stats", 
		linecolor = :green, linewidth=2, thickness_scaling = 1)
for ix in 3:10
	plot!(P_u, u_multeps[1][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
		linecolor = :green, linewidth=2, thickness_scaling = 1)
end
for irun in 2:length_series
	for ix in 2:10
		plot!(P_u, u_multeps[irun][ix,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :green, linewidth=2, thickness_scaling = 1)
	end
end
plot!(P_u, u_multeps[1][1,1:end], xaxis=:log, yaxis=:log, label="multi eps - mean 1", 
		linecolor = :coral, linewidth=2, thickness_scaling = 1)
for irun in 2:length_series
	plot!(P_u, u_multeps[irun][1,1:end], xaxis=:log, yaxis=:log, label=false, 
			linecolor = :coral, linewidth=2, thickness_scaling = 1)
end
display(P_u)