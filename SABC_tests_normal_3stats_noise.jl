import Pkg
Pkg.activate(@__DIR__)   # run scripts in ~/SABC
Pkg.instantiate()
                   
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
using KernelDensity
using LaTeXStrings
using Colors
using Measures

include("./AffineInvMCMC.jl")
using .AffineInvMCMC

"""
Plot SABC trajectories (ε, ρ, u) in a 3x1 grid (log plots).

Inputs:
  eps_hist : Vector{T} (single_eps) OR Matrix (n_stats x T) (multi_eps)
  rho_hist : Matrix (n_stats x T)
  u_hist   : Matrix (n_stats x T)

Optional:
  stat_labels : Vector{String} of length n_stats (e.g. ["μ","σ"])
  algorithm   : Symbol (:single_eps or :multi_eps) -> only affects ε plotting
  title_str   : global title
"""
function plot_sabc_trajectories(
    eps_hist,
    rho_hist::AbstractMatrix,
    u_hist::AbstractMatrix;
    stat_labels::Union{Nothing,Vector{String}} = nothing,
    algorithm::Symbol = :single_eps,
    title_str::String = "",
    size_px::Tuple{Int,Int} = (600, 900),
    margin_len = 6mm,
)

    ρ = Array(rho_hist)
    u = Array(u_hist)

    n_stats, T = size(ρ)
    it = 0:(T-1)

    # labels
    if stat_labels === nothing
        stat_labels = ["stat $(i)" for i in 1:n_stats]
    end

    # helper: plot multiple lines
    function plot_lines(Y; ylabel_str="", xlabel_str="", title_local="")
        p = plot(
            yscale = :log10,
            grid = true,
            ylabel = ylabel_str,
            xlabel = xlabel_str,
            title = title_local,
        )
        for k in 1:n_stats
            plot!(p, it, vec(Y[k, :]); label = stat_labels[k])
        end
        return p
    end

    # -------------------------
    # ε (log only)
    # -------------------------
    if algorithm == :single_eps
        ϵ = vec(eps_hist)
        p_eps = plot(
            it, ϵ;
            yscale = :log10,
            grid = true,
            ylabel = "ϵ",
            title = "ϵ (log)",
            label = false,
        )
    else
        p_eps = plot_lines(
            Array(eps_hist);
            ylabel_str = "ϵ",
            title_local = "ϵ (log)",
        )
    end

    # -------------------------
    # ρ (log)
    # -------------------------
    p_rho = plot_lines(
        ρ;
        ylabel_str = "Mean ρ",
        title_local = "Mean ρ (log)",
    )

    # -------------------------
    # u (log)
    # -------------------------
    p_u = plot_lines(
        u;
        ylabel_str = "Mean u",
        xlabel_str = "Iteration",
        title_local = "Mean u (log)",
    )

    # -------------------------
    # Combine (3 × 1)
    # -------------------------
    p_all = plot(
        p_eps,
        p_rho,
        p_u;
        layout = (3, 1),
        size = size_px,
        margin = margin_len,
        plot_title = title_str,
    )

    return p_all
end

# ------------------------------------------------
# --- Generate data ---
# ------------------------------------------------
Random.seed!(1822)
# Random.seed!()
true_mu = 10.0
true_sigma = 15.0
num_samples = 1000

y_obs = rand(Normal(true_mu, true_sigma), num_samples)  # generate data

# ------------------------------------------------
# --- Plot observed data and generating distribution ---
# ------------------------------------------------
# Grid for the true PDF
x = range(minimum(y_obs), maximum(y_obs), length=300)

# Histogram
histogram(
    y_obs;
    bins=20,
    normalize=:pdf,
    alpha=0.7,
    label="Data",
    linecolor=:black,
    linewidth=1,
    grid=true,
)

# Overlay true distribution
plot!(
    x,
    pdf.(Normal(true_mu, true_sigma), x);
    color=:red,
    linewidth=2,
    label="True distribution",
    grid=true,
)

xlabel!("Observed values")
ylabel!("Density")
title!("Observed data and true generating distribution")

display(current())

# ------------------------------------------------
# --- Generate MCMC samples for true posterior ---
# ------------------------------------------------
# Prior
μ_min = -10.0; μ_max = 20.0    # parameter 1: mean
σ_min = 0.0; σ_max = 25.0      # parameter 2: std
prior = product_distribution(Uniform(μ_min, μ_max), Uniform(σ_min, σ_max))

# True posterior
llhood = theta -> begin
	m, s  = theta;
	return -num_samples*log(s) - sum((y_obs.-m).^2)/(2*s^2)
end

lprior = theta -> begin
	m, s = theta;
	if (μ_min <= m <= μ_max) && (σ_min <= s <= σ_max)
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

numdims = 2
numwalkers = 20
thinning = 100
numsamples_perwalker = 5000
burnin = 10000;

rng = MersenneTwister(1822);
theta0 = Array{Float64}(undef, numdims, numwalkers);
theta0[1, :] = rand(rng, Uniform(μ_min, μ_max), numwalkers);  # mu
theta0[2, :] = rand(rng, Uniform(σ_min, σ_max), numwalkers);  # sigma

chain, llhoodvals = runMCMCsample(lprob, numwalkers, theta0, burnin, 1);
chain, llhoodvals = runMCMCsample(lprob, numwalkers, chain[:, :, end], numsamples_perwalker, thinning);
flatchain, flatllhoodvals = flattenMCMCarray(chain, llhoodvals)

mu_truepost = flatchain[1, :]
sigma_truepost = flatchain[2, :]

# ------------------------------------------------
# --- Plot true posterior ---
# ------------------------------------------------
mu_lims    = (8.0, 12.0)
sigma_lims = (13.0, 17.0)
n_levels   = 10

mu_grid  = range(mu_lims[1],    mu_lims[2],    length=250)
sig_grid = range(sigma_lims[1], sigma_lims[2], length=250)

kde_grid = kde((mu_truepost, sigma_truepost), (mu_grid, sig_grid))   # <-- fast

wb = cgrad([RGB(1,1,1), RGB(0,0,0)])  # white -> black

p = contourf(
    kde_grid.x, kde_grid.y, kde_grid.density;
    levels=n_levels,
    c=wb,
    colorbar=true,
    label=false,
    background_color=:white,
    background_color_inside=:white,
    foreground_color=:black,
    grid=:on,
    gridalpha=0.45,
    gridlinewidth=0.7,
    gridcolor=:black,
    fillalpha=0.65,
)

contour!(
    kde_grid.x, kde_grid.y, kde_grid.density;
    levels=n_levels,
    linecolor=:black,
    linewidth=0.6,
    alpha=0.6,
    label=false,
)

scatter!(
    [true_mu], [true_sigma];
    color=:yellow,
    marker=:x,
    markersize=5,
    linewidth=8,
    label=false,
)

xlims!(mu_lims)
ylims!(sigma_lims)
xlabel!("μ")
ylabel!("σ")
title!("True posterior")

display(p)

# ------------------------------------------------
# --- Plot true marginals ---
# ------------------------------------------------
μ_edges = range(minimum(mu_truepost), maximum(mu_truepost), length=21)
σ_edges = range(minimum(sigma_truepost), maximum(sigma_truepost), length=21)

p1 = histogram(
    mu_truepost;
    bins = μ_edges,
    normalize = :pdf,
    color = :gray,
    linecolor = :black,
    alpha = 0.7,
    xlabel = "μ",
    ylabel = "Density",
    title = "Marginal posterior of μ",
    label = false,
    xlims = mu_lims,
    ylims = (0, 1.5),
    legend = false,              
)

p2 = histogram(
    sigma_truepost;
    bins = σ_edges,
    normalize = :pdf,
    color = :gray,
    linecolor = :black,
    alpha = 0.7,
    xlabel = "σ",
    ylabel = "",
    title = "Marginal posterior of σ",
    label = false,
    xlims = sigma_lims,
    ylims = (0, 1.5),
    yticks = false,
    legend = false,              
)

vline!(p1, [true_mu]; color=:yellow, linewidth=3, label=false)
vline!(p2, [true_sigma]; color=:yellow, linewidth=3, label=false)

p_marginals = plot(
    p1, p2;
    layout = (1,2),
    size = (700,300),
    legend = false,
	margin = 4mm,          # overall outer margin
)

display(p_marginals)

# ------------------------------------------------
# --- Reset seed for inference ---
# ------------------------------------------------
Random.seed!(1822)
# Random.seed!()

# ------------------------------------------------
# --- Inference ---
# --- Two stats: mean and std ---
# ------------------------------------------------
function sum_stats(data)
	stat1 = mean(data)
	stat2 = std(data) 
    stat3 = randn()
	return [stat1, stat2, stat3]
end

function model(θ)
	y = rand(Normal(θ[1],θ[2]), num_samples)
	return sum_stats(y)
end

function f_dist(θ)
	# Data-generating model
	ss = model(θ)
	# Distance
	rho = [euclidean(ss[ix], ss_obs[ix]) for ix in 1:size(ss,1)]
	return rho
end

ss_obs = sum_stats(y_obs)


np = 1000       # number of particles
ns = 1_000_000  # number of particle updates

out_singeps = sabc(f_dist, prior; 
				n_particles = np, 
				n_simulation = ns,
				v = 1.0, 
				algorithm = :single_eps,
				proposal = DifferentialEvolution(n_para=length(prior)))

out_singeps_2 = update_population!(out_singeps, f_dist, prior; v = 1.0, n_simulation = ns)

pop_singeps = hcat(out_singeps_2.population...)
eps_singeps = hcat(out_singeps_2.state.ϵ_history...)
rho_singeps = hcat(out_singeps_2.state.ρ_history...)
u_singeps = hcat(out_singeps_2.state.u_history...)

out_multeps = sabc(f_dist, prior; 
				n_particles = np, 
				n_simulation = ns,
				v = 1.0, 
				algorithm = :multi_eps,
				proposal = DifferentialEvolution(n_para=length(prior)))

out_multeps_2 = update_population!(out_multeps, f_dist, prior; v = 1.0, n_simulation = ns)

pop_multeps = hcat(out_multeps_2.population...)
eps_multeps = hcat(out_multeps_2.state.ϵ_history...)
rho_multeps = hcat(out_multeps_2.state.ρ_history...)
u_multeps = hcat(out_multeps_2.state.u_history...)

# ------------------------------------------------
# --- SABC posterior plot: single_eps ---
# ------------------------------------------------
mu_sabc_singeps    = pop_singeps[1, :]
sigma_sabc_singeps = pop_singeps[2, :]

kde_sabc_singeps = kde((mu_sabc_singeps, sigma_sabc_singeps), (mu_grid, sig_grid))

p_sabc_singeps = contourf(
    kde_sabc_singeps.x, kde_sabc_singeps.y, kde_sabc_singeps.density;
    levels = n_levels,
    c = wb,
    colorbar = false,
    label = false,
    background_color = :white,
    background_color_inside = :white,
    foreground_color = :black,
    grid = :on,
    gridalpha = 0.45,
    gridlinewidth = 0.7,
    gridcolor = :black,
    fillalpha = 0.65,
)

contour!(
    p_sabc_singeps,
    kde_sabc_singeps.x, kde_sabc_singeps.y, kde_sabc_singeps.density;
    levels = n_levels,
    linecolor = :black,
    linewidth = 0.6,
    alpha = 0.6,
    label = false,
)

scatter!(
    p_sabc_singeps,
    [true_mu], [true_sigma];
    color = :yellow,
    marker = :x,
    markersize = 5,
    linewidth = 8,
    label = false,
)

xlims!(p_sabc_singeps, mu_lims)
ylims!(p_sabc_singeps, sigma_lims)
xlabel!(p_sabc_singeps, "μ")
ylabel!(p_sabc_singeps, "σ")
title!(p_sabc_singeps, "SABC, single_eps, 3 stats")

plot!(p; colorbar = false)  # ensure equal panel widths

p_both_singeps = plot(
    p, p_sabc_singeps;
    layout = (1, 2),
    size = (900, 450),
    margin = 6mm,
)

display(p_both_singeps)

# ------------------------------------------------
# --- SABC posterior plot: multi_eps ---
# ------------------------------------------------
mu_sabc_multeps    = pop_multeps[1, :]
sigma_sabc_multeps = pop_multeps[2, :]

kde_sabc_multeps = kde((mu_sabc_multeps, sigma_sabc_multeps), (mu_grid, sig_grid))

p_sabc_multeps = contourf(
    kde_sabc_multeps.x, kde_sabc_multeps.y, kde_sabc_multeps.density;
    levels = n_levels,
    c = wb,
    colorbar = false,
    label = false,
    background_color = :white,
    background_color_inside = :white,
    foreground_color = :black,
    grid = :on,
    gridalpha = 0.45,
    gridlinewidth = 0.7,
    gridcolor = :black,
    fillalpha = 0.65,
)

contour!(
    p_sabc_multeps,
    kde_sabc_multeps.x, kde_sabc_multeps.y, kde_sabc_multeps.density;
    levels = n_levels,
    linecolor = :black,
    linewidth = 0.6,
    alpha = 0.6,
    label = false,
)

scatter!(
    p_sabc_multeps,
    [true_mu], [true_sigma];
    color = :yellow,
    marker = :x,
    markersize = 5,
    linewidth = 8,
    label = false,
)

xlims!(p_sabc_multeps, mu_lims)
ylims!(p_sabc_multeps, sigma_lims)
xlabel!(p_sabc_multeps, "μ")
ylabel!(p_sabc_multeps, "σ")
title!(p_sabc_multeps, "SABC, multi_eps, 3 stats")
plot!(p; colorbar = false)  # ensure equal panel widths

p_both_multeps = plot(
    p, p_sabc_multeps;
    layout = (1, 2),
    size = (900, 450),
    margin = 6mm,
)

display(p_both_multeps)


# ------------------------------------------------
# --- Plot trajectories ---
# ------------------------------------------------

display(plot_sabc_trajectories(
    eps_singeps, rho_singeps, u_singeps;
    algorithm = :single_eps,
    stat_labels = ["μ", "σ", "noise"],
    title_str = "Algorithm: single_eps | Summary stats: μ, σ, noise",
	size_px = (700, 900)
))


display(plot_sabc_trajectories(
    eps_multeps, rho_multeps, u_multeps;
    algorithm = :multi_eps,
    stat_labels = ["μ", "σ", "noise"],
    title_str = "Algorithm: multi_eps | Summary stats: μ, σ, noise",
	size_px = (700, 900)
))


# ------------------------------------------------
# --- 2x2 grid: TRUE marginals (top) vs SABC (bottom)
# ------------------------------------------------
# --- bottom row: SABC marginals ---
col_mu = palette(:auto)[1]
col_σ  = palette(:auto)[2]

# --- single_eps ---
p1_sabc_singeps = histogram(
    mu_sabc_singeps;
    bins = μ_edges,
    normalize = :pdf,
    color = col_mu,
    alpha = 0.35,          # transparent fill
    linecolor = col_mu,
    linewidth = 1.5,
    xlabel = "μ",
    ylabel = "Density",
    title = "SABC marginal of μ - single-eps",
    label = false,
    xlims = mu_lims,
    ylims = (0, 1.5),
    legend = false,
)

vline!(p1_sabc_singeps, [true_mu]; color=:yellow, linewidth=3, label=false)

p2_sabc_singeps = histogram(
    sigma_sabc_singeps;
    bins = σ_edges,
    normalize = :pdf,
    color = col_σ,
    alpha = 0.35,          # transparent fill
    linecolor = col_σ,
    linewidth = 1.5,
    xlabel = "σ",
    ylabel = "",
    title = "SABC marginal of σ - single-eps",
    label = false,
    xlims = sigma_lims,
    ylims = (0, 1.5),
    yticks = false,
    legend = false,
)

vline!(p2_sabc_singeps, [true_sigma]; color=:yellow, linewidth=3, label=false)

# --- combine (top: p1 p2 already exist) ---
p_grid_singeps = plot(
    p1, p2,
    p1_sabc_singeps, p2_sabc_singeps;
    layout = (2, 2),
    size = (900, 600),
    margin = 6mm,
)

display(p_grid_singeps)

# --- multi_eps ---

p1_sabc_multeps = histogram(
    mu_sabc_multeps;
    bins = μ_edges,
    normalize = :pdf,
    color = col_mu,
    alpha = 0.35,          # transparent fill
    linecolor = col_mu,
    linewidth = 1.5,
    xlabel = "μ",
    ylabel = "Density",
    title = "SABC marginal of μ - multi-eps",
    label = false,
    xlims = mu_lims,
    ylims = (0, 1.5),
    legend = false,
)

vline!(p1_sabc_multeps, [true_mu]; color=:yellow, linewidth=3, label=false)

p2_sabc_multeps = histogram(
    sigma_sabc_multeps;
    bins = σ_edges,
    normalize = :pdf,
    color = col_σ,
    alpha = 0.35,          # transparent fill
    linecolor = col_σ,
    linewidth = 1.5,
    xlabel = "σ",
    ylabel = "",
    title = "SABC marginal of σ - multi-eps",
    label = false,
    xlims = sigma_lims,
    ylims = (0, 1.5),
    yticks = false,
    legend = false,
)

vline!(p2_sabc_multeps, [true_sigma]; color=:yellow, linewidth=3, label=false)

# --- combine (top: p1 p2 already exist) ---
p_grid_multeps = plot(
    p1, p2,
    p1_sabc_multeps, p2_sabc_multeps;
    layout = (2, 2),
    size = (900, 600),
    margin = 6mm,
)

display(p_grid_multeps)
