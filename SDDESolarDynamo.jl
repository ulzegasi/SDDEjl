module SDDESolarDynamo

"""
Include the following lines in the main script:
---
include("./SDDESolarDynamo.jl")
using .SDDESolarDynamo
---
"""

export sn, summary_statistics, summary_statistics_ii, hann_window

using StochasticDelayDiffEq
using SpecialFunctions: erf
using StaticArrays
import FFTW

# --- Nonlinear function

ftilde(x, Bmin, Bmax) = x/4 * (1 + erf(x^2-Bmin^2)) * (1 - erf(x^2-Bmax^2))


# ---------------------------------
# B-field WITHOUT Jupiter
# ---------------------------------
# See here for the SDDE interface: https://docs.sciml.ai/DiffEqDocs/stable/tutorials/dde_example/

# --- Model : B field

function f(u,h,p,t)     # Drift function
    #  u = [B, dB/dt]
    # du = [dB/dt, d^2B/dt^2]
    τ, T, Nd, sigma, Bmax = p
    # --- with Jupiter
    hist = h(p, t - T, idxs = 1)    # B[1](t-T)
    du1 = u[2]
    du2 = -u[1]/τ^2 - 2*u[2]/τ - Nd/τ^2*ftilde(hist, 1, Bmax)
    SA[du1, du2]
end

function g(u,h,p,t)     # Diffusion function
    τ, T, Nd, sigma, Bmax = p
    du1 = 0
    du2 = Bmax*sigma / (τ^(3/2))
    SA[du1, du2]
end


function bfield(θ, Tsim; kwargs...)

    τ, T, Nd, sigma, Bmax = θ

    # --- define initial values
    u0 = SA[Bmax, 0.0]
    # define inital values [B(t), dB(t)/dt] if t < t0
    # h(p, t) = (Bmax, 0.0)
    # we can speed things up by providing a call by index, see example linked above:
    h(p, t; idxs = nothing) = idxs == 1 ? Bmax : (Bmax, 0.0)

    # use constant lags
    lags = (T, )
    tspan = (0.0, Tsim)

    prob = SDDEProblem(f, g, u0, h, tspan, θ; constant_lags = lags)
    solve(prob, EM(); dt=0.1, saveat=1.0, kwargs...)
end


"""
```
sn(θ; Tobs = 929, Twarmup = 200, kwargs...)
```

Stochastic simulations of the number of sunspots

### Arguments
- `θ`: parameter vector `[τ, T, Nd, sigma, Bmax]`
- `Tobs`: length of the output
- `Twarmup`: length of the warm up period
- `kwargs...`: keyword arguments pased to `solve`. Mostly used ot pass a seed for the random number generator (`seed = 123`).
"""
function sn(θ; Twarmup = 200, Tobs = 929, kwargs...)

    Tsim = Twarmup + Tobs  # Total simulation steps

    sol = bfield(θ, Tsim; kwargs...)

    # square result and get rid of warm up points
    y = map(abs2, sol[1, (Twarmup + 2):end])

    return y
end

# -------------
# Summary statistics
# -------------


"""
`hann_window(Tmax)`

Generate a Hann window of size `Tmax`.
"""
hann_window(Tmax) = [0.5*(1 - cos(2.0*π*(t-1)/(Tmax-1))) for t in 1:Tmax]


"""
`summary_statistics(data, window; fourier_range=1:6:120)`

Compute summary statistics for the input data based on the Fourier transform using a given window.

## Arguments

- `data`: The input time series data.
- `window`: Vector of windowing weights to be applied to the data. Defaults to `hann_window(length(data))`.
- `fourier_range`: Indices of the Fourier-transformed components to include in as summary statistics. Defaults to `1:6:120`.
"""
function summary_statistics(data, window=hann_window(length(data));
                            fourier_range=1:6:120)
    fs = FFTW.ifft(window .* data)
    ss = abs.(fs[fourier_range])
    return ss
end

# We consider the first 120 Fourier components, Re and Im parts (-> 240 components).
# Indices 0-119 are Re parts, indices 120-239 are the Im parts

# Observed SN (yearly resolution), Tobs = 271:
# [1, 129, 68, 197, 7, 75, 76, 206, 79, 210, 211, 56, 57, 127]

# C-14 data (Usoskin et al., A&A, 2021), Tobs = 925:
# [65, 1, 2, 142, 145, 113, 28, 29]

function summary_statistics_ii(data, window=hann_window(length(data));
                                fourier_range=1:6:120)
    fs = FFTW.ifft(window .* data)[1:120]
    ss = [real.(fs); imag.(fs)][fourier_range]
    return ss
end


end
