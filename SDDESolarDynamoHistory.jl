module SDDESolarDynamoHistory

"""
Include the following lines in the main script:
---
include("./SDDESolarDynamoHistory.jl")
using .SDDESolarDynamoHistory
---
"""

export sn, summary_statistics, summary_statistics_ii, hann_window

using StochasticDelayDiffEq
using SpecialFunctions: erf
using StaticArrays
using Interpolations
using CSV
using DataFrames
import FFTW


# ---------------------------------
# B-field
# ---------------------------------
# See here for the SDDE interface: https://docs.sciml.ai/DiffEqDocs/stable/tutorials/dde_example/

f(x, Bmin, Bmax) = x/4 * (1 + erf(x^2-Bmin^2)) * (1 - erf(x^2-Bmax^2))

function f(u,h,p,t)     # Drift function
    #  u = [B, dB/dt]
    # du = [dB/dt, d^2B/dt^2]
    τ, T, Nd, sigma, Bmax = p
    # --- with Jupiter
    hist = h(p, t - T, idxs = 1)    # B[1](t-T)
    du1 = u[2]
    du2 = -u[1]/τ^2 - 2*u[2]/τ - Nd/τ^2*f(hist, 1, Bmax)
    SA[du1, du2]
end

function g(u,h,p,t)     # Diffusion function
    τ, T, Nd, sigma, Bmax = p
    du1 = 0
    du2 = Bmax*sigma / (τ^(3/2))
    SA[du1, du2]
end

# Observed SN record: use cycle 1 as history
# First 18 elements of the yearly B-field time-series (1749 - 1766) 
# P.S.: B-field was calculated manually by inverting odd SN cycles
const history_values = [11.4919, 11.758, 8.78221, 8.80992, 7.0074, -4.06075, -3.50382,
                    -4.03961, -7.17582, -8.8632, -9.44598, -10.2133, -11.9016, -10.0069,
                    -8.57009, -7.69339, -5.85163, -0.783952]  

# Define time points for history values: t = -N_hist+1, ..., -1, 0
const time_points = collect(-length(history_values) + 1 : 0)
const time_range = range(time_points[1], stop=time_points[end], length=length(time_points))

# Interpolate
const itp = Interpolations.scale(interpolate(history_values, BSpline(Cubic(Line(OnGrid())))), time_range)
# Compute derivative at time 0
const dB0 = only(Interpolations.gradient(itp, 0)) 


function b_field(θ, T_sim; kwargs...)

    τ, T, Nd, sigma, Bmax = θ

    # --- define initial values
    u0 = SA[history_values[end], dB0]
    # define inital values [B(t), dB(t)/dt] if t < t0
    # h(p, t) = (Bmax, 0.0)
    # we can speed things up by providing a call by index, see example linked above:
    # h(p, t; idxs = nothing) = idxs == 1 ? Bmax : (Bmax, 0.0)

    function h(p, t; idxs=nothing)
        if t == 0  # Use interpolation to avoid discontinuities
            return idxs == 1 ? itp(0) : (itp(0), dB0)
        elseif t < 0  # This covers all t < 0 cases
            B_t = itp(t)  # Interpolated value 
            dB_t = only(Interpolations.gradient(itp, t))   # Approximate derivative
            return idxs == 1 ? B_t : (B_t, dB_t)
        else 
            error("Trying to evaluate history at time t > 0. History is defined only for t <= 0.")
        end
    end

    # use constant lags
    lags = (T, )
    tspan = (0.0, T_sim)

    prob = SDDEProblem(f, g, u0, h, tspan, θ; constant_lags = lags)
    solve(prob, EM(); dt=0.1, saveat = 1.0, kwargs...)
end


"""
```
sn(θ; T_sim = 254, kwargs...)
```

Stochastic simulations of the number of sunspots

### Arguments
- `θ`: parameter vector `[τ, T, Nd, sigma, Bmax]`
- `T_sim`: length of the output
- `kwargs...`: keyword arguments passed to `solve`. Mostly used to pass a seed for the random number generator (`seed = 123`).
"""
function sn(θ; T_sim = 254, kwargs...)

    sol = b_field(θ, T_sim; kwargs...)

    # square result
    y = map(abs2, sol[1, :])

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
