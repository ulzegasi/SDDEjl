module SDDESolarDynamo_withPlanetsOnN_SN

"""
Include the following lines in the main script:
---
include("./SDDESolarDynamo_withPlanetsOnN_SN.jl")
using .SDDESolarDynamo_withPlanetsOnN_SN
---
"""

export sn, summary_statistics, hann_window,
        dft_at_frequency, amplitude_phase_at_period, summary_statistics_at_periods

using StochasticDelayDiffEq
using SpecialFunctions: erf
using StaticArrays
using Interpolations
using CSV
using DataFrames
import FFTW

# --- Nonlinear function
ftilde(x, Bmin, Bmax) = x/4 * (1 + erf(x^2-Bmin^2)) * (1 - erf(x^2-Bmax^2))

# --- Load torque data
# torque_data = CSV.read(joinpath(@__DIR__, "torque_for_SABC_for_SN.csv"), DataFrame, header=false)
tidal_data = CSV.read(joinpath(@__DIR__, "tides_for_SABC_for_SN.csv"), DataFrame, header=false)
# --- Extract years and values
years = tidal_data[:, 1]
const values = tidal_data[:, 2]
# --- Create time range
const years_range = range(years[1], stop=years[end], length=length(years))
# --- Interpolate
const modulation_function = Interpolations.scale(interpolate(values, BSpline(Cubic(Line(OnGrid())))), years_range)

# --- Modulation function 
function modulation(t, ϵ, ϕ)
	return (1 + ϵ * modulation_function(t+ϕ)) 
end

# ---------------------------------
# B-field WITH planetary torque
# ---------------------------------
# See here for the SDDE interface: https://docs.sciml.ai/DiffEqDocs/stable/tutorials/dde_example/

# --- Model : B field

function f(u,h,p,t)     # Drift function
    #  u = [B, dB/dt]
    # du = [dB/dt, d^2B/dt^2]
    τ, T, Nd, sigma, Bmax, eps, phi = p
    # --- with Jupiter
    hist = h(p, t - T, idxs = 1)    # B[1](t-T)
    du1 = u[2]
    du2 = -u[1]/τ^2 - 2*u[2]/τ - modulation(t, eps, phi) * Nd/τ^2*ftilde(hist, 1, Bmax)
    SA[du1, du2]
end

function g(u,h,p,t)     # Diffusion function
    τ, T, Nd, sigma, Bmax, eps, phi = p
    du1 = 0
    du2 = Bmax*sigma / (τ^(3/2))
    SA[du1, du2]
end

function bfield(θ, Tsim; kwargs...)

    τ, T, Nd, sigma, Bmax, eps, phi = θ

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


"""
Function to calculate DFT at arbitrary frequency. 
It matches FFTW convention. 
It matches fft(x) for all integer-bin frequencies f = k/N.

## Arguments

- `data`: The input time series data.
- `f`: Frequency at which to compute the Fourier component (f=1/T).
"""
function dft_at_frequency(x::AbstractVector{<:Real}, f::Float64)
    z = cis(-2π*f)             # e^{-iω}
    w = 1.0 + 0im
    s = 0.0 + 0im
    # By default, Julia checks array bounds every time you access an element (e.g. data[n]) 
    # to make sure you don’t go out of range.
    # That’s a safety feature that prevents segmentation faults and bugs.
    # However, these checks cost a tiny bit of time in tight inner loops.
    # @inbounds tells Julia: 
    # “I guarantee all my array indices are valid — skip the bounds checking.”
    @inbounds for n in 1:length(x)   # x[1] ≡ n=0
        s += x[n] * w
        w *= z
    end
    return s
end

"""
Calculate amplitude and phase at a specific period T

## Arguments
- data: input time series data.
- T: period at which to compute the amplitude and phase.
"""
function amplitude_at_period(data::AbstractVector{<:Real}, T::Float64)
    y = data .- mean(data)
    f = 1.0 / T
    X = dft_at_frequency(y, f)
    # returns amplitude and phase (radians)
    # return [(2/length(data))*abs(X), angle(X)]
    # returns amplitude
    return (2/length(data))*abs(X)
end

"""
Compute summary statistics for the input data based on 'dft_at_frequency' algorithm using a given window.

## Arguments

- `data`: The input time series data.
- `window`: Vector of windowing weights to be applied to the data. Defaults to `hann_window(length(data))`.
- `periods`: List of periods at which to compute amplitude and phase. Defaults to `[88.0, 104.0, 150.0, 208.0, 506.0]`.
"""
function summary_statistics_at_periods(data; 
                            window=hann_window(length(data)),
                            periods=float([11.0, 88.0, 104.0, 150.0, 208.0, 506.0]))
    windowed_data = window .* data
    # return reduce(vcat, [amplitude_at_period(windowed_data, T) for T in periods])
    return [amplitude_at_period(windowed_data, T) for T in periods]
end


end

