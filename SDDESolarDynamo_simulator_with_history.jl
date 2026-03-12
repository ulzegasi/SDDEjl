using Pkg
Pkg.activate("SABC") 
using Plots
using CSV 
using DataFrames
using FFTW
using StochasticDelayDiffEq
using SpecialFunctions
using StaticArrays
using DifferentialEquations
using Interpolations
using Statistics


# ---------------------------------
# B-field WITHOUT Jupiter
# ---------------------------------
# See here for the SDDE interface: https://docs.sciml.ai/DiffEqDocs/stable/tutorials/dde_example/

# --- Model : B field

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


function b_field(θ, Tsim, history_values; kwargs...)

    τ, T, Nd, sigma, Bmax = θ

    N_hist = length(history_values)

    # Define time points for history values: t = -N_hist+1, ..., -1, 0
    time_points = collect(-N_hist+1:0)
    time_range = range(time_points[1], stop=time_points[end], length=length(time_points))

    # Interpolate
    itp = Interpolations.scale(interpolate(history_values, BSpline(Cubic(Line(OnGrid())))), time_range)

    # Compute derivative at time 0
    dB0 = only(Interpolations.gradient(itp, 0)) 

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
function sn(θ, T_obs, history_values; kwargs...)

    sol = b_field(θ, T_obs, history_values; kwargs...)

    # square result and get rid of warm up points
    y = map(abs2, sol[1, :])

    return y
end

b_data = reshape(Matrix(CSV.read("/Users/ulzg/SABC/bfield_from_sunspots_yearly_resolution.csv", DataFrame, header=false)),:)

# Correct last point:
b_data[end] = 1.616017346845035

topSamples = Matrix(CSV.read("/Users/ulzg/SABC/topSamples_for_cycle_25.csv", DataFrame, header=false))

out = sn(topSamples[:,99], 470, b_data)
plot(out)