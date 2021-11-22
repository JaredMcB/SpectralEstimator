module WhiteningFiltersScalar

using LinearAlgebra
using DSP
using FFTW
using ToeplitzMatrices: Toeplitz
using StatsBase: var

at = include("AnalysisToolbox.jl")
mr = include("WFMR.jl")

function get_whf(X::Vector{T}; flags...) where T<: Number;
    h_m, h_w = get_whf(reshape(X,1,:); flags...)
    h_m[:], h_w[:]
end

function get_whf(X::Array{T,2};
    subrout = CKMS_vec,
    nfft = 0,
    par = 1500,             # Order of approximating Lauenat Poly
    M = par,               # number of output lags
    win = "Par",            # Type of smoother for autocov
    verb = false,
    flags...) where T<: Number;

    steps = size(X,2)
    verb && println("steps: ", steps)
    if subrout == CKMS_vec
        nfft = nfft == 0 ? nextfastfft(steps) : nfft
    else
        nfft = max(nfft,par,2*M)
    end

    ## We need M ≤ par < steps
    M > steps - 1 && println("M is bigger than length of time series")
    M   = min(M, steps - 1)
    par = max(M,par)
    println("par = ",par)
    par = min(par,steps - 1)

    R_pred_smoothed = at.my_smoothed_autocov(X; L = par, win)

    h_m, h_w = subrout(R_pred_smoothed; nfft, M, flags...)
end

function CKMS_vec(R;          # fitst three are common to both subroutines
    nfft,                 # resolution of z-spectrum on unit cirlce
    M,                    # number of out put lags.
    tol_ckms = 0,     # the rest are subroutine specific parameters
    N_ckms = 10^4,
    verb = false) where T<: Number;

    nu, par = size(R)[2:3]

    # Compute coefficients of spectral factorization of z-spect-pred
    S⁻ = mr.spectfact_matrix_CKMS(R; ϵ = tol_ckms, N_ckms);

    Err  = S⁻[2] ###
    S⁻ = S⁻[1]                            # the model filter ###

    h_m = S⁻[:,:,1:M]

    S⁻ = nfft >= par ? cat(dims = 3,S⁻,zeros(nu,nu,nfft - par)) :
                                (@view S⁻[:,:,1:nfft])

    fft!(S⁻,3)   # z-spectrum of model filter

    Sinv = zeros(ComplexF64,nu,nu,nfft)
    for i = 1 : nfft
        Sinv[:,:,i] = inv(@view S⁻[:,:,i])
    end

    h_w = ifft(Sinv,3)[:,:,1:M];

    verb ? (h_m, h_w, Err) : (h_m, h_w)
end


"""
    get_itr_whf is a function that computes whitening filters in an iteratatively

Examples:
~~~~~
x = pred[:]

h_m, h_w = whf.get_itr_whf(x; maxit = 3, par = 100, getter = whf.get_whf_B);

wx = filter_and_plot(h_w,x)
~~~~~
~~~~~
x = pred[:]

h_m, h_w = whf.get_itr_whf(x; maxit = 5, par = 1000, subrout = whf.whf_cholesky)

wx = filter_and_plot(h_w,x)
~~~~~
"""

function get_itr_whf(X::Vector{T}; flags...) where T<: Number;
    h_m, h_w = get_itr_whf(reshape(X,1,:); flags...)
    h_m[:], h_w[:]
end

function get_itr_whf(X::Array{T,2};
    getter = get_whf,
    maxit = 5,
    par = 1500,
    verb = false,
    flags_itr...) where T<: Number;

    d = size(X,1)

    h_w = zeros(ComplexF64,d,d,1); h_w[:,:,1] = I + zeros(d,d)
    h_m = copy(h_w)
    verb && println("Starting iterations")
    for i = 1 : maxit
        wx       = at.my_filt(h_w, X)
        Out      = getter(wx[:,par*(i-1)+1:end]; par, verb, flags_itr...)[1:2];
        h_m      = at.my_conv(h_m, Out[1])
        h_w      = at.my_conv(h_w, Out[2])
    end
    verb && println("Ending iterations")
    h_m, h_w
end

### Spectral estimate ##########################################################
################################################################################

function powerspec_mf(y::Vector{<:Number}; flags...)
    powerspec_mf(reshape(y,1,:); flags...)[:]
end

function powerspec_mf(y::Array{T,2}; Nex = 2^13,
    maxit = 3,
    par = 5000,
    smoothing_factor = 10,
    verb = false,
    win = "Par") where T <: Number;

    h_m = get_itr_whf(y; maxit, par,verb)[1]
    M = size(h_m,3) - 1
    l = M ÷ smoothing_factor
    lam = at._window(l; win,  two_sided = false)
    lam = [lam; zeros(M - l)]

    for i = 1: size(h_m,3)
        h_m[:,:,i] *= lam[i]
    end

    H = at.transferfun(h_m; Nex)

    S = zeros(eltype(H),size(H))
    for i = 1: size(H,3)
        S[:,:,i] = H[:,:,i]*H[:,:,i]'
    end
    S
end


function my_smoothed_autocov_mf(y::Array{T,2};
    L = L = min(size(pred,2)-1,2000),
    Nex = 2^13,
    maxit = 3) where T <: Number

    Sy_mf = powerspec_mf(y; par = L, Nex, maxit);

    Nh = Nex÷2

    A = ifft(Sy_mf,3)[:,:,1:Nh]

    A = Nh > L ? A[:,:,1:L] : cat(dims = 3, A, zeros(eltype(A),L - Nh))
end
end # module
