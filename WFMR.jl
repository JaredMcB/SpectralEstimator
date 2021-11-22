module WFMR

using FFTW
using LinearAlgebra
using DSP: conv, nextfastfft
using StatsBase
using SparseArrays

at = include("AnalysisToolbox.jl")

"""
    `get_wf` provides the causal wiener filter that best approximates `signal`
in terms of `Psi(signal[:,i-1])`. it has built in the one step prediction. First,
it generates the predictor series. and then employs the `vector_wiener_filter_fft`
function to obtain the Wiener filter. Inputs are the signal, a 2d array, d-by-N
where d is the dimension of the state space an N is the number of point. There
are two key word arguments, `M_out` specifies the number of coefficents desired
from the Weiner filter. And `rl` which if true just take the real part of the
Wiener filter as the output.
"""

function get_wf(
    signal, # Vector valued process
    Psi; # column vector valued function
    M_out = 20,
    n = 3, p = 1500, par = 1500,
    ty = "bin",
    xspec_est = "old",
    nfft = 0,
    rl = true,
    Preds = false,
    N_ckms = 10^5,
    PI = false,
    rtol = 1e-6,
    verb = false)
    # We would like a presample since we want the
    # times series to be offset by one.

    d, steps = size(signal)
    nu = size(Psi(zeros(d,1)),1)

    sig = @view signal[:,2:steps]   # sig is now one a head of signal
    steps -= 1                      # this makes steps the length of sig

    pred = zeros(ComplexF64, nu, steps)
    for n = 1:steps
        pred[:,n] = Psi(@view signal[:,n])
    end # pred is now even with signal and therefore one step
        # behind sig. I.e. pred[:,n] = Psi(sig[:,n-1])
        # which is what we want so as to ensure the reduced
        # model can run explicitly.

    if verb
        println("==================== New Run $steps =====================")
    end

    h_wf = vector_wiener_filter_fft(sig, pred; M_out,
            n, p, par, nfft, ty, xspec_est, PI, N_ckms, rtol,verb)

    h_wf = rl ? real(h_wf) : h_wf
    Preds ? [h_wf, pred] : h_wf
end

function get_wf(
    sig::Array{<:Number,2}, # Vector valued process
    pred::Array{<:Number,2}; # column vector valued function
    M_out = 20,
    n = 3, p = 1500, par = 1500,
    ty = "bin",
    xspec_est = "old",
    nfft = 0,
    rl = true,
    Preds = false,
    N_ckms = 10^5,
    PI = false,
    rtol = 1e-6,
    verb = false)

    h_wf = vector_wiener_filter_fft(sig, pred; M_out,
            n, p, par, nfft, ty, xspec_est, PI, N_ckms, rtol,verb)

    h_wf = rl ? real(h_wf) : h_wf
    Preds ? [h_wf, pred] : h_wf
end

function get_pred(signal, Psi)
    d, steps = size(signal)
    nu = size(Psi(zeros(d,1)),1)

    pred = zeros(ComplexF64, nu, steps)
    for n = 1:steps
        pred[:,n] = Psi(@view signal[:,n])
    end
    pred
end

"""
    spectfact_matrix_CKMS take in m+1 coefficents of a (d x d)-matrix-values Laurent
polynomial of the form
        M(z) = P[:,:,1] + sum_{i=1}^{m} P[:,:,i+1]*z^{-i} + P[:,:,i+1]'z^i
and produce the first m+1 coefficients of the (dxd)-matrix valued polynomial
(in z^{-1})
        L(z) = l[:,:,1] + sum_{i=1}^{m} l[:,:,i+1]*z^{-i}
which satisfies
        M(z) = L(z)*L(z^{-1}}')'
    For this function the input is P and the output is l.
"""

function spectfact_matrix_CKMS(P; ϵ = 1e-10,
    update = 10,
    N_ckms = 10^4,
    rtol = 1e-6,
    verb = false)

    d = size(P)[1];
    m = size(P)[3] - 1

    NN = reverse((@view P[:,:,2:m+1]),dims = 3)
    Re = Rr = p0 = @view P[:,:,1]

    F = sparse([[spzeros(d,d*(m-1)); sparse(I,d*(m-1),d*(m-1))] spzeros(d*m,d)])
    h = sparse([spzeros(d,d*(m-1)) sparse(I,d,d)])

    K = zeros(ComplexF64,d*m,d)
    for i = 0 : m-1
        K[d*i + 1: d*(i+1),:] = NN[:,:,i+1]
    end
    FL = K
    i = 0
    errK = errR = 1
    Err = zeros(2,N_ckms)
    
    while (errK > ϵ || errR > ϵ) && i <= N_ckms - 1
        hL = h*FL
        FL = F*FL

        # Rr_pinv = pinv(Rr, rtol = rtol)
        # Re_pinv = pinv(Re, rtol = rtol)

        # Stopping criteria stuff
        i += 1

        FL_RrhLt = FL / Rr * hL'
        hL_RrhLt = hL / Rr * hL'
        Err[1,i] = errK = norm(FL_RrhLt, Inf)
        Err[2,i] = errR = norm(hL_RrhLt, Inf)

        FL -= K / Re * hL
        K  -= FL_RrhLt
        Rr -= hL' / Re * hL
        Re -= hL_RrhLt
    end

    if verb
        r = .1
        ErrK, ErrR = maximum(Err[:, floor(Int,i*(1 - r)):i], dims = 2)
        println("Number of CKMS iterations: $i")
        println("max diff in K and R of past $(100*r)%: ", ErrK,",   ", ErrR)
    end
    
    K /= Re

    sqrt_re = sqrt(Re)

    l = zeros(ComplexF64,d,d,m+1)
    l[:,:,1] = sqrt_re;
    for i = m-1:-1:0
        l[:,:,m-i+1] = K[d*i + 1: d*(i+1),:]*sqrt_re
    end


    l, Err ###
end


"""
    vector_wiener_filter_fft

"""
function vector_wiener_filter_fft(
    sig,
    pred::Array{T,2} where T <: Number;
    M_out = 20,
    par::Int64 = 1500,
    nfft = 0,
    win = "Par",
    n = 3,
    p = 1500,
    ty = "bin",
    xspec_est = "old",
    N_ckms = 10^5,
    PI = true,
    rtol = 1e-6,
    verb = false
    )

    d, stepsy = size(sig)
    nu, stepsx = size(pred)

    stepsx == stepsy || print("X and Y are not the same length. Taking min.")
    steps = minimum([stepsx stepsy])
    nfft = nfft == 0 ? nextfastfft(steps) : nfft
    nffth = nfft ÷ 2
    L = min(par,steps-1)

    R_pred_smoothed = @timed at.my_smoothed_autocov(pred; L, steps, nu, win)
    if verb
        println("Time taken for autocov: ", R_pred_smoothed.time)
        println("Bytes Allocated: ", R_pred_smoothed.bytes)
    end

    # Compute coefficients of spectral factorization of z-spect-pred
    S_pred⁻ = @timed spectfact_matrix_CKMS(R_pred_smoothed.value; N_ckms)
    if verb
        println("Time taken for spectfact: ",S_pred⁻.time)
        println("Bytes Allocated: ",S_pred⁻.bytes)
    end
    
    S_pred⁻ = nfft >= L+1 ? cat(dims = 3,S_pred⁻.value[1],zeros(nu,nu,nfft - L)) :
                            (@view S_pred⁻.value[1][:,:,1:nfft])
    
    fft!(S_pred⁻,3)                                 # the final S_pred⁻

    S_pred⁺ = zeros(ComplexF64,nu,nu,nfft)
    for i = 1 : nfft
        S_pred⁺[:,:,i] = (@view S_pred⁻[:,:,i])'
    end                                             # the final S_pred⁺

    # Compute z-cross-spectrum of sigpred
    S = @timed xspec_est == "SP" ? at.z_crossspect_fft(sig, pred;
                        nfft, n, p, ty) : at.z_crossspect_fft_old(sig, pred; L, Nex = nfft);
    if verb
        println("Time taken for crossspect: ",S.time)
        println("Bytes Allocated: ",S.bytes)
    end

    S = S.value
    for i = 1 : nfft                # point-wise divide in time domain S_{YX}
        S[:,:,i] /= @view S_pred⁺[:,:,i]        # by S_x^+
    end
    ifft!(S,3)                      # Fourier space
    S[:,:, nffth + 1 : nfft] = zeros(d,nu,nfft - nffth) # Causal part
    fft!(S,3)                       # Back to time domain,{S_{yx}/S_x^+}_+
    for i = 1: nfft                 # Obtain transfer function H by dividing
        S[:,:,i] /= @view S_pred⁻[:,:,i]  # {S_{yx}/S_x^+}_+ by S_x^-
    end
    ifft!(S, 3)                     # impulse responce of Weiner filter,
                                    # fourier space
                                    # Extrct tranfer function coeffifcients

    # Truncate
    M_out > nfft && println("M_out > nfft, taking min")
    M = min(M_out, nfft)

    h_num_fft = S[:,:,1:M]
end

function redmodrun(
   sig,              # Here only the first M_out = size(h_wf,3) are needed
   h_wf,             # The Wiener filter
   Psi;              # The basis functions of the reduced model
   sig_m = zeros(size(sig[:,1])),            # The mean of the signal proces
   pred_m = zeros(size(Psi(sig[:,1]))),      # the mean of the predictor process
   steps,            # How many steps you want to run the RM
   discard = steps,          # How many steps we discard
   noise = false,    # true includes the noise term
   noise_dist        # The distribution of the noise term
   )

   d, nu, M_out = size(h_wf)
   steps_tot = steps + discard

   ## Run reduced model with no noise
   sig_rm = complex(zeros(d,steps_tot))
   sig_rm[:,1:M_out] = sig[:,1:M_out]

   C = sig_m + sum(h_wf[:,:,k] for k = 1:M_out)*pred_m

   # load presamples
   PSI_past = complex(zeros(nu,steps_tot))
   for i=1:M_out
       PSI_past[:,i] = Psi(sig[:,i])
   end
   # Move forward without original data
   for i = M_out+1:steps_tot
       sig_rm[:,i] = sum(h_wf[:,:,k]*PSI_past[:,i-k] for k = 1:M_out) +
                     C + (noise ? rand(noise_dist) : zeros(d))
       isnan(sig_rm[1,i]) && break
       PSI_past[:,i] = Psi(sig_rm[:,i])
   end
   sig_rm[:,discard+1:end]
end

###########################
function get_Psi_2017(h)
    # Build PSI
    function InvBurgRK4_1step(x)
       lx = length(x)
       function F(x)
           𝑥 = [conj(@view x[lx:-1:1]) ;0; x]
           conv(𝑥,𝑥)[2*lx+2:3*lx+1]
       end
       k1 = F(x)
       k2 = F(x .+ h*k1/2)
       k3 = F(x .+ h*k2/2)
       k4 = F(x .+ h*k3)
       A = @. x + h/6*(k1 + 2k2 + 2k3 + k4)
    end

    function Inertialman_part(x)
       lx = length(x)
       𝑥(j) = ( j <= lx ? x[j] : im*sum(x[l]*x[j-l] for l = j-lx:lx) )

       L = complex(zeros(lx^2))
       for j = 1:lx
          for k = 1:lx
             L[ (j-1)*lx+k] = 𝑥(j+lx)*𝑥(j+lx-k)
          end
       end
       L
    end

    Psi(x) = [x; InvBurgRK4_1step(x); Inertialman_part(x)]
    return Psi
end

end #Module
