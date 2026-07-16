"""CMB lensing convergence κ from halos, following the Websky recipe (Stein et al. 2020,
§3.2.3; Fortran peak-patch `pks2map` profile code `kap`).

Under the Born approximation, κ(n̂) = ∫dχ W_κ(χ) δ(χn̂) with the lensing kernel

    W_κ(χ) = (3/2) Ω_m (H₀/c)² (1+z) χ (1 - χ/χ*),

so each halo contributes W_κ(χ_h)·Σ(θχ_h)/ρ̄_m, where Σ is the comoving projected mass
density of the halo. Websky assumes all halos follow an NFW profile with concentration
c ≡ r200m/r_s = 7, independent of mass and redshift, extended by ρ(r) =
ρ_NFW(r200)·(r/r200)⁻² for r200 < r < 2·r200 (approximate nonlinear infall) and
truncated beyond. r200m encloses 200× the MEAN matter density, matching the M200m
masses of peak-patch/Websky catalogs.

Conventions: halo masses are M200m in Msun (NOT Msun/h — convert peak-patch catalogs),
distances internally in physical Mpc via Cosmology.jl. The halo-only map painted here
has ⟨κ⟩ > 0; Websky's released kap.fits additionally contains the mass-conserving field
component (matter outside halos, from the LPT displacement field) and a z>4.5 Gaussian
tail — those require simulation-side data and are out of scope for catalog painting.

Note the total projected halo mass is M200m·(1 + c²/(1+c)²/f_nfw(c)) ≈ 1.64·M200m for
c=7: the r⁻² extension carries the (physical) mass between r200 and 2·r200.

Resolution caveat: painting samples the profile at pixel centers, and the NFW Σ has a
logarithmic cusp. For generic (non-grid-aligned) halo positions the painted mass
integral is accurate to ~2% at 0.25′ pixels and converges to 0.1% by 0.05′; a halo
sitting exactly on a pixel center is pathological (~1.5× at 0.25′), so avoid
grid-aligned test positions and prefer resolutions that resolve r_s/χ for the halos
that matter.
"""

struct NFWKappaProfile{T,C,I1,I2} <: AbstractGNFW{T}
    cosmo::C
    cnfw::T     # concentration r200m/r_s (Websky: 7, mass/z-independent)
    xmax::T     # truncation radius in units of r200m (Websky: 2)
    χstar::T    # comoving distance to the source plane [Mpc]
    ρm0::T      # comoving mean matter density [Msun/Mpc³]
    z2chi::I1   # z → comoving distance [Mpc]
    gx::I2      # log(x) → dimensionless projected profile g(x), x = R⊥/r_s
end

"""
    NFWKappaProfile(; Omega_c=0.2589, Omega_b=0.0486, h=0.6774, cnfw=7, xmax=2,
                    z_star=1089, z_max=6.0)

Websky-style halo lensing convergence profile: truncated NFW (c=`cnfw` w.r.t. r200m,
r⁻² extension to `xmax`·r200m) × CMB lensing kernel with source at `z_star`.
Evaluate as `model(θ, M200m_Msun, z)` (θ in radians); paint with the standard
`paint!(map, workspace, model, masses, redshifts, αs, δs)`.
"""
function NFWKappaProfile(; Omega_c::T=0.2589, Omega_b::T=0.0486, h::T=0.6774,
                         cnfw::T=7.0, xmax::T=2.0, z_star=1089.0, z_max=6.0) where {T<:Real}
    OmegaM = Omega_b + Omega_c
    cosmo = get_cosmology(T, h=h, OmegaM=OmegaM)
    χstar = T(ustrip(u"Mpc", Cosmology.comoving_radial_dist(u"Mpc", cosmo, z_star)))
    ρm0 = T(2.77536627e11) * OmegaM * h^2   # Msun/Mpc³, comoving
    z2chi = build_z2r_interpolator(T(0.0), T(z_max), cosmo)
    gx = _build_nfw_kappa_gx(T, cnfw, xmax)
    return NFWKappaProfile(cosmo, cnfw, xmax, χstar, ρm0, z2chi, gx)
end

# dimensionless density in r_s units: NFW inside c, (c/u)² tail to xmax·c, zero beyond
function _nfw_trunc_density(u::T, c::T, umax::T) where T
    u < c && return 1 / (u * (1 + u)^2)
    u < umax && return c / ((1 + c)^2 * u^2)
    return zero(T)
end

# g(x) = 2∫₀^lmax f(√(l²+x²)) dl : dimensionless projected profile, Σ = ρ_s·r_s·g(R/r_s)
function _nfw_kappa_g_quadrature(x::T, c::T, umax::T) where T
    x >= umax && return zero(T)
    lmax = sqrt(umax^2 - x^2)
    integral, _ = quadgk(l -> _nfw_trunc_density(sqrt(l^2 + x^2), c, umax), zero(T), lmax,
                         rtol=sqrt(eps(T)))
    return 2integral
end

# g(x) is self-similar (depends only on c, xmax): tabulate once on a log-x grid
function _build_nfw_kappa_gx(::Type{T}, cnfw, xmax; N=1024, xmin=1e-8) where T
    umax = xmax * cnfw
    logxs = range(T(log(xmin)), T(log(umax)), length=N)
    g = [_nfw_kappa_g_quadrature(exp(lx), T(cnfw), T(umax)) for lx in logxs]
    return cubic_spline_interpolation(logxs, T.(g), extrapolation_bc=Flat())
end

"comoving r200m [Mpc] of a halo of mass M200m [Msun]"
r200m_comoving(model::NFWKappaProfile, M_Msun) = cbrt(3 * M_Msun / (800π * model.ρm0))

"CMB lensing kernel W_κ(z) [1/Mpc] for comoving-distance integration"
function lensing_kernel(model::NFWKappaProfile{T}, z) where T
    χ = model.z2chi(z)
    H0_c = model.cosmo.h / T(2997.92458)   # H0/c in 1/Mpc
    return T(1.5) * model.cosmo.Ω_m * H0_c^2 * (1 + z) * χ * (1 - χ / model.χstar)
end

"total projected mass in Msun (M200m + the r⁻² extension out to xmax·r200m)"
function total_kappa_mass(model::NFWKappaProfile, M_Msun)
    c = model.cnfw
    tail = c^2 * (model.xmax - 1) / ((1 + c)^2 * f_nfw(c))
    return M_Msun * (1 + tail)
end

function convergence(model::NFWKappaProfile{T}, θ, M_Msun, z) where T
    χ = model.z2chi(z)
    r200 = r200m_comoving(model, M_Msun)
    rs = r200 / model.cnfw
    x = max(θ, eps(T)) * χ / rs
    x >= model.xmax * model.cnfw && return zero(T)
    g = max(model.gx(log(x)), zero(T))
    ρs_over_ρm = 200 * model.cnfw^3 / (3 * f_nfw(model.cnfw))   # ρ_s/ρ̄_m for NFW at Δ=200m
    return lensing_kernel(model, z) * ρs_over_ρm * rs * g
end

# evaluation functions never take Unitful inputs; θ in radians, mass in Msun (M200m)
(model::NFWKappaProfile)(θ, M_Msun, z) = convergence(model, θ, M_Msun, z)

# the profile is exactly zero beyond xmax·r200m, so the paint radius is known analytically
# (the generic compute_θmax uses 4×R200c, which over-covers by ~35%)
function compute_θmax(model::NFWKappaProfile{T}, M_Δ, z; mult=1) where T
    M_Msun = M_Δ isa Unitful.Mass ? ustrip(u"Msun", M_Δ) : M_Δ
    return T(mult * model.xmax * r200m_comoving(model, M_Msun) / model.z2chi(z))
end
