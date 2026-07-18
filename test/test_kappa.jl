using Test
using XGPaint
using Pixell
using QuadGK

@testset "kappa NFW lensing profile" begin
    # Websky cosmology: Om=0.31, Ob=0.049, h=0.68
    model = NFWKappaProfile(Omega_c=0.261, Omega_b=0.049, h=0.68)

    # source plane: chi* to z=1089 should be ~13.9-14.0 Gpc (paper: ~13.8-14.2 Gpc)
    @test 13.5e3 < model.χstar < 14.5e3

    # kernel: zero at observer and source, positive in between, peaks around z~1-2
    @test lensing_kernel(model, 1e-4) ≈ 0 atol = 1e-7
    @test lensing_kernel(model, 1.5) > lensing_kernel(model, 0.1)
    @test lensing_kernel(model, 1.5) > 0

    # mass conservation: kappa integrated over solid angle recovers the total projected
    # mass times the kernel: ∫κ(θ)2πθdθ = W_κ(z)·M_tot/(ρ̄_m χ²)
    M, z = 3e14, 0.7
    χ = model.z2chi(z)
    θmax = XGPaint.compute_θmax(model, M, z)
    integral, _ = quadgk(θ -> model(θ, M, z) * 2π * θ, 0.0, θmax, rtol=1e-8)
    expected = lensing_kernel(model, z) * total_kappa_mass(model, M) / (model.ρm0 * χ^2)
    @test integral ≈ expected rtol = 2e-3

    # total mass: c=7, xmax=2 → M_tot/M200m = 1 + 49/64/f_nfw(7) ≈ 1.636
    @test total_kappa_mass(model, 1.0) ≈ 1 + 49 / 64 / XGPaint.f_nfw(7.0) rtol = 1e-12

    # profile is exactly zero beyond the truncation radius, positive and decreasing inside
    @test model(1.01 * θmax, M, z) == 0
    θs = θmax .* [0.01, 0.05, 0.2, 0.8]
    κs = [model(θ, M, z) for θ in θs]
    @test all(κs .> 0) && issorted(κs, rev=true)

    # amplitude sanity: massive cluster at z=0.5 has κ ~ 0.1-1 within an arcminute
    @test 0.03 < model(deg2rad(1 / 60), 1e15, 0.5) < 3

    # paint smoke test: two halos on a small CAR patch; map integral = sum of the
    # analytic per-halo integrals (pixel area ~ cos(dec)·Δα·Δδ near the equator).
    # Positions must not be grid-aligned and pixels must resolve r_s: pixel-center
    # sampling of the NFW cusp is pathological for a halo exactly on a pixel center
    # (painted/analytic 1.5 at 0.25', vs 1.02 generic; converges to 1.000 by 0.05').
    box = [1.0 -1.0; -1.0 1.0] * Pixell.degree
    res = 0.05 * Pixell.arcminute
    shape, wcs = geometry(Pixell.CarClenshawCurtis{Float64}, box, res)
    m = Enmap(zeros(shape), wcs)
    workspace = profileworkspace(shape, wcs)
    masses = [3e14, 8e13]
    zs = [0.7, 1.5]
    αs = deg2rad.([0.137, 0.311])
    δs = deg2rad.([0.0959, -0.213])
    paint!(m, workspace, model, masses, zs, αs, δs)
    @test maximum(m) > 0
    pixarea = (0.05 * π / 180 / 60)^2   # sr
    mapint = sum(m) * pixarea
    expint = sum(lensing_kernel(model, zs[i]) * total_kappa_mass(model, masses[i]) /
                 (model.ρm0 * model.z2chi(zs[i])^2) for i in 1:2)
    @test mapint ≈ expint rtol = 1e-2
end

@testset "kappa compensated profile (Websky overdensity-3 sphere)" begin
    model = NFWKappaProfile(Omega_c=0.261, Omega_b=0.049, h=0.68)
    modelc = NFWKappaProfile(Omega_c=0.261, Omega_b=0.049, h=0.68, delta_comp=3.0)
    M, z = 3e14, 0.7

    # compensation sphere: same total mass at overdensity 3, R = 4.78 r200m > 2 r200m
    Rc = comp_radius_comoving(modelc, M)
    @test Rc ≈ cbrt(3 * total_kappa_mass(modelc, M) / (4π * 3 * modelc.ρm0)) rtol = 1e-12
    @test Rc / r200m_comoving(modelc, M) ≈ cbrt(1.636 * 200 / 3) rtol = 1e-2
    @test comp_radius_comoving(model, M) == 0

    # paint radius covers the (larger) compensation sphere
    θmax = XGPaint.compute_θmax(modelc, M, z)
    @test θmax ≈ Rc / modelc.z2chi(z) rtol = 1e-12

    # net projected mass is zero: ∫κ_comp(θ)2πθdθ ≈ 0 (to a tiny fraction of the
    # uncompensated integral); NFW cusp positive at center, negative wings outside
    intc, _ = quadgk(θ -> modelc(θ, M, z) * 2π * θ, 0.0, θmax, rtol=1e-9)
    intu, _ = quadgk(θ -> model(θ, M, z) * 2π * θ, 0.0, θmax, rtol=1e-9)
    @test abs(intc) < 3e-3 * intu
    @test modelc(1e-6, M, z) > 0
    @test modelc(0.9 * θmax, M, z) < 0

    # inside the NFW region the compensated profile is the plain one minus the sphere
    θtest = 0.3 * XGPaint.compute_θmax(model, M, z)
    b = θtest * modelc.z2chi(z)
    expected = model(θtest, M, z) - lensing_kernel(modelc, z) * 3 * 2 * sqrt(Rc^2 - b^2)
    @test modelc(θtest, M, z) ≈ expected rtol = 1e-10
end
