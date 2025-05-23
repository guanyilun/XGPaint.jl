
struct RSZPerturbativeProfile{T,C} <: AbstractGNFW{T}
    f_b::T  # Omega_b / Omega_c = 0.0486 / 0.2589
    cosmo::C
    X::T  # X = 2.6408 corresponding to frequency 150 GHz
end

function RSZPerturbativeProfile(; Omega_c::T=0.2589, Omega_b::T=0.0486, h::T=0.6774, x::T=2.6408) where {T <: Real}
    OmegaM=Omega_b+Omega_c
    f_b = Omega_b / OmegaM
    cosmo = get_cosmology(T, h=h, OmegaM=OmegaM)
    X = x
    return RSZPerturbativeProfile(f_b, cosmo, X)
end

# wrap the compton y in a type so it uses the Battaglia+16 profile
function compton_y(model::RSZPerturbativeProfile, r, M_200c, z)
    return P_e_los(
        Battaglia16ThermalSZProfile(model.f_b, model.cosmo), r, M_200c, z) * P_e_factor
end

"""
    T_vir_calc(model, M, z)

Calculates the virial temperature for a given halo using Wang et al. 2007.
"""
function T_vir_calc(model::AbstractProfile{T}, M, z) where T

    µ = 0.6  #µ is the mean molecular weight -> used the primordial abundance
    if z >= 1
        d_c = T(178)
    else
        d_c = T(356/(1 + z))
    end
    T_vir = 4.8e-3 * (M/M_sun)^(2/3) * (1 + z) * (model.cosmo.Ω_m/0.3)^(1/3) * (d_c/178)^(1/3) * (µ/0.59) * u"K"
    return T_vir
end

"""
    T_mass_calc(model,M,z::T; scale_type="Ty", sim_type="combination") where T

Calculates the temperature for a given halo using https://arxiv.org/pdf/2207.05834.pdf.
"""
function T_mass_calc(model::AbstractProfile{T}, M, z; scale_type="Ty", sim_type="combination") where T

    E_z = model.cosmo.Ω_m*(1 + z)^3 + model.cosmo.Ω_Λ
    par_dict_scale = Dict([("Ty",[1.426,0.566,0.024]),("Tm",[1.207,0.589,0.003]),("Tsl",[1.196,0.641,-0.048])])
    par_dict_sim = Dict([("combination",[1.426,0.566,0.024]),("bahamas",[2.690,0.323,0.023]),("the300",[2.294,0.350,0.013]),("magneticum",[2.789,0.379,0.030]),("tng",[2.154,0.365,0.032])])
    
    if scale_type=="Ty"
        params = par_dict_sim[sim_type]
    else
        params = par_dict_scale[scale_type]
    end
    
    T_e = E_z^(2/3) * params[1] * (M/(10^14 *M_sun))^(params[2] + params[3] * log10(M/(10^14 * M_sun))) * u"keV"
    
    return T_e  
end

"""
    rSZ_perturbative(model, r, M_200, z; T_scale="virial", sim_type="combination", showT=true)

Calculates the integrated relativistic compton-y signal along the line of sight. Mass 
requires units.
"""
function rSZ_perturbative(model, r, M_200, z; T_scale="virial", sim_type="combination", showT=false)

    #X = (constants.ħ*ω)/(constants.k_B*T_cmb) # omega is standard frequency in Hz
    X = model.X
    if T_scale=="virial"
        T_e = T_vir_calc(model, M_200, z)
    elseif typeof(T_scale)==String
        T_e = uconvert(u"K",(T_mass_calc(model, M_200, z, scale_type=T_scale, sim_type=sim_type)/constants.k_B))
    else
        T_e = T_scale
    end
    θ_e = (constants.k_B*T_e)/(constants.m_e*constants.c_0^2)
    ω = (X*constants.k_B*T_cmb)/constants.h

    Xt = X*coth(X/2)
    St = X/(sinh(X/2))

    Y_0 = -4 + Xt
    Y_1 = -10 + (47/2)*Xt -(42/5)*Xt^2 + (7/10)*Xt^3 + St^2*((-21/5)+(7/5)*Xt)
    Y_2 = (-15/2) + (1023/8)*Xt - (868/5)*Xt^2 + (329/5)*Xt^3 - (44/5)*Xt^4 + (11/30)*Xt^5 +
        St^2*((-434/5) + (658/5)*Xt - (242/5)*Xt^2 + (143/30)*Xt^3) + St^4*((-44/4) + (187/60)*Xt)
    Y_3 = (15/2) + (2505/8)*Xt - (7098/5)*Xt^2 + (14253/10)*Xt^3 - (18594/35)*Xt^4 + (12059/140)*Xt^5 -
        (128/21)*Xt^6 + (16/105)*Xt^7 + St^2*((-7098/10) + (14253/5)*Xt - (102267/35)*Xt^2 +
        (156767/140)*Xt^3 - (1216/7)*Xt^4 + (64/7)*Xt^5) + St^4*((-18594/35) + (205003/280)*Xt -
        (1920/7)*Xt^2 + (1024/35)*Xt^3) + St^6*((-544/21) + (992/105)*Xt)
    Y_4 = (-135/32) + (30375/128)*Xt - (62391/10)*Xt^2 + (614727/40)*Xt^3 - (124389/10)*Xt^4 +
        (355703/80)*Xt^5 - (16568/21)*Xt^6 + (7516/105)*Xt^7 - (22/7)*Xt^8 + (11/210)*Xt^9 +
        St^2*((-62391/20) + (614727/20)*Xt - (1368279/20)*Xt^2 + (4624139/80)*Xt^3 - (157396/7)*Xt^4 +
        (30064/7)*Xt^5 - (2717/7)*Xt^6 + (2761/210)*Xt^7) + St^4*((-124389/10) + (6046951/160)*Xt -
        (248520/7)*Xt^2 + (481024/35)*Xt^3 - (15972/7)*Xt^4 + (18689/140)*Xt^5) + St^6*((-70414/21) +
        (465992/105)*Xt - (11792/7)*Xt^2 + (19778/105)*Xt^3) + St^8*((-628/7) + (7601/210)*Xt)

    prefac = ((X*ℯ^X)/(ℯ^X-1))*θ_e*(Y_0+θ_e*Y_1+θ_e^2*Y_2+θ_e^3*Y_3+θ_e^4*Y_4)
    y = compton_y(model, r, M_200, z)
    n = prefac * (constants.m_e*constants.c_0^2)/(T_e*constants.k_B) * y
    I = (X^3/(ℯ^X-1)) * (2*(constants.k_B*T_cmb)^3)/((constants.h*constants.c_0)^2) * n 
    T_over_Tcmb = I/abs((2 * constants.h^2 * ω^4 * ℯ^X)/(
        constants.k_B * constants.c_0^2 * T_cmb * (ℯ^X - 1)^2))

    if showT
        return abs(T_over_Tcmb) + 0
    else
        return I
    end
end

function (model::RSZPerturbativeProfile)(r, M, z; 
        T_scale="virial", sim_type="combination", showT=false)
    return ustrip(uconvert(u"MJy/sr", rSZ_perturbative(
        model, r, M, z, T_scale=T_scale, sim_type=sim_type, showT=showT)))
end

function calc_null(model, M_200, z)
    T_e = T_vir_calc(model, M_200 * M_sun, z)
    θ_e = (constants.k_B*T_e)/(constants.m_e*constants.c_0^2)
    X_0 = 3.830*(1 + 1.1674*θ_e - 0.8533*(θ_e^2))
    return X_0
end

# like the usual paint, but use the sign of the null as the sign of the perturbation
function profile_paint!(m::Enmap{T, 2, Matrix{T}, CarClenshawCurtis{T}}, 
                        workspace::CarClenshawCurtisProfileWorkspace, 
                        model::RSZPerturbativeProfile, Mh, z, α₀, δ₀, θmax) where T
    X_0 = calc_null(model, Mh, z)
    profile_paint_generic!(m, model, workspace, Mh, z, α₀, δ₀, θmax, sign(model.X - X_0))
end

# like the usual paint, but use the sign of the null as the sign of the perturbation
function profile_paint!(m::HealpixMap{T, RingOrder}, 
        workspace::HealpixSerialProfileWorkspace, model::RSZPerturbativeProfile, 
        Mh, z, α₀, δ₀, θmax) where T
    X_0 = calc_null(model, Mh, z)
    profile_paint_generic!(m, workspace, model, Mh, z, α₀, δ₀, θmax, sign(X - X_0))
end
