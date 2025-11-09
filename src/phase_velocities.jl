# Phase Velocities
# This may make a nice stand-alone package to collect different velocity parameterisations and convert between them
# As a semantic point, maybe better to have a phase_speed function that returns only propagation speed and phase_velocity
# function that returns both speed and polarisation.

# Seismic Phase Types -- Used to dispatch on velocity calculations

# Refractions, Reflections, Conversions (can be either refracted or reflected arrivals)
abstract type SeismicPhase end
struct UnspecifiedPhase <: SeismicPhase end
struct BodyP <: SeismicPhase end
struct BodyS{T} <: SeismicPhase
    pazm::T
end
struct BodySfast <: SeismicPhase end
struct BodySslow <: SeismicPhase end
struct BodySmean <: SeismicPhase end
struct BodySiso <: SeismicPhase end

# Isotropic Velocity
struct IsotropicVelocity{T}
    vp::T
    vs::T
end
function Base.getindex(V::IsotropicVelocity, index...)
    return IsotropicVelocity(V.vp[index...], V.vs[index...])
end
function phase_velocity(V::IsotropicVelocity)
    return V.vp, V.vs
end
function phase_velocity(::BodyP, V::IsotropicVelocity)
    return V.vp
end
function phase_velocity(::BodyS, V::IsotropicVelocity)
    return V.vs
end
function phase_velocity(::BodySmean, V::IsotropicVelocity)
    return V.vs
end
function phase_velocity(::BodySiso, V::IsotropicVelocity)
    return V.vs
end

# Arbirtrary Elliptical Velocity
struct EllipticalVelocity{T}
    v_mean::T
    f::T
    sx::T
    sy::T
    sz::T
end
function EllipticalVelocity(v_mean, f, azm, elv)
    sx, sy, sz = unit_vector(azm, elv)
    return EllipticalVelocity(v_mean, f, sx, sy, sz)
end
# Indexing
function Base.getindex(V::EllipticalVelocity, index...)
    return EllipticalVelocity(V.v_mean[index...], V.f[index...], V.sx[index...], V.sy[index...], V.sz[index...])
end
function phase_velocity(V::EllipticalVelocity, (ux, uy, uz))
    dv = V.f*V.v_mean
    cosθ = V.sx*ux + V.sy*uy + V.sz*uz
    vani = V.v_mean + 2.0*dv*cosθ*cosθ - dv
    return vani
end
# Method for checking performance when using angles instead of unit vectors
function phase_velocity(V::EllipticalVelocity, propagation_azimuth, propagation_elevation)
    sym_azm, sym_elv, _ = cartesian_to_spherical(V.sx, V.sy, V.sz)
    cosθ = symmetry_axis_cosine(sym_azm, sym_elv, propagation_azimuth, propagation_elevation)
    dv = V.f*V.v_mean
    return V.v_mean + 2.0*dv*cosθ*cosθ - dv
end

# UNFINISHED! Intended to deprecate ThomsenVelocity and use faster symmetry axis vector parameterisation
# SYMMETRY AXIS VECTOR IS ONLY FASTER FOR P: For S, a simple dot product is not sufficient
struct WeakHexagonalThomsenVelocity{T,A}
    alpha::T
    beta::T
    epsilon::T
    delta::T
    gamma::T
    symmetry::A
end
# Outer Constructors
function WeakHexagonalThomsenVelocity(alpha, beta, epsilon, delta, gamma, azm, elv)
    CI = CartesianIndices(size(alpha))
    symmetry = zeros(eltype(alpha), 3, size(alpha)...)
    for n in eachindex(azm)
        sinλ, cosλ = sincos(azm[n])
        sinϕ, cosϕ = sincos(elv[n])
        jkl = CI[n]
        symmetry[1,jkl], symmetry[2,jkl], symmetry[3,jkl] = cosϕ*cosλ, cosϕ*sinλ, sinϕ
    end
    return WeakHexagonalThomsenVelocity(alpha, beta, epsilon, delta, gamma, symmetry)
end
# Indexing
function Base.getindex(V::WeakHexagonalThomsenVelocity, index...)
    α, β, ϵ, δ, γ = V.alpha[index...], V.beta[index...], V.epsilon[index...], V.delta[index...], V.gamma[index...]
    s = @view V.symmetry[:, index...]
    return WeakHexagonalThomsenVelocity(α, β, ϵ, δ, γ, s)
end

# Weak Hexagonal Anisotropy using Thomsen's parameters
struct ThomsenVelocity{T}
    alpha::T
    beta::T
    epsilon::T
    delta::T
    gamma::T
    azm::T
    elv::T
end
# Indexing
function Base.getindex(V::ThomsenVelocity, index...)
    α, β, ϵ, δ, γ = V.alpha[index...], V.beta[index...], V.epsilon[index...], V.delta[index...], V.gamma[index...]
    λ, ϕ = V.azm[index...], V.elv[index...]
    return ThomsenVelocity(α, β, ϵ, δ, γ, λ, ϕ)
end
function phase_velocity(V::ThomsenVelocity, propagation_azimuth, propagation_elevation)
    cosθ = symmetry_axis_cosine(V.azm, V.elv, propagation_azimuth, propagation_elevation)
    cosθ_2 = cosθ^2
    sinθ_2 = 1.0 - cosθ_2
    q_αβ = V.alpha/V.beta
    vqp = V.alpha*sqrt( 1.0 + 2.0*V.epsilon*(sinθ_2^2) + 2.0*V.delta*sinθ_2*cosθ_2 )
    vqs1 = V.beta*sqrt( 1.0 + 2.0*(q_αβ^2)*(V.epsilon - V.delta)*sinθ_2*cosθ_2 )
    vqs2 = V.beta*sqrt( 1.0 + 2.0*V.gamma*sinθ_2 )
    return vqp, vqs1, vqs2
end
function phase_velocity(::BodyP, V::ThomsenVelocity, propagation_azimuth, propagation_elevation)
    cosθ = symmetry_axis_cosine(V.azm, V.elv, propagation_azimuth, propagation_elevation)
    cosθ_2 = cosθ^2
    sinθ_2 = 1.0 - cosθ_2
    return V.alpha*sqrt( 1.0 + 2.0*V.epsilon*(sinθ_2^2) + 2.0*V.delta*sinθ_2*cosθ_2 )
end
function phase_velocity(P::BodyS, V::ThomsenVelocity, propagation_azimuth, propagation_elevation)
    cosθ, ζ = symmetry_axis_cosine(V.azm, V.elv, propagation_azimuth, propagation_elevation, P.pazm)
    cosθ_2 = cosθ^2
    sinθ_2 = 1.0 - cosθ_2
    q_αβ = V.alpha/V.beta
    vqs1 = V.beta*sqrt( 1.0 + 2.0*(q_αβ^2)*(V.epsilon - V.delta)*sinθ_2*cosθ_2 )
    vqs2 = V.beta*sqrt( 1.0 + 2.0*V.gamma*sinθ_2 )
    uqs1, uqs2 = 1.0/vqs1, 1.0/vqs2
    us = uqs2 - (uqs2 - uqs1)*(cos(ζ)^2)
    return 1.0/us
end
function phase_velocity(::BodySfast, V::ThomsenVelocity, propagation_azimuth, propagation_elevation)
    cosθ = symmetry_axis_cosine(V.azm, V.elv, propagation_azimuth, propagation_elevation)
    cosθ_2 = cosθ^2
    sinθ_2 = 1.0 - cosθ_2
    q_αβ = V.alpha/V.beta
    vqs1 = V.beta*sqrt( 1.0 + 2.0*(q_αβ^2)*(V.epsilon - V.delta)*sinθ_2*cosθ_2 )
    vqs2 = V.beta*sqrt( 1.0 + 2.0*V.gamma*sinθ_2 )
    return max(vqs1, vqs2)
end
function phase_velocity(::BodySslow, V::ThomsenVelocity, propagation_azimuth, propagation_elevation)
    cosθ = symmetry_axis_cosine(V.azm, V.elv, propagation_azimuth, propagation_elevation)
    cosθ_2 = cosθ^2
    sinθ_2 = 1.0 - cosθ_2
    q_αβ = V.alpha/V.beta
    vqs1 = V.beta*sqrt( 1.0 + 2.0*(q_αβ^2)*(V.epsilon - V.delta)*sinθ_2*cosθ_2 )
    vqs2 = V.beta*sqrt( 1.0 + 2.0*V.gamma*sinθ_2 )
    return min(vqs1, vqs2)
end
function phase_velocity(::BodySmean, V::ThomsenVelocity, propagation_azimuth, propagation_elevation)
    cosθ = symmetry_axis_cosine(V.azm, V.elv, propagation_azimuth, propagation_elevation)
    cosθ_2 = cosθ^2
    sinθ_2 = 1.0 - cosθ_2
    q_αβ = V.alpha/V.beta
    vqs1 = V.beta*sqrt( 1.0 + 2.0*(q_αβ^2)*(V.epsilon - V.delta)*sinθ_2*cosθ_2 )
    vqs2 = V.beta*sqrt( 1.0 + 2.0*V.gamma*sinθ_2 )
    return 0.5*(vqs1 + vqs2)
end
function phase_velocity(::BodySiso, V::ThomsenVelocity, propagation_azimuth, propagation_elevation)
    return V.beta*sqrt( 1.0 + (2/3)*V.gamma + (2/15)*((V.alpha^2)/(V.beta^2))*(V.epsilon - V.delta) )
end

#############
# UTILITIES #
#############

# Interpolate array weights
function interpolate_vert_weights(x::LinRange, y::LinRange, z::LinRange, vert_weights::Array,
    xq::Array, yq::Array, zq::Array)
    # Construct an interpolant with nearest-boundary value extrapolation
    itp = extrapolate(interpolate((x,y,z), vert_weights, Gridded(Linear())), Flat())
    # Interpolate weights
    intp_weights = zeros(size(xq))
    [intp_weights[i] = itp(xq[i], yq[i], zq[i]) for i in eachindex(intp_weights)]
    return intp_weights
end
# May be type unstable...but prioritizing for convenience at the moment
function interpolate_vert_weights(x::LinRange, y::LinRange, z::LinRange, vert_weights,
    xq::Array, yq::Array, zq::Array)
    new_weights = deepcopy(vert_weights)
    flds = fieldnames(typeof(vert_weights))
    for f_i in flds
        v_f = getfield(new_weights, f_i)
        v_f .= interpolate_vert_weights(x, y, z, getfield(vert_weights, f_i), xq, yq, zq)
    end
    return new_weights
end

function symmetry_axis_cosine(symmetry_azimuth, symmetry_elevation, propagation_azimuth, propagation_elevation)
    cosΔλ = cos(propagation_azimuth - symmetry_azimuth) 
    sinϕp, cosϕp = sincos(propagation_elevation)
    sinϕs, cosϕs = sincos(symmetry_elevation)
    # Cosine of angle between propagation direction and symmetry axis
    cosθ = cosΔλ*cosϕp*cosϕs + sinϕp*sinϕs
    
    return cosθ
end
function symmetry_axis_cosine(symmetry_azimuth, symmetry_elevation, propagation_azimuth, propagation_elevation, qt_polarization)
    sinΔλ, cosΔλ = sincos(propagation_azimuth - symmetry_azimuth) 
    sinϕp, cosϕp = sincos(propagation_elevation)
    sinϕs, cosϕs = sincos(symmetry_elevation)
    # Cosine of angle between propagation direction and symmetry axis
    cosθ = cosΔλ*cosϕp*cosϕs + sinϕp*sinϕs
    # Angle between polarization vector and projection of symmetry axis in QT-plane (i.e. ray-normal plane)
    # Do not return cos(ζ). The sign of this angle is important for splitting intensity.
    ζ = atan(-sinΔλ*cosϕs, cosΔλ*sinϕp*cosϕs - cosϕp*sinϕs) - qt_polarization
    
    return cosθ, ζ
end