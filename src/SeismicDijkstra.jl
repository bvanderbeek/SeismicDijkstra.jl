module SeismicDijkstra

# SeismicDijkstra: Seismic ray tracing with Dijkstra's shortest path algorithm.

# To-do List
# + How to handle different phases? (P, S, S-slow, S-fast, PmP)
# -> Refractions, Reflections; Single, Multi-leg; Direct, Indirect
# -> Either graph parameters are phase specific or we include a phase parameter
# + Spherical Bullen gradient benchmark
# + Add elevation
# -> Remove nodes above some topographic surface?
# -> Perturb surface nodes such that they follow exactly topography?
# -> Check that connection does not pass through 'air'
# + Implement anisotropic tracing
# -> add arc_weight functions that accommodate different vertex weights
# + Add carving
# + Add more complicated phases (reflections, conversions)
# + Add line integration (can pre-compute nearest indices along connections)

# Quiaro & Sacchi (GJI 2024; Shortest-path ray tracing on self-adapting random grids) provides a nice
# review of modifications to this method that would be worth exploring/including.

# Could we use the analytic solutions for travel-times in a linear gradient to improve travel-time estimates?

# Optimization Tips
# + Use performant priority queue (QuickHeaps)
# -> https://discourse.julialang.org/t/fast-er-priority-queues/81269/13
# + Avoid cache misses
# -> Minimise large array indexing
# -> Loop through forward star vertices in sensible way
# + Make sure *complete* type signatures are visible in structures
# + Using @inbounds has appreciable effect on performance

using QuickHeaps
using DelimitedFiles
using Interpolations
using NearestNeighbors
using StaticArrays
using WriteVTK

include("utilities.jl")
include("phase_velocities.jl")
# include("IndexedMinPQ.jl") # ChatGPT Priority Queue

export load_velocity_1D_file, load_aquisition_file, load_observation_file, make_graph, travel_times

##################
### STRUCTURES ###
##################

# Dijkstra Data: Stores the main Dijkstra algorithm data structures
struct DijkstraData{Q, S, L<:Vector, P<:Vector, B<:BitArray} # PriorityQueue Type in signature; needs to be known at compile time for good performance!
    Queue::Q # Priority Queue
    lengths::L # Objective function 
    predecessors::P # Previous vertex/next vertex in path
    distinguished::B # Visited vertices
    source::S # Store initialization type (e.g. point source, interface source, exploding reflector)
end
function DijkstraData(num_vertices, source)
    # Q = IndexedMinPQ{Float64}(num_vertices) # ChatGPT Priority Queue
    Q = FastPriorityQueue{Float64}(num_vertices)
    P = PointSource(source...)
    return DijkstraData(Q, fill(Inf, num_vertices), zeros(Int, num_vertices), falses(num_vertices), P)
end

abstract type SourceType end
struct PointSource{T} <: SourceType
    x::T
    y::T
    z::T
end
struct ExplodingReflector <: SourceType end
struct PlaneWave <: SourceType end

# Structured Graph: Vertices are organized in regular array but spacing varies with position
# Neighbour indices (connectivity) is constant
# Better to just put dimensions in type signature?
struct StructuredGraph3D{A<:AbstractArray,V,N,T}
    x::A
    y::A
    z::A
    nx::Int
    ny::Int
    nz::Int
    NNTree::T
    forward_star::N
    vert_weights::V
    num_vertices::Int
    num_vertices_xy::Int
end
# Build structured graph from regular weight arrays
function StructuredGraph3D(x1::LinRange, x2::LinRange, x3::LinRange, vert_weights;
    grid_noise = 0.0, grid_coords = (a,b,c) -> (a,b,c),
    neighbours = 5, leafsize = 10)
    nx1, nx2, nx3 = length(x1), length(x2), length(x3)
    dx1, dx2, dx3 = grid_noise*step(x1), grid_noise*step(x2), grid_noise*step(x3)
    xg, yg, zg = zeros(nx1, nx2, nx3), zeros(nx1, nx2, nx3), zeros(nx1, nx2, nx3)
    for (k, x3_k) in enumerate(x3), (j, x2_j) in enumerate(x2), (i, x1_i) in enumerate(x1)
        # Define (optional) noisy coordinate arrays
        ddx1, ddx2, ddx3 = (2.0*rand() - 1.0)*dx1, (2.0*rand() - 1.0)*dx2, (2.0*rand() - 1.0)*dx3
        xg[i,j,k], yg[i,j,k], zg[i,j,k] = x1_i + ddx1, x2_j + ddx2, x3_k + ddx3
    end
    # Re-interpolate vertex weights (do before coordinate system conversion!)
    # Creates new weight array (do not want to overwrite inputs)
    new_weights = interpolate_vert_weights(x1, x2, x3, vert_weights, xg, yg, zg)
    # Apply (optional) conversion to cartesian coordinates
    [(xg[ind], yg[ind], zg[ind]) = grid_coords(xg[ind], yg[ind], zg[ind]) for ind in eachindex(xg)]

    return StructuredGraph3D(xg, yg, zg, (nx1, nx2, nx3), new_weights; r_neighbours = neighbours, leafsize = leafsize)
end
function StructuredGraph3D(x, y, z, n_xyz, vert_weights; r_neighbours = 5, leafsize = 10)
    # Build nearest neighbours interpolation tree...to simplify locating points inside graph
    NNTree = KDTree(transpose(hcat(vec(x), vec(y), vec(z))); leafsize = leafsize)
    # Define search ranges
    nx, ny, nz = n_xyz
    ri, rj, rk = min(nx - 1, r_neighbours), min(ny - 1, r_neighbours), min(nz - 1, r_neighbours)
    forward_star = -ri:ri, -rj:rj, -rk:rk
    return StructuredGraph3D(x, y, z, nx, ny, nz, NNTree, forward_star, vert_weights, nx*ny*nz, nx*ny)
end
# StructuredGraph3D Methods
function Base.size(G::StructuredGraph3D)
    return G.nx, G.ny, G.nz
end
function linear_index(G::StructuredGraph3D, q_ijk)
    return linear_index(G.nx, G.num_vertices_xy, q_ijk)
end
function cartesian_index(G::StructuredGraph3D, q)
    return cartesian_index(G.nx, G.num_vertices_xy, q)
end
function get_nearest_vertex(G::StructuredGraph3D, p_xyz)
    q, _ = nn(G.NNTree, p_xyz)
    q_ijk = cartesian_index(G, q)
    return q, q_ijk
end


################
### DIJKSTRA ###
################

# Initialize Dijkstra data structures for path calculations
# Eventually there will be different initialization functions for different source types (e.g. reflections)
function initialize_dijkstra(G, p_xyz; phase = UnspecifiedPhase(), length_0 = 0.0, pred = 0)
    D = DijkstraData(G.num_vertices, p_xyz)
    # # Exclude vertices above surface (if defined) -- ASSUMES Z COORDINATE IS ELEVATION...NOT TRUE FOR GLOBAL CARTESIAN
    # if haskey(G.interfaces, "surface")
    #     for j in axes(G.interfaces["surface"]), i in axes(G.interfaces["surface"])
    #     end
    # end
    initialize_dijkstra!(D, G; phase = phase, length_0 = length_0, pred = pred)
    return D
end
function initialize_dijkstra!(D, G; phase = UnspecifiedPhase(), length_0 = 0.0, pred = 0)
    # Nearest vertex to initialization point
    p_xyz = [D.source.x, D.source.y, D.source.z] # Must be vector
    q, q_ijk = get_nearest_vertex(G, p_xyz)
    q_weight = G.vert_weights[q]

    # Evaluate length to nearest vertex
    if !D.distinguished[q]
        q_xyz = G.x[q], G.y[q], G.z[q]
        length_pq = length_0 + arc_weight(phase, q_weight, p_xyz, q_weight, q_xyz)
        if length_pq < D.lengths[q]
            D.Queue[q] = length_pq
            D.lengths[q] = length_pq
            D.predecessors[q] = pred
        end
    end

    # Compute neighbour lengths accounting for true initialisation point coordinate
    evaluate_neighbours!(D, G, length_0, pred, q_ijk, p_xyz, q_weight; phase = phase)

    return nothing
end

# Main Dijkstra loop
function dijkstra!(D, G; phase = UnspecifiedPhase())
    while !isempty(D.Queue)
        # Extract next minimum vertex
        q_min = dequeue!(D.Queue)
        q_ijk = cartesian_index(G, q_min)
        q_xyz = G.x[q_min], G.y[q_min], G.z[q_min]
        q_weight = G.vert_weights[q_min]
        # Update lengths to neighbours
        D.distinguished[q_min] = true
        evaluate_neighbours!(D, G, D.lengths[q_min], q_min, q_ijk, q_xyz, q_weight; phase = phase)
    end
    return nothing
end

# Compute lengths to neighbouring nodes (forward star evaluation)
function evaluate_neighbours!(D, G, length_pq, q, q_ijk, q_xyz, q_weight; phase = UnspecifiedPhase())
    # Convenience variables
    nx, ny, nz = size(G)
    q_i, q_j, q_k = q_ijk
    n_i, n_j, n_k = G.forward_star

    @inbounds for dk in n_k
        r_k = q_k + dk
        (r_k < 1 || r_k > nz) && continue
        @inbounds for dj in n_j
            r_j = q_j + dj
            (r_j < 1 || r_j > ny) && continue
            @inbounds for di in n_i
                r_i = q_i + di
                (r_i < 1 || r_i > nx) && continue
                # Check vertex r has not yet been distinguished
                r = linear_index(G, (r_i, r_j, r_k))
                D.distinguished[r] && continue

                # Check new path: length_pq + length_qr < length_pr ?
                r_xyz = G.x[r], G.y[r], G.z[r]
                length_pqr = length_pq + arc_weight(phase, q_weight, q_xyz, G.vert_weights[r], r_xyz)
                if length_pqr < D.lengths[r]
                    D.Queue[r] = length_pqr
                    D.lengths[r] = length_pqr
                    D.predecessors[r] = q
                end
            end
        end
    end

    return nothing
end

# Collection of arc weight functions (i.e. how long does it take to travel a specific path)
# Implement different arc weight functions for different vertex paramerisations and phases
function arc_weight(::SeismicPhase, q_weight, q_xyz, r_weight, r_xyz)
    dx, dy, dz = r_xyz[1] - q_xyz[1], r_xyz[2] - q_xyz[2], r_xyz[3] - q_xyz[3]
    dqr = sqrt(dx^2 + dy^2 + dz^2)
    return 0.5*(q_weight + r_weight)*dqr
end
function arc_weight(P::SeismicPhase, q_weight::T, q_xyz, r_weight::T, r_xyz) where {T<:EllipticalVelocity}
    dx, dy, dz = r_xyz[1] - q_xyz[1], r_xyz[2] - q_xyz[2], r_xyz[3] - q_xyz[3]
    dqr = sqrt(dx^2 + dy^2 + dz^2)
    if dqr > 0.0
        n_xyz = dx/dqr, dy/dqr, dz/dqr
        u_q, u_r = phase_velocity(q_weight, n_xyz), phase_velocity(r_weight, n_xyz)
        u_q, u_r = 1.0/u_q, 1.0/u_r
        w = 0.5*(u_q + u_r)*dqr
    else
        w = 0.0
    end

    # Compute velocities using angles...much slower
    # dx, dy, dz = r_xyz[1] - q_xyz[1], r_xyz[2] - q_xyz[2], r_xyz[3] - q_xyz[3]
    # ray_azm, ray_elv, ray_len = cartesian_to_spherical(dx, dy, dz)
    # u_q, u_r = phase_velocity(q_weight, ray_azm, ray_elv), phase_velocity(r_weight, ray_azm, ray_elv)
    # u_q, u_r = 1.0/u_q, 1.0/u_r
    # w = 0.5*(u_q + u_r)*ray_len
    return w
end
function arc_weight(P::SeismicPhase, q_weight::T, q_xyz, r_weight::T, r_xyz) where {T<:IsotropicVelocity}
    dx, dy, dz = r_xyz[1] - q_xyz[1], r_xyz[2] - q_xyz[2], r_xyz[3] - q_xyz[3]
    dqr = sqrt(dx^2 + dy^2 + dz^2)
    vq, vr = phase_velocity(P, q_weight), phase_velocity(P, r_weight)
    return 2.0*dqr/(vq + vr)
end
function arc_weight(P::SeismicPhase, q_weight::T, q_xyz, r_weight::T, r_xyz) where {T<:ThomsenVelocity}
    dx, dy, dz = r_xyz[1] - q_xyz[1], r_xyz[2] - q_xyz[2], r_xyz[3] - q_xyz[3]
    ray_azm, ray_elv, ray_len = cartesian_to_spherical(dx, dy, dz)
    vq, vr = phase_velocity(P, q_weight, ray_azm, ray_elv), phase_velocity(P, r_weight, ray_azm, ray_elv)
    return 2.0*ray_len/(vq + vr)
end

# Construct path starting from vertex q
function get_path(D, q)
    path = Vector{Int}()
    while q > 0
        push!(path, q)
        q = D.predecessors[q]
    end
    return path
end

# Construct path from arbitary point to initialization point (neither of which must coincide with graph vertex)
function get_path(D, G, xyz_start; phase = UnspecifiedPhase(), length_0 = 0.0)
    # Get end point (i.e. source initialisation point)
    xyz_end = D.source.x, D.source.y, D.source.z
    # Locate nearest vertex to desired path start point
    min_length, r_min = get_nearest_connection(D, G, xyz_start; phase = phase)
    min_length += length_0

    # Construct vertex path
    vert_path = get_path(D, r_min)

    # Check if exact start and end points are on path
    v_a, v_b = vert_path[1], vert_path[end]
    x_a, y_a, z_a = G.x[v_a], G.y[v_a], G.z[v_a]
    x_b, y_b, z_b = G.x[v_b], G.y[v_b], G.z[v_b]
    add_start = x_a == xyz_start[1] && y_a == xyz_start[2] && z_a == xyz_start[3] ? 0 : 1
    add_end = x_b == xyz_end[1] && y_b == xyz_end[2] && z_b == xyz_end[3] ? 0 : 1

    # Construct cartesian path
    xyz_path = zeros(3, add_start + add_end + length(vert_path))
    for (n, v) in enumerate(vert_path)
        col = n + add_start
        xyz_path[1,col], xyz_path[2,col], xyz_path[3,col] = G.x[v], G.y[v], G.z[v]
    end
    if add_start > 0
        xyz_path[1,1], xyz_path[2,1], xyz_path[3,1] = xyz_start[1], xyz_start[2], xyz_start[3]
    end
    if add_end > 0
        xyz_path[1,end], xyz_path[2,end], xyz_path[3,end] = xyz_end[1], xyz_end[2], xyz_end[3]
    end

    return min_length, xyz_path, vert_path
end

function refine_path(xyz_path, dr; min_vert = 2)

    # Compute path length 
    num_seg = size(xyz_path, 2) - 1
    len_path = zeros(size(xyz_path, 2))
    @views x, y, z = xyz_path[1,:], xyz_path[2,:], xyz_path[3,:]
    for i in 1:num_seg
        j = i + 1
        dx_i, dy_i, dz_i = x[j] - x[i], y[j] - y[i], z[j] - z[i]
        len_path[j] = len_path[i] + sqrt(dx_i^2 + dy_i^2 + dz_i^2)
    end
    total_length = len_path[end]

    # Create path interpolants
    itpx = linear_interpolation(len_path, x)
    itpy = linear_interpolation(len_path, y)
    itpz = linear_interpolation(len_path, z)

    # Re-interpolate path
    num_vert = max(1 + round(Int, total_length/dr), min_vert)
    fine_path = zeros(3, num_vert)
    rq = range(start = 0.0, stop = total_length, length = num_vert)
    @views xq, yq, zq = fine_path[1,:], fine_path[2,:], fine_path[3,:]
    [xq[k] = itpx(rq_k) for (k, rq_k) in enumerate(rq)]
    [yq[k] = itpy(rq_k) for (k, rq_k) in enumerate(rq)]
    [zq[k] = itpz(rq_k) for (k, rq_k) in enumerate(rq)]

    return fine_path
end
function refine_path(G, xyz_path, dr; phase = UnspecifiedPhase(), dist = 0.0, min_vert = 2)
    # Construct resampled path
    fine_path = refine_path(xyz_path, dr; min_vert = min_vert)
    # Re-integrate weights
    nvert = size(fine_path, 2)
    for i in 1:(nvert-1)
        xyz_q, xyz_r = @views fine_path[:,i], fine_path[:,i+1]
        q, _ = nn(G.NNTree, xyz_q)
        r, _ = nn(G.NNTree, xyz_r)
        weight_q, weight_r = G.vert_weights[q], G.vert_weights[r]
        dist += arc_weight(phase, weight_q, xyz_q, weight_r, xyz_r)
    end

    return dist, fine_path
end

# Returns shortest connection from an arbitrary point to a graph vertex
function get_nearest_connection(D, G, p_xyz; phase = UnspecifiedPhase())
    nx, ny, nz = size(G)
    # Locate nearest vertex to point p_xyz
    q, (q_i, q_j, q_k) = get_nearest_vertex(G, p_xyz)
    q_xyz = G.x[q], G.y[q], G.z[q]
    q_weight = G.vert_weights[q]

    # Locate shortest connection to point p_xyz
    min_length, r_min = D.lengths[q] + arc_weight(phase, q_weight, p_xyz, q_weight, q_xyz), q
    if p_xyz != q_xyz # Search neighbours if p_xyz is not a vertex
        n_i, n_j, n_k = G.forward_star
        @inbounds for dk in n_k
            r_k = q_k + dk
            (r_k < 1 || r_k > nz) && continue
            @inbounds for dj in n_j
                r_j = q_j + dj
                (r_j < 1 || r_j > ny) && continue
                @inbounds for di in n_i
                    r_i = q_i + di
                    (r_i < 1 || r_i > nx) && continue

                    # Check new path
                    r = linear_index(G, (r_i, r_j, r_k))
                    r_xyz = G.x[r], G.y[r], G.z[r]
                    length_pr = D.lengths[r] + arc_weight(phase, q_weight, p_xyz, G.vert_weights[r], r_xyz)
                    if length_pr < min_length
                        min_length, r_min = length_pr, r
                    end
                end
            end
        end
    end

    return min_length, r_min
end


#################
### UTILITIES ###
#################

function cartesian_index(dimsize::NTuple{3, Int}, linear_index::Int)
    linear_index -= 1
    k, linear_index = divrem(linear_index, dimsize[1]*dimsize[2])
    j, i = divrem(linear_index, dimsize[1])
    k += 1
    j += 1
    i += 1
    return i, j, k
end
function cartesian_index(ni::Int, ninj::Int, linear_index::Int)
    linear_index -= 1
    k, linear_index = divrem(linear_index, ninj)
    j, i = divrem(linear_index, ni)
    k += 1
    j += 1
    i += 1
    return i, j, k
end
function linear_index(dimsize::NTuple{3, Int}, cart_index::NTuple{3, Int})
    ni, nj, _ = dimsize
    i, j, k = cart_index
    return (k-1)*ni*nj + (j-1)*ni + i
end
function linear_index(ni::Int, ninj::Int, cart_index::NTuple{3, Int})
    i, j, k = cart_index
    return i + (j-1)*ni + (k-1)*ninj
end

function unit_vector(azm, elv)
    sin_azm, cos_azm = sincos(azm)
    sin_elv, cos_elv = sincos(elv)
    return cos_elv*cos_azm, cos_elv*sin_azm, sin_elv
end
function unit_vector(azm::T, elv::T) where {T<:AbstractArray}
    ux, uy, uz = zeros(size(azm)), zeros(size(azm)), zeros(size(azm))
    [(ux[i], uy[i], uz[i]) = unit_vector(azm[i], elv[i]) for i in eachindex(azm)]
    return ux, uy, uz
end

function write_vert_weights_to_vts(output_file, xg, yg, zg, vert_weights::Array)
    vtk_grid(output_file, xg, yg, zg) do vtk
        vtk["weights"] = vert_weights
    end
    return nothing
end
function write_vert_weights_to_vts(output_file, xg, yg, zg, vert_weights)
    flds = fieldnames(typeof(vert_weights))
    vtk_grid(output_file, xg, yg, zg) do vtk
        for f_i in flds
            vtk[string(f_i)] = getfield(vert_weights, f_i)
        end
    end
    return nothing
end


##################
### BENCHMARKS ###
##################

# Analytic solution for travel-times in a 3D linear gradient velocity model
function traveltimes_linear_gradient(x_1, x_2, x_0, v_0, g)
    dxdx = sum((x_2 .- x_1).^2) # Distance-squared between point 1 and 2
    if all(g .== 0.0)
        tt = sqrt(dxdx)/v_0
    else
        norm_g = sqrt(sum(g.^2)) # Norm of velocity gradient
        u_1 = 1.0/(v_0 + sum(g.*(x_1 .- x_0))) # Slowness at start coordinate
        u_2 = 1.0/(v_0 + sum(g.*(x_2 .- x_0))) # Slowness at end coordinate
        tt = (1.0/norm_g)*acosh(1.0 + 0.5*u_1*u_2*(norm_g^2)*dxdx)
    end
    return tt
end
function traveltimes_linear_gradient(G::StructuredGraph3D, x_1, x_0, v_0, g)
    travel_times = zeros(G.nx, G.ny, G.nz)
    for n in 1:G.num_vertices
        x_2 = G.x[n], G.y[n], G.z[n]
        travel_times[n] = traveltimes_linear_gradient(x_1, x_2, x_0, v_0, g)
    end

    return travel_times
end

# Dijkstra execution time (via @time)
# Average of 5 runs using default parameters after compilation
# 10.1115 s; 5 allocations: 3.612 MiB
# -> Min., Max., Mean Error = -2.0e-15, 19.3, 3.8 ms
# Original PSI_S implementation (SeismicDijkstra is ~27% faster)
# 14.0903 s: 82 allocations: 44.170 MiB
# -> Min., Max., Mean Error = -2.6e-12, 19.3, 3.8 ms
function benchmark_structured_linear(; p_xyz = [0.0, 0.0, 0.0], min_xyz = (-5.0, -5.0, -10.0), max_xyz = (5.0, 5.0, 0.0),
    n_xyz = (101, 101, 101), x_0 = [0.0, 0.0, 0.0], v_0 = 5.0, velocity_gradient = (0.0, 0.0, 0.0), grid_noise = 0.0,
    r_neighbours = 5, leafsize = 10, length_0 = 0.0, pred = 0)

    # Coordinate vectors
    xvec = LinRange(min_xyz[1], max_xyz[1], n_xyz[1])
    yvec = LinRange(min_xyz[2], max_xyz[2], n_xyz[2])
    zvec = LinRange(min_xyz[3], max_xyz[3], n_xyz[3])
    x_inc, y_inc, z_inc = step(xvec), step(yvec), step(zvec)

    # Define coordinate arrays and vertex weights
    gx, gy, gz = velocity_gradient
    x_ref, y_ref, z_ref = x_0
    num_vertices = prod(n_xyz)
    vert_weights = zeros(num_vertices)
    xcoords, ycoords, zcoords = zeros(num_vertices), zeros(num_vertices), zeros(num_vertices)
    n = 0 # Index counter
    for zk in zvec
        for yj in yvec
            for xi in xvec
                n += 1
                # Define grid w/ noise
                ddx, ddy, ddz = 2.0*rand() - 1.0, 2.0*rand() - 1.0, 2.0*rand() - 1.0
                ddx, ddy, ddz = grid_noise*x_inc*ddx, grid_noise*y_inc*ddy, grid_noise*z_inc*ddz
                x_ijk, y_ijk, z_ijk = xi + ddx, yj + ddy, zk + ddz
                # Compute velocity
                dx, dy, dz = x_ijk - x_ref, y_ijk - y_ref, z_ijk - z_ref
                v_ijk = v_0 + dx*gx + dy*gy + dz*gz
                # Fill graph
                xcoords[n], ycoords[n], zcoords[n] = x_ijk, y_ijk, z_ijk
                vert_weights[n] = 1.0/v_ijk
            end
        end
    end
    # Build graph
    G = StructuredGraph3D(xcoords, ycoords, zcoords, n_xyz, vert_weights;
    r_neighbours = r_neighbours, leafsize = leafsize)

    # Initialise Dijkstra
    D = initialize_dijkstra(G, p_xyz; length_0 = length_0, pred = pred)

    # Call Dijkstra
    @time dijkstra!(D, G)
    tt_dijkstra = reshape(D.lengths, n_xyz)

    # Compute true velocity field
    tt_true = traveltimes_linear_gradient(G, p_xyz, x_0, v_0, velocity_gradient)
    tt_true = reshape(tt_true, n_xyz)

    return tt_dijkstra, tt_true, G, D
end
function benchmark_structured_linear!(G::StructuredGraph3D, p_xyz; length_0 = 0.0, pred = 0)
    @warn "This may fail if grid was converted to cartesian from geographic..."
    # Compute linear gradient velocity model based on current graph weights
    nx, ny, nz = size(G)
    i_0, j_0, k_0 = 1 + round(Int, 0.5*nx), 1 + round(Int, 0.5*ny), 1 + round(Int, 0.5*nz)
    # Origin
    p_0 = linear_index(G, (i_0,j_0,k_0))
    x_0, v_0 = (G.x[p_0], G.y[p_0], G.z[p_0]), 1.0/G.vert_weights[p_0]
    # Velocity gradient in x-direction
    px_1, px_2 = linear_index(G, (1,j_0,k_0)), linear_index(G, (nx,j_0,k_0))
    gx = ((1.0/G.vert_weights[px_2]) - (1.0/G.vert_weights[px_1]))/(G.x[px_2] - G.x[px_1])
    # Velocity gradient in y-direction
    py_1, py_2 = linear_index(G, (i_0,1,k_0)), linear_index(G, (i_0,ny,k_0))
    gy = ((1.0/G.vert_weights[py_2]) - (1.0/G.vert_weights[py_1]))/(G.y[py_2] - G.y[py_1])
    # Velocity gradient in z-direction
    pz_1, pz_2 = linear_index(G, (i_0,j_0,1)), linear_index(G, (i_0,j_0,nz))
    gz = ((1.0/G.vert_weights[pz_2]) - (1.0/G.vert_weights[pz_1]))/(G.z[pz_2] - G.z[pz_1])
    
    return benchmark_structured_linear!(G, p_xyz, x_0, v_0, (gx, gy, gz); length_0 = 0.0, pred = 0)
end
function benchmark_structured_linear!(G::StructuredGraph3D, p_xyz, x_0, v_0, velocity_gradient; length_0 = 0.0, pred = 0)
    # Update vertex weights with linear gradient velocity model
    x_ref, y_ref, z_ref = x_0
    gx, gy, gz = velocity_gradient
    for n in 1:G.num_vertices
        x_ijk, y_ijk, z_ijk = G.x[n], G.y[n], G.z[n]
        dx, dy, dz = x_ijk - x_ref, y_ijk - y_ref, z_ijk - z_ref
        v_ijk = v_0 + dx*gx + dy*gy + dz*gz
        v_ijk <= 0.0 && @error "Negative or null velocities!"
        G.vert_weights[n] = 1.0/v_ijk
    end

    # Initialise Dijkstra
    D = initialize_dijkstra(G, p_xyz; length_0 = length_0, pred = pred)

    # Call Dijkstra
    @time dijkstra!(D, G)
    tt_dijkstra = reshape(D.lengths, size(G))

    # Compute true velocity field
    tt_true = traveltimes_linear_gradient(G, p_xyz, x_0, v_0, velocity_gradient)
    tt_true = reshape(tt_true, size(G))

    return tt_dijkstra, tt_true, G, D
end


#################
# DEVELOPMENTAL #
#################

# Is line integration worth it? Or is reducing grid spacing more efficient?
# Was lazy and generated the bresenham3d algorithm via ChatGPT o4
"""
    bresenham3d(p0::NTuple{3,Int}, p1::NTuple{3,Int}) -> Vector{NTuple{3,Int}}

Compute all integer lattice points approximating the line segment
from `p0 = (x0,y0,z0)` to `p1 = (x1,y1,z1)` using Bresenham’s 3D algorithm.
"""
function bresenham3d(p0::NTuple{3,Int}, p1::NTuple{3,Int})
    x0, y0, z0 = p0
    x1, y1, z1 = p1

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)

    xs = x1 > x0 ? 1 : -1
    ys = y1 > y0 ? 1 : -1
    zs = z1 > z0 ? 1 : -1

    points = Vector{NTuple{3,Int}}()

    if dx >= dy && dx >= dz
        # x is driving axis
        err_y = 2*dy - dx
        err_z = 2*dz - dx
        x, y, z = x0, y0, z0
        for i in 0:dx
            push!(points, (x, y, z))
            if err_y ≥ 0
                y += ys
                err_y -= 2dx
            end
            if err_z ≥ 0
                z += zs
                err_z -= 2dx
            end
            err_y += 2dy
            err_z += 2dz
            x += xs
        end

    elseif dy >= dx && dy >= dz
        # y is driving axis
        err_x = 2*dx - dy
        err_z = 2*dz - dy
        x, y, z = x0, y0, z0
        for i in 0:dy
            push!(points, (x, y, z))
            if err_x ≥ 0
                x += xs
                err_x -= 2dy
            end
            if err_z ≥ 0
                z += zs
                err_z -= 2dy
            end
            err_x += 2dx
            err_z += 2dz
            y += ys
        end

    else
        # z is driving axis
        err_x = 2*dx - dz
        err_y = 2*dy - dz
        x, y, z = x0, y0, z0
        for i in 0:dz
            push!(points, (x, y, z))
            if err_x ≥ 0
                x += xs
                err_x -= 2dz
            end
            if err_y ≥ 0
                y += ys
                err_y -= 2dz
            end
            err_x += 2dx
            err_y += 2dy
            z += zs
        end
    end

    return points
end

function line_integral_indices(forward_start::NTuple{3, UnitRange})
    ri, rj, rk = forward_start[1], forward_start[2], forward_start[3]
    ni, nj, nk = length(ri), length(rj), length(rk)
    indices = Array{Vector{NTuple{3, Int}}}(undef, ni, nj, nk)
    npts = 0
    @inbounds for (k, dk) in enumerate(rk)
        @inbounds for (j, dj) in enumerate(rj)
            @inbounds for (i, di) in enumerate(ri)
                indices[i,j,k] = bresenham3d((0,0,0), (di, dj, dk))
                npts += length(indices[i,j,k])
            end
        end
    end

    return indices, npts
end

end # module SeismicDijkstra
