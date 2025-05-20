# Run Benchmarks
using Pkg
Pkg.activate("/Users/bvanderbeek/research/software/GitRepos/SeismicDijkstra")
using SeismicDijkstra
using Plots

# Default homogeneous isotropic benchmark
tt_d, tt_0, G, D = SeismicDijkstra.benchmark_structured_linear();

# Benchmark with custom parameters
min_xyz = (-5.0, -5.0, -10.0) # Minimum cartesian grid coordinates
max_xyz = (5.0, 5.0, 0.0) # Minimum cartesian grid coordinates
num_xyz = (101, 101, 101) # Number of vertices in each direction
graph_origin = [0.0, 0.0, 0.0] # Origin of graph; used for computing velocities in linear gradient model
velocity_origin = 5.0 # Velocity at graph origin
velocity_gradient = (0.0, 0.0, 0.0) # Velocity gradient in x, y, and z directions
initialization_point = [0.0, 0.0, 0.0] # Coordinates of where we start Dijkstra algorithm
grid_noise = 0.5 # Grid noise level (should be ≥ 0 and ≤ 0.5)
r_neighbours = 5 # Forward star level (i.e. search distance)
leafsize = 10 # For KDD nearest neighbour search (just for locating points inside graph)
length_0 = 0.0 # Starting time (usually zero)
pred = 0 # Prior vertex in path (usually 0)

# Run custom benchmark
tt_d, tt_0, G, D = SeismicDijkstra.benchmark_structured_linear(; p_xyz = initialization_point, min_xyz = min_xyz, max_xyz = max_xyz,
    n_xyz = num_xyz, x_0 = graph_origin, v_0 = velocity_origin, velocity_gradient = velocity_gradient, grid_noise = grid_noise,
    r_neighbours = r_neighbours, leafsize = leafsize, length_0 = length_0, pred = pred)

# Elliptical Anisotropy Test
# Homogenous elliptical anisotropy parameters
v_0, f, azm, elv = 5.0, 0.05, deg2rad(22.0), 0.0

# Build elliptical anisotropy graph
V = SeismicDijkstra.EllipticalVelocity(v_0*ones(size(G)), f*ones(size(G)), azm*ones(size(G)), elv*ones(size(G)))
G = SeismicDijkstra.StructuredGraph3D(G.x, G.y, G.z, size(G), V; r_neighbours = r_neighbours, leafsize = leafsize)

# Run Dijkstra shortest path
D = SeismicDijkstra.initialize_dijkstra(G, initialization_point)
@time SeismicDijkstra.dijkstra!(D, G)
tt_ani = reshape(D.lengths, size(G))

# Anisotropic - Isotropic Traveltime field
dt_ani = 1000.0*(tt_ani[:,:,end] .- tt_d[:,:,end])
dt_ani = transpose(dt_ani)
heatmap(dt_ani, aspect_ratio = 1)

# Thomsen Anisotropy Test
# Homogenous elliptical anisotropy parameters
ϵ, δ, γ = -2.0*f/(1.0 + f), -2.0*f/(1.0 + f), -2.0*f/(1.0 + f)
α, β = v_0*(1.0 + f), v_0*(1.0 + f)
phase = SeismicDijkstra.BodyP()

# Build elliptical anisotropy graph
V = SeismicDijkstra.ThomsenVelocity(α*ones(size(G)), β*ones(size(G)), ϵ*ones(size(G)), δ*ones(size(G)), γ*ones(size(G)), azm*ones(size(G)), elv*ones(size(G)))
G = SeismicDijkstra.StructuredGraph3D(G.x, G.y, G.z, size(G), V; r_neighbours = r_neighbours, leafsize = leafsize)

# Run Dijkstra shortest path
D = SeismicDijkstra.initialize_dijkstra(G, initialization_point; phase = phase)
@time SeismicDijkstra.dijkstra!(D, G; phase = phase)
tt_ani = reshape(D.lengths, size(G))

# Anisotropic - Isotropic Traveltime field
dt_ani = 1000.0*(tt_ani[:,:,end] .- tt_d[:,:,end])
dt_ani = transpose(dt_ani)
heatmap(dt_ani, aspect_ratio = 1)