
# Load Dependencies
ENV["PSI_S"] = "/Users/bvanderbeek/research/software/GitRepos/PSI_S"
include(ENV["PSI_S"]*"/src/pre_processing.jl")
include("/Users/bvanderbeek/research/software/GitRepos/SeismicDijkstra/src/SeismicDijkstra.jl")
using DelimitedFiles
using WriteVTK

# Additional Functions
function global_cartesian_coordinates(lon, lat, elv; R_earth = R_earth)
    sinλ, cosλ = sincosd(lon)
    sinϕ, cosϕ = sincosd(lat)
    r = R_earth + elv
    return r*cosϕ*cosλ, r*cosϕ*sinλ, r*sinϕ
end
function linearly_interpolate(x, v, qx::Number; tf_extrapolate = false, tf_harmonic = false)
    # Identify minimum and maximum indices
    n = length(x)
    if x[1] < x[n]
        imin = 1
        imax = n
    else
        imin = n
        imax = 1
    end
    # Interpolate
    if (qx < x[imin])
        # Minimum bounds
        if tf_extrapolate
            qv = v[imin]
        else
            qv = NaN
        end
    elseif (qx > x[imax])
        # Maximum bounds
        if tf_extrapolate
            qv = v[imax]
        else
            qv = NaN
        end
    else
        # Inside
        j = searchsortedlast(x, qx)  # Find the index of the largest sample point less than or equal to qx[i]
        w = (qx - x[j]) / (x[j+1] - x[j])  # Compute the interpolation factor
        # Return interpolated value as a weighted arithmetic (harmonic) average
        if tf_harmonic
            qv = 1.0/(((1.0 - w)/v[j]) + (w/v[j+1]))
        else
            qv = (1.0 - w)*v[j] + w*v[j+1] 
        end
    end

    return qv
end
function make_graph(lon_grid, lat_grid, elv_grid, v1D; r_neighbours = 5, leafsize = 10, grid_noise = 0.0, R_earth = 6371.0)
    # Graph coordinate vectors
    lon_vec = LinRange(lon_grid[1], lon_grid[2], lon_grid[3])
    lat_vec = LinRange(lat_grid[1], lat_grid[2], lat_grid[3])
    elv_vec = LinRange(elv_grid[1], elv_grid[2], elv_grid[3])
    # Grid spacing and noise level
    lon_inc, lat_inc, elv_inc = step(lon_vec), step(lat_vec), step(elv_vec)
    dlon, dlat, delv = grid_noise*lon_inc, grid_noise*lat_inc, grid_noise*elv_inc

    # Fill graph
    num_vertices = lon_grid[3] * lat_grid[3] * elv_grid[3]
    vert_weights = zeros(num_vertices)
    xcoords, ycoords, zcoords = zeros(num_vertices), zeros(num_vertices), zeros(num_vertices)
    n = 0
    for (k, elv_k) in enumerate(elv_vec)
        u_k = 1.0 / v1D[k] # Get k'th slowness
        for lat_j in lat_vec
            for lon_i in lon_vec
                n += 1
                # Define grid noise
                ddlon, ddlat, ddelv = (2.0 * rand() - 1.0) * dlon, (2.0 * rand() - 1.0) * dlat, (2.0 * rand() - 1.0) * delv
                # Compute noisey cartesian coordinates
                lon_ijk, lat_ijk, elv_ijk = lon_i + ddlon, lat_j + ddlat, elv_k + ddelv
                x_ijk, y_ijk, z_ijk = global_cartesian_coordinates(lon_ijk, lat_ijk, elv_ijk; R_earth = R_earth)
                # Fill graph
                xcoords[n], ycoords[n], zcoords[n], vert_weights[n] = x_ijk, y_ijk, z_ijk, u_k
            end
        end
    end
    # Build graph
    G = StructuredGraph3D(xcoords, ycoords, zcoords, (lon_grid[3], lat_grid[3], elv_grid[3]), vert_weights;
        r_neighbours=r_neighbours, leafsize=leafsize)
    return G
end
# Run ray tracer
function run_ray_tracer(G, Events, Stations, Data; R_earth = 6371.0)
    unique_station_ids = unique(Data.station_id)
    tt_ref, npt = zeros(length(Data.observation)), 0
    for sid in unique_station_ids
        npt += 1
        # Index station
        jsta = Stations.position[sid]
        # Get coordinates
        lon_sta, lat_sta, elv_sta = Stations.longitude[jsta], Stations.latitude[jsta], Stations.elevation[jsta]
        x_sta, y_sta, z_sta = global_cartesian_coordinates(lon_sta, lat_sta, elv_sta; R_earth=R_earth)
        # Run dijkstra
        D = initialize_dijkstra(G, [x_sta, y_sta, z_sta]; length_0=0.0, pred=0)
        @time dijkstra!(D, G)

        # Travel-times for this station
        for (k, evt_k) in enumerate(Data.event_id)
            if Data.station_id[k] == sid
                ievt = Events.position[evt_k]
                lon_evt, lat_evt, elv_evt = Events.longitude[ievt], Events.latitude[ievt], Events.elevation[ievt]
                x_evt, y_evt, z_evt = global_cartesian_coordinates(lon_evt, lat_evt, elv_evt)
                t_min, _ = get_nearest_connection(D, G, [x_evt, y_evt, z_evt])

                tt_ref[k] = t_min
            end
        end
        println("Finished point " * string(npt) * " of 38.")
    end
    return tt_ref
end

# Connect to local directory
cd(@__DIR__)

# Input Data Files
dlm = isspace # What is used to deliminate columns in the below data files (isspace or ",")
event_file = "../input/evt_0N0E_SED_auto_HQ_MEDD.dat"
station_file = "../input/sta_0N0E_Station_Inventory_COSEISMIQ_ISOR_IMO_UNIQUE.dat"
data_file = "../input/uni_0N0E_tt_P_hengill_hq_hdi90_subset.dat"
j_id, j_lat, j_lon, j_elv = 1, 2, 3, 4
j_evt, j_sta, j_obs, j_phs = 1, 2, 3, 4
tf_event_depth, tf_station_depth = true, false
eid_type, sid_type = Int, Int

# Grid Parameters
lon_grid = (-0.1050, 0.1510, 143)
lat_grid = (-0.1090, 0.1280, 133)
elv_grid = (-15.0, 1.4, 83)
r_neighbours = 5
leafsize = 10
grid_noise = 0.5
R_earth = 6371.0
file_velocity_model = "../input/vel1D_hengill_grigoli_35km.dat"

# Load data into structures
Events = read_aquisition_file(event_file; id_type = eid_type, order = (j_id, j_lat, j_lon, j_elv), tf_depth = tf_event_depth, dlm = dlm)
Stations = read_aquisition_file(station_file; id_type = sid_type, order = (j_id, j_lat, j_lon, j_elv), tf_depth = tf_station_depth, dlm = dlm)
Data = read_observation_file(data_file; eid_type = eid_type, sid_type = sid_type, order = (j_evt, j_sta, j_obs, j_phs), dlm = dlm)

# Interpolate 1D model
elv_vec = LinRange(elv_grid[1], elv_grid[2], elv_grid[3])
vmodel = readdlm(file_velocity_model)
vmodel[:,1] .= R_earth .- vmodel[:,1] # DEPTH! Required for interpolation function...monotonic increasing
v1D = zeros(length(elv_vec))
[v1D[k] = linearly_interpolate(vmodel[:,1], vmodel[:,2], -qelv; tf_extrapolate = true) for (k, qelv) in enumerate(elv_vec)]

# Make Graph
G = make_graph(lon_grid, lat_grid, elv_grid, v1D;
r_neighbours = r_neighbours, leafsize = leafsize, grid_noise = grid_noise, R_earth = R_earth)

# Run ray tracer
tt_ref = run_ray_tracer(G, Events, Stations, Data; R_earth = R_earth)

# Compute delays
Delays = deepcopy(Data)
Delays.observation .-= tt_ref
rdt, npe = compute_event_demeaned_delays(Delays, Events)

# Source-receiver lengths anbd orientations
L, azm, elv = zeros(size(Data.observation)), zeros(size(Data.observation)), zeros(size(Data.observation))
for k in eachindex(Data.observation)
    evt_id, sta_id = Data.event_id[k], Data.station_id[k]
    ievt, jsta = Events.position[evt_id], Stations.position[sta_id]
    zevt, xevt, yevt = global_cartesian_coordinates(Events.longitude[ievt], Events.latitude[ievt], Events.elevation[ievt]; R_earth = R_earth)
    zsta, xsta, ysta = global_cartesian_coordinates(Stations.longitude[jsta], Stations.latitude[jsta], Stations.elevation[jsta]; R_earth = R_earth)
    dx, dy, dz = xsta - xevt, ysta - yevt, zsta - zevt

    L[k], azm[k], elv[k] = sqrt(dx^2 + dy^2 + dz^2), atand(dy,dx), atand(dz, sqrt(dx^2 + dy^2))
end

# writedlm(hcat(L, azm, elv, rdt))

# Data.observation .= tt_ref .+ rdt
# write_psi_s_observations("../input/uni_0N0E_tt_P_hengill_hq_hdi90_subset_demean.dat", Data, Events, Stations)
