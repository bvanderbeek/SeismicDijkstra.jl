
# Read simple event/station delimted file (id, latitude, longitude, elevation/depth)
# ASSUMES RADIUS; DECREASING!!!
function load_velocity_1D_file(velocity_file; order = (1,2), dlm = isspace)
    rad, vel = Vector{Float64}(), Vector{Float64}()
    j_rad, j_vel = order # Column indexing order

    # Loop over lines in data file
    nline, num_col = 0, maximum(order)
    for line in readlines(velocity_file)
        nline += 1
        line = split(line, dlm; keepempty = false)
        if length(line) >= num_col
            push!(rad, parse(Float64, line[j_rad]))
            push!(vel, parse(Float64, line[j_vel]))
        else
            @warn "Skipping line " * string(nline) * ". Expected " * string(num_col) * " lines but found " * string(length(line)) * "!"
        end
    end

    return (r = rad, v = vel) # Returns NamedTuple
end

# Read simple event/station delimted file (id, latitude, longitude, elevation/depth)
function load_aquisition_file(aquisition_file; id_type = Int, order = (1,2,3,4), tf_depth = false, dlm = isspace)
    tf_string_id = id_type == String # Treat IDs as Strings?
    position = Dict{id_type, Int}() # Dictionary to index of specific ID
    id, lat, lon, elv = Vector{id_type}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
    j_id, j_lat, j_lon, j_elv = order # Column indexing order

    # Loop over lines in data file
    k, nline, num_col = 0, 0, maximum(order)
    for line in readlines(aquisition_file)
        nline += 1
        line = split(line, dlm; keepempty = false)
        if length(line) >= num_col
            k += 1
            id_k = tf_string_id ? string(strip(line[j_id])) : parse(id_type, line[j_id])
            position[id_k] = k
            push!(id, id_k)
            push!(lat, parse(Float64, line[j_lat]))
            push!(lon, parse(Float64, line[j_lon]))
            push!(elv, parse(Float64, line[j_elv]))
        else
            @warn "Skipping line " * string(nline) * ". Expected " * string(num_col) * " lines but found " * string(length(line)) * "!"
        end
    end
    if tf_depth
        elv .*= -1.0
    end

    return (position = position, id = id, latitude = lat, longitude = lon, elevation = elv) # Returns NamedTuple
end

# Read simple observation delimited file (event id, station id, observation, phase)
function load_observation_file(observation_file; eid_type = Int, sid_type = String, order = (1,2,3,4), dlm = isspace)
    tf_string_event_id = eid_type == String # Treat event IDs as Strings?
    tf_string_station_id = sid_type == String # Treat station IDs as Strings?
    evt_id, sta_id, b, phs = Vector{eid_type}(), Vector{sid_type}(), Vector{Float64}(), Vector{String}()
    j_evt, j_sta, j_obs, j_phs = order

    # Loop over lines in data file
    k, nline, num_col = 0, 0, maximum(order)
    for line in readlines(observation_file)
        nline += 1
        line = split(line, dlm; keepempty = false)
        if length(line) >= num_col
            k += 1
            eid_k = tf_string_event_id ? string(strip(line[j_evt])) : parse(eid_type, line[j_evt])
            sid_k = tf_string_station_id ? string(strip(line[j_sta])) : parse(sid_type, line[j_sta])
            push!(evt_id, eid_k)
            push!(sta_id, sid_k)
            push!(b, parse(Float64, line[j_obs]))
            push!(phs, string(strip(line[j_phs])))
        else
            @warn "Skipping line " * string(nline) * ". Expected " * string(num_col) * " lines but found " * string(length(line)) * "!"
        end
    end

    return (event_id = evt_id, station_id = sta_id, observation = b, phase = phs) # Returns NamedTuple
end

# Dictionary of initialisation indices (keys) and the observation and destination indices (values)
# This could be generalized better to also create start/end point lists
function build_itinerary(Paths, Sources, Receivers)
    
    src_init = Dict{Int, Vector{NTuple{2, Int}}}()
    [src_init[Sources.position[evt_i]] = Vector{NTuple{2, Int}}() for evt_i in Paths.event_id]
    for (k, sta_j) in enumerate(Paths.station_id)
        evt_i = Paths.event_id[k]
        push!(src_init[Sources.position[evt_i]], (k, Receivers.position[sta_j]))
    end

    rcv_init = Dict{Int, Vector{NTuple{2, Int}}}()
    [rcv_init[Receivers.position[sta_j]] = Vector{NTuple{2, Int}}() for sta_j in Paths.station_id]
    for (k, evt_i) in enumerate(Paths.event_id)
        sta_j = Paths.station_id[k]
        push!(rcv_init[Receivers.position[sta_j]], (k, Sources.position[evt_i]))
    end

    itinerary, tf_reverse = length(rcv_init) < length(src_init) ? (rcv_init, true) : (src_init, false)

    return itinerary, tf_reverse
end

# Consider wrapping this into build_itinerary.
# Need to consider coordinate systems.
function collect_start_end_points(Data, Events, Stations; R_earth = 6371.0)
    # Build shortest path itinerary
    itinerary, tf_reverse = SeismicDijkstra.build_itinerary(Data, Events, Stations)
    # Collect source coordinates
    src_points = Vector{NTuple{3, Float64}}(undef, length(Events.id))
    for i in eachindex(Events.id)
        lon_i, lat_i, elv_i = Events.longitude[i], Events.latitude[i], Events.elevation[i]
        src_points[i] = SeismicDijkstra.global_cartesian_coordinates(lon_i, lat_i, elv_i; R_earth = R_earth)
    end
    # Collect the receiver coordinates
    rcv_points = Vector{NTuple{3, Float64}}(undef, length(Stations.id))
    for j in eachindex(Stations.id)
        lon_j, lat_j, elv_j = Stations.longitude[j], Stations.latitude[j], Stations.elevation[j]
        rcv_points[j] = SeismicDijkstra.global_cartesian_coordinates(lon_j, lat_j, elv_j; R_earth = R_earth)
    end
    # Select fewest points as the initialization points for Dijkstra
    start_points, end_points = tf_reverse ? (rcv_points, src_points) : (src_points, rcv_points)

    return itinerary, start_points, end_points, tf_reverse
end

# Example function for computing times with optional re-integration of ray path.
# Needs to be generalized. Assumes isotropic graph, 
function compute_times(G, Data, Events, Stations; phase = UnspecifiedPhase(),
    dl = 0.0, min_vert = 2, length_0 = 0.0, pred = 0, R_earth = 6371.0)

    # Build itinerary of paths that need to be computed
    itinerary, start_points, end_points, tf_reverse = collect_start_end_points(Data, Events, Stations; R_earth = R_earth)
    
    # Travel-time predictions (includes optional re-integration along path)
    println("Starting shortest path calculations")
    tf_reverse ? println("stations -> events...") : println("events -> stations...")

    num_paths = sum(length, values(itinerary))
    ttimes, dist = zeros(num_paths), zeros(num_paths)
    n, num_src = 0, length(itinerary)
    for (i_start, destinations) in itinerary
        x_start = SVector(start_points[i_start])
        D = initialize_dijkstra(G, x_start; phase = phase, length_0 = length_0, pred = pred)
        @time dijkstra!(D, G; phase = phase)

        for (k_pth, j_end) in destinations
            x_end = SVector(end_points[j_end])
            t_min, xyz_path, _ = get_path(D, G, x_end; phase = phase, length_0 = length_0)
            dist[k_pth] = return_path_length(xyz_path; path_length = 0.0)
            if dl > 0.0
                t_min, _ = refine_path(G, xyz_path, dl; phase=phase, dist=length_0, min_vert=min_vert)
            end
            ttimes[k_pth] = t_min
        end
        n += 1
        println("Finished initialisation point " * string(n) * " of " * string(num_src) * ".")
    end
    
    return ttimes, dist
end
function shortest_paths(G, itinerary, start_points, end_points;
    phase = UnspecifiedPhase(), length_0 = 0.0, pred = 0, dl = 0.0, min_vert = 2)

    num_paths = sum(length, values(itinerary))
    path_lengths, total_distance = zeros(num_paths), zeros(num_paths)
    n, num_src = 0, length(itinerary)
    for (i_start, destinations) in itinerary
        x_start = SVector(start_points[i_start])
        D = initialize_dijkstra(G, x_start; phase = phase, length_0 = length_0, pred = pred)
        @time dijkstra!(D, G; phase = phase)

        for (k_pth, j_end) in destinations
            x_end = SVector(end_points[j_end])
            t_min, xyz_path, _ = get_path(D, G, x_end; phase = phase, length_0 = length_0)
            total_distance[k_pth] = return_path_length(xyz_path; path_length = 0.0)
            if dl > 0.0
                t_min, xyz_path = refine_path(G, xyz_path, dl; phase=phase, dist=length_0, min_vert=min_vert)
            end
            path_lengths[k_pth] = t_min
        end

        n += 1
        println("Finished initialisation point " * string(n) * " of " * string(num_src) * ".")
    end

    return path_lengths, total_distance
end
function return_path_length(xyz_path; path_length = 0.0)
    num_seg = size(xyz_path, 2) - 1
    for k in 1:num_seg
        dx, dy, dz = xyz_path[1,k+1] - xyz_path[1,k], xyz_path[2,k+1] - xyz_path[2,k], xyz_path[3,k+1] - xyz_path[3,k]
        path_length += sqrt(dx^2 + dy^2 + dz^2)
    end
    return path_length
end

# Requires start_point to be a vector and end_points to be a vector of vectors
function shortest_paths(G, start_point, end_points; phase = UnspecifiedPhase(), length_0 = 0.0, pred = 0)
    D = initialize_dijkstra(G, start_point; phase = phase, length_0 = length_0, pred = pred)
    @time dijkstra!(D, G; phase = phase)

    num_paths = length(end_points)
    path_lengths = Vector{Float64}(undef, num_paths)
    path_coordinates = Vector{Array{Float64,2}}(undef, num_paths)
    path_vertices = Vector{Int}(undef, num_paths)
    for (i, x_end) in enumerate(end_points)
        path_lengths[i], path_coordinates[i], path_vertices[i] = get_path(D, G, x_end; phase = phase, length_0 = length_0)
    end

    return path_lengths, path_coordinates, path_vertices
end


# Convenience function to make a graph
function make_graph(lon_grid, lat_grid, elv_grid, v1D; r_neighbours = 5, leafsize = 10, grid_noise = 0.0, R_earth = 6371.0)
    # Graph coordinate vectors
    lon_vec = LinRange(lon_grid[1], lon_grid[2], lon_grid[3])
    lat_vec = LinRange(lat_grid[1], lat_grid[2], lat_grid[3])
    elv_vec = LinRange(elv_grid[1], elv_grid[2], elv_grid[3])
    # Grid spacing and noise level
    lon_inc, lat_inc, elv_inc = step(lon_vec), step(lat_vec), step(elv_vec)
    dlon, dlat, delv = grid_noise*lon_inc, grid_noise*lat_inc, grid_noise*elv_inc

    # Interpolation structure (convert radius to depth to satisfy linear_interpolation increasing order assumption)
    vel_interp = linear_interpolation(R_earth .- v1D.r, v1D.v, extrapolation_bc=Flat())

    # Fill graph
    num_vertices = lon_grid[3] * lat_grid[3] * elv_grid[3]
    vert_weights = zeros(num_vertices)
    xcoords, ycoords, zcoords = zeros(num_vertices), zeros(num_vertices), zeros(num_vertices)
    n = 0
    for elv_k in elv_vec
        for lat_j in lat_vec
            for lon_i in lon_vec
                n += 1
                # Define grid noise
                ddlon, ddlat, ddelv = (2.0 * rand() - 1.0) * dlon, (2.0 * rand() - 1.0) * dlat, (2.0 * rand() - 1.0) * delv
                # Compute noisey cartesian coordinates
                lon_ijk, lat_ijk, elv_ijk = lon_i + ddlon, lat_j + ddlat, elv_k + ddelv
                x_ijk, y_ijk, z_ijk = global_cartesian_coordinates(lon_ijk, lat_ijk, elv_ijk; R_earth = R_earth)
                # Compute slowness
                v_ijk = vel_interp(-elv_ijk) # Velocity interpolation function uses depth!
                # Fill graph
                xcoords[n], ycoords[n], zcoords[n], vert_weights[n] = x_ijk, y_ijk, z_ijk, 1.0/v_ijk
            end
        end
    end
    # Build graph
    G = StructuredGraph3D(xcoords, ycoords, zcoords, (lon_grid[3], lat_grid[3], elv_grid[3]), vert_weights;
        r_neighbours=r_neighbours, leafsize=leafsize)
    return G
end

# Compute travel-times for every observation
function travel_times(G, Events, Stations, Data; R_earth = 6371.0)
    unique_station_ids = unique(Data.station_id)
    snum_station = string(length(unique_station_ids)) # For display purposes
    tt_predicted, npt = zeros(length(Data.observation)), 0
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
                x_evt, y_evt, z_evt = global_cartesian_coordinates(lon_evt, lat_evt, elv_evt; R_earth = R_earth)
                t_min, _ = get_nearest_connection(D, G, [x_evt, y_evt, z_evt])
                tt_predicted[k] = t_min
            end
        end
        println("Finished point " * string(npt) * " of "*snum_station*".")
    end
    return tt_predicted
end

# Global cartesian coordinates -- deprecate to use cartesian_to_spherical
function global_cartesian_coordinates(lon, lat, elv; R_earth = R_earth)
    sinλ, cosλ = sincosd(lon)
    sinϕ, cosϕ = sincosd(lat)
    r = R_earth + elv
    return r*cosϕ*cosλ, r*cosϕ*sinλ, r*sinϕ
end

function cartesian_to_spherical(x, y, z)
    azm = atan(y,x)
    elv = atan(z,sqrt(x^2 + y^2))
    r = sqrt(x^2 + y^2 + z^2)
    return azm, elv, r
end
function spherical_to_cartesian(azm, elv, r)
    sinλ, cosλ = sincos(azm)
    sinϕ, cosϕ = sincos(elv)
    return r*cosϕ*cosλ, r*cosϕ*sinλ, r*sinϕ
end