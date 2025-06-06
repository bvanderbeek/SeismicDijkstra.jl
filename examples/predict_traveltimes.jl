# Example: Compute travel-times from event, station, and arrival time files
using SeismicDijkstra
using Plots

#########
# INPUT #
#########

# Data Files
dlm = isspace # Data file delimiter (e.g., isspace, ",")
# Events and Stations
event_file = "data/events.dat" # Event file
station_file = "data/stations.dat" # Station file
j_id, j_lat, j_lon, j_elv = 1, 2, 3, 4 # Column numbers where ID, latitude, longitude, and elevation are found in above files
tf_event_depth, tf_station_depth = true, false # Are events/station vertical coordinates defined as depth (true) or elevation (false)
eid_type, sid_type = Int, Int # Are the event/station IDs integer- (Int) or string- (String) valued
# Arrival times
data_file = "data/traveltimes_p.dat" # Arrival time file
j_evt, j_sta, j_obs, j_phs = 1, 2, 3, 4 # Column numbers where event, station, observation, and phase are found in above files
# 1D Velocity Model
velocity_file = "data/velocity_1D.dat" # 1D velocity model file
j_rad, j_vel = 1, 2 # Column numbers where radial coordinate and velocity are stored

# Grid Parameters
lon_grid = (-0.1050, 0.1510, 143) # (min. longitude, max. longitude, number vertices in longitude)
lat_grid = (-0.1090, 0.1280, 133) # (min. latitude, max. latitude, number vertices in latitude)
elv_grid = (-15.0, 1.4, 83) # (min. elevation, max. elevation, number of vertices in elevation)
r_neighbours = 5 # Forward star level (i.e. search distance for connections)
leafsize = 10 # Leaf size for kdtree -- used for nearest neighbour interpolations (inconsequential to Dijkstra results)
grid_noise = 0.5 # Fractional grid noise level; should be ≥ 0 and ≤ 0.5
R_earth = 6371.0 # Reference spherical earth radius

# Load data into structures
Events = read_aquisition_file(event_file; id_type = eid_type, order = (j_id, j_lat, j_lon, j_elv), tf_depth = tf_event_depth, dlm = dlm)
Stations = read_aquisition_file(station_file; id_type = sid_type, order = (j_id, j_lat, j_lon, j_elv), tf_depth = tf_station_depth, dlm = dlm)
Data = read_observation_file(data_file; eid_type = eid_type, sid_type = sid_type, order = (j_evt, j_sta, j_obs, j_phs), dlm = dlm)
v1D = read_velocity_1D_file(velocity_file; order = (j_rad, j_vel), dlm = dlm)

# Make Graph
G = make_graph(lon_grid, lat_grid, elv_grid, v1D;
r_neighbours = r_neighbours, leafsize = leafsize, grid_noise = grid_noise, R_earth = R_earth)

# Run ray tracer
tt_predicted = travel_times(G, Events, Stations, Data; R_earth = R_earth)

# Plot residuals
histogram(Data.observation .- tt_predicted)