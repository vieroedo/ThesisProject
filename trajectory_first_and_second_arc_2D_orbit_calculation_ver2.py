from handle_functions import *

# known issues:
# Flyby is instantaneous but the change in position assumes it to be continuous in time
# -> need for a simpler version i guess

skip_regula_falsi = False
choose_sigma_angle_then = 156.15  # deg

choose_sigma_angle_then = choose_sigma_angle_then * np.pi/180

# Set flyby epoch -> edit this to be initial epoch!!
first_arc_arrival_epoch_days = 11293.  # days

# PROBLEM PARAMETERS ###################################################################################################
# (Consider moving to related script in case other ones using this library need different parameters)

# Atmospheric entry conditions
arrival_pericenter_altitude = 2000e3  # m (DO NOT CHANGE - consider changing only with valid and sound reasons)
flight_path_angle_at_atmosphere_entry = -3.  # degrees

# Jupiter arrival conditions
interplanetary_arrival_velocity_in_jupiter_frame = 5600 # m/s
delta_angle_from_hohmann_trajectory = 0.98  # degrees

# Trajectory geometry
choose_flyby_moon = 'Callisto'
safety_altitude_flyby = 0.

# Plotting parameters
number_of_epochs_to_plot = 200

########################################################################################################################


# Parameters reprocessing ##############################################################################################
flight_path_angle_at_atmosphere_entry = flight_path_angle_at_atmosphere_entry * np.pi / 180  # rad
delta_angle_from_hohmann_trajectory = delta_angle_from_hohmann_trajectory * np.pi / 180  # rad
first_arc_number_of_points = number_of_epochs_to_plot
second_arc_number_of_points = number_of_epochs_to_plot
flyby_number_of_points = number_of_epochs_to_plot
########################################################################################################################


# Retrieve moon state at flyby epoch
moon_epoch_of_flyby = first_arc_arrival_epoch_days * constants.JULIAN_DAY  # s
moon_flyby_state = spice_interface.get_body_cartesian_state_at_epoch(
    target_body_name=choose_flyby_moon,
    observer_body_name="Jupiter",
    reference_frame_name=global_frame_orientation,
    aberration_corrections="NONE",
    ephemeris_time=moon_epoch_of_flyby)

# ASSUMPTION: Trajectory lies on the moon orbital plane

# Calculate first arc orbital plane based on the one of the moon
first_arc_angular_momentum_cap = unit_vector(np.cross(moon_flyby_state[0:3], moon_flyby_state[3:6]))
first_arc_line_of_nodes_cap = np.cross(z_axis, first_arc_angular_momentum_cap)

# First arc initial known conditions
first_arc_departure_radius = jupiter_SOI_radius
first_arc_departure_velocity = interplanetary_arrival_velocity_in_jupiter_frame
radius_velocity_vectors_angle = np.pi - delta_angle_from_hohmann_trajectory

# Calculate first arc orbital energy
first_arc_orbital_energy = first_arc_departure_velocity ** 2 / 2 - central_body_gravitational_parameter / first_arc_departure_radius

# Calculate angular momentum magnitude and build the vector
first_arc_angular_momentum = first_arc_departure_radius * first_arc_departure_velocity * np.sin(radius_velocity_vectors_angle)
first_arc_angular_momentum_vector = first_arc_angular_momentum * first_arc_angular_momentum_cap

# Calculate p, sma, and e of the first arc
first_arc_semilatus_rectum = first_arc_angular_momentum ** 2 / central_body_gravitational_parameter
first_arc_semimajor_axis = - central_body_gravitational_parameter / (2 * first_arc_orbital_energy)
first_arc_eccentricity = np.sqrt(1 - first_arc_semilatus_rectum/ first_arc_semimajor_axis)


########################################################################################################################
# START OF ROOT FINDING FOR SIGMA ANGLE ################################################################################
########################################################################################################################

# Retrieve flyby moon data: Radius, mu, velocity vector
moon_radius = galilean_moons_data[choose_flyby_moon]['Radius']
moon_SOI_radius = galilean_moons_data[choose_flyby_moon]['SOI_Radius']
mu_moon = galilean_moons_data[choose_flyby_moon]['mu']
moon_velocity = moon_flyby_state[3:6]

# Set the arrival radius to be at the edge of Jupiter's atmosphere
second_arc_arrival_radius = jupiter_radius + arrival_pericenter_altitude

# Sub-problem free parameter: sigma_angle

# REGULA FALSI #########################################################################################################

if not skip_regula_falsi:
    interval_left_boundary_a = 0.
    interval_right_boundary_b = np.pi

    fpa_a = calculate_fpa_from_flyby_geometry(sigma_angle=interval_left_boundary_a,
                                              arc_1_initial_velocity=first_arc_departure_velocity,
                                              arc_1_initial_radius=first_arc_departure_radius,
                                              delta_hoh=delta_angle_from_hohmann_trajectory,
                                              arc_2_final_radius=second_arc_arrival_radius,
                                              mu_moon=mu_moon,
                                              moon_SOI=moon_SOI_radius,
                                              moon_state_at_flyby=moon_flyby_state)

    f_a = fpa_a - flight_path_angle_at_atmosphere_entry

    fpa_b = calculate_fpa_from_flyby_geometry(sigma_angle=interval_right_boundary_b,
                                              arc_1_initial_velocity=first_arc_departure_velocity,
                                              arc_1_initial_radius=first_arc_departure_radius,
                                              delta_hoh=delta_angle_from_hohmann_trajectory,
                                              arc_2_final_radius=second_arc_arrival_radius,
                                              mu_moon=mu_moon,
                                              moon_SOI=moon_SOI_radius,
                                              moon_state_at_flyby=moon_flyby_state)

    f_b = fpa_b - flight_path_angle_at_atmosphere_entry

    if f_a * f_b > 0:
        raise Exception('Interval has either zero or multiple zeroes')

    a_int = interval_left_boundary_a
    b_int = interval_right_boundary_b

    tolerance = 1e-5
    max_iter = 200
    i = 0
    for i in range(max_iter):
        c_point = (a_int * f_b - b_int * f_a) / (f_b - f_a)

        fpa_c = calculate_fpa_from_flyby_geometry(sigma_angle=c_point,
                                                  arc_1_initial_velocity=first_arc_departure_velocity,
                                                  arc_1_initial_radius=first_arc_departure_radius,
                                                  delta_hoh=delta_angle_from_hohmann_trajectory,
                                                  arc_2_final_radius=second_arc_arrival_radius,
                                                  mu_moon=mu_moon,
                                                  moon_SOI=moon_SOI_radius,
                                                  moon_state_at_flyby=moon_flyby_state)
        f_c = fpa_c - flight_path_angle_at_atmosphere_entry

        if abs(f_c) < tolerance:
            # Root found
            break

        if f_a * f_c > 0:
            a_int = c_point
            f_a = f_c
        else:
            b_int = c_point
            f_b = f_c


    # Found root
    sigma_angle = c_point
    calculated_fpa = f_c + flight_path_angle_at_atmosphere_entry


    # Debugging
    print(f'Number of iterations: {i}')
    # print(f'Flyby pericenter altitude: {(flyby_pericenter-moon_radius)/1e3} km')
    print(f'Sigma angle used: {sigma_angle*180/np.pi} deg')
    print(f'f.p.a. result of root finder: {calculated_fpa*180/np.pi} deg')
else:
    sigma_angle = choose_sigma_angle_then
    calculated_fpa = -10
########################################################################################################################


########################################################################################################################
# END OF ROOT FINDING FOR SIGMA ANGLE ##################################################################################
########################################################################################################################


# CALCULATE FLYBY WITH FOUND PERICENTER RADIUS #########################################################################
moon_position = moon_flyby_state[0:3]

orbit_axis = unit_vector(np.cross(moon_position, moon_velocity))
flyby_initial_position = rotate_vectors_by_given_matrix(rotation_matrix(orbit_axis, -sigma_angle), unit_vector(moon_velocity)) * moon_SOI_radius

# Energy and angular momentum for first arc are calculated above

first_arc_arrival_position = moon_position + flyby_initial_position
first_arc_arrival_radius = LA.norm(first_arc_arrival_position)
first_arc_arrival_velocity = np.sqrt(2 * (first_arc_orbital_energy + central_body_gravitational_parameter/first_arc_arrival_radius))

first_arc_arrival_fpa = - np.arccos(first_arc_angular_momentum/(first_arc_arrival_radius*first_arc_arrival_velocity))
first_arc_arrival_velocity_vector = rotate_vectors_by_given_matrix(rotation_matrix(orbit_axis, np.pi/2 - first_arc_arrival_fpa), unit_vector(first_arc_arrival_position)) * first_arc_arrival_velocity

flyby_initial_velocity_vector = first_arc_arrival_velocity_vector - moon_velocity
flyby_v_inf_t = LA.norm(flyby_initial_velocity_vector)

flyby_axis = unit_vector(np.cross(flyby_initial_position, flyby_initial_velocity_vector))


phi_2_angle = np.arccos(np.dot(unit_vector(-moon_velocity),unit_vector(flyby_initial_velocity_vector)))
if np.dot(np.cross(-moon_velocity, flyby_initial_velocity_vector), flyby_axis) < 0:
    phi_2_angle = - phi_2_angle + 2 * np.pi

delta_angle = np.arccos(np.dot(unit_vector(-moon_velocity), unit_vector(flyby_initial_position)))
if np.dot(np.cross(-moon_velocity, flyby_initial_position), flyby_axis) < 0:
    delta_angle = 2 * np.pi - delta_angle

B_parameter = moon_SOI_radius * np.sin(phi_2_angle - delta_angle)
flyby_alpha_angle = 2 * np.arcsin(1 / np.sqrt(1 + (B_parameter ** 2 * flyby_v_inf_t**4 )/ mu_moon ** 2))
beta_angle = phi_2_angle + flyby_alpha_angle / 2 - np.pi / 2

position_rot_angle = 2 * (- delta_angle + beta_angle)

flyby_final_position = rotate_vectors_by_given_matrix(rotation_matrix(flyby_axis,position_rot_angle), flyby_initial_position)

flyby_final_velocity_vector = rotate_vectors_by_given_matrix(rotation_matrix(flyby_axis, flyby_alpha_angle), flyby_initial_velocity_vector)

second_arc_departure_position = moon_position + flyby_final_position
second_arc_departure_velocity_vector = moon_velocity + flyby_final_velocity_vector

second_arc_departure_radius = LA.norm(second_arc_departure_position)
second_arc_departure_velocity = LA.norm(second_arc_departure_velocity_vector)

second_arc_angular_momentum_vector = np.cross(second_arc_departure_position, second_arc_departure_velocity_vector)
second_arc_angular_momentum = LA.norm(second_arc_angular_momentum_vector)
second_arc_orbital_energy = second_arc_departure_velocity**2/2 - central_body_gravitational_parameter / second_arc_departure_radius

second_arc_arrival_velocity = np.sqrt(2 * (second_arc_orbital_energy + central_body_gravitational_parameter / second_arc_arrival_radius))

second_arc_arrival_fpa = - np.arccos(np.clip(second_arc_angular_momentum/(second_arc_arrival_radius * second_arc_arrival_velocity), -1, 1))


root_argument = 1 + (B_parameter**2 * flyby_v_inf_t**4)/(mu_moon**2)
flyby_pericenter = mu_moon/(flyby_v_inf_t**2) * (np.sqrt(root_argument) - 1)
flyby_altitude = flyby_pericenter - moon_radius

flyby_delta_v = LA.norm(flyby_final_velocity_vector-flyby_initial_velocity_vector)

alpha_max = 2 * np.arcsin(1/(1+flyby_v_inf_t**2/(mu_moon/moon_radius)))
flyby_delta_v_max = 2 * flyby_v_inf_t * np.sin(alpha_max/2)

# Checks if the flyby pericenter is above minimum safety altitude set at the beginning
print(f'\nFlyby altitude: {flyby_altitude/1e3:.3f} km')
print(f'Flyby alpha angle: {flyby_alpha_angle*180/np.pi:.3f} deg')
print(f'Flyby delta_v: {flyby_delta_v/1e3:.3f} km/s')
print(f'Max delta_v achievable for {choose_flyby_moon}: {flyby_delta_v_max/1e3:.3f} km/s')
if flyby_altitude < safety_altitude_flyby:
    warnings.warn(f'\nMOON IMPACT - FLYBY FAILED')

###########################################################################################

# For debugging
second_arc_arrival_fpa_debugging = calculated_fpa

# Print for debugging
print(f'\nDebugging: all values should match')
print(f'Root finder fpa: {second_arc_arrival_fpa_debugging*180/np.pi} deg')
print(f'Recalculated fpa: {second_arc_arrival_fpa*180/np.pi} deg')
print(f'Problem set fpa: {flight_path_angle_at_atmosphere_entry*180/np.pi} deg')


# Calculate other orbital parameters of the second arc
second_arc_semilatus_rectum = second_arc_angular_momentum**2/central_body_gravitational_parameter
second_arc_semimajor_axis = - central_body_gravitational_parameter / (2*second_arc_orbital_energy)
second_arc_eccentricity = np.sqrt(1-second_arc_semilatus_rectum/second_arc_semimajor_axis)


# ASSUMPTION: flyby doesn't change orbital plane
# ASSUMPTION: moon velocity vector lays on flyby plane

# Plotting for insight gaining
print('\nJupiter starting conditions:\n'
      f'- velocity: {interplanetary_arrival_velocity_in_jupiter_frame/1e3} km/s\n'
      f'- angle between velocity and radius: {delta_angle_from_hohmann_trajectory*180/np.pi} deg\n'
      f'- eccentricity: {second_arc_eccentricity}')
print('\nAtmospheric entry conditions:\n'
      f'- altitude: {arrival_pericenter_altitude/1e3} km\n'
      f'- velocity: {second_arc_arrival_velocity/1e3:.3f} km/s\n'
      f'- flight path angle: {second_arc_arrival_fpa*180/np.pi:.3f} deg')

### DRAW FLYBY #########################################################################################################

flyby_pericenter_velocity = B_parameter * flyby_v_inf_t / flyby_pericenter
flyby_angular_momentum = flyby_pericenter * flyby_pericenter_velocity
flyby_orbital_energy = flyby_pericenter_velocity**2/2 - mu_moon/flyby_pericenter

flyby_angular_momentum_vector = flyby_angular_momentum * flyby_axis

flyby_semilatus_rectum = flyby_angular_momentum**2/mu_moon
flyby_sma = - mu_moon / (2*flyby_orbital_energy)
flyby_eccentricity = np.sqrt(1-flyby_semilatus_rectum/flyby_sma)

true_anomaly_limit = true_anomaly_from_radius(moon_SOI_radius, flyby_eccentricity, flyby_sma)

true_anomaly_range = np.array([-true_anomaly_limit, true_anomaly_limit])
delta_t_flyby = delta_t_from_delta_true_anomaly(true_anomaly_range, flyby_eccentricity, flyby_sma, mu_moon)
print(f'\nFlyby elapsed time: {delta_t_flyby/3600} hrs')

flyby_pericenter_position = rotate_vectors_by_given_matrix(rotation_matrix(flyby_axis, beta_angle), -unit_vector(moon_velocity)) * flyby_pericenter

flyby_pericenter_angle_wrt_x_axis = np.arccos(np.dot(unit_vector(flyby_pericenter_position),x_axis))
if np.dot(np.cross(x_axis, flyby_pericenter_position), z_axis) < 0:
    flyby_pericenter_angle_wrt_x_axis = - flyby_pericenter_angle_wrt_x_axis + 2 * np.pi

true_anomaly_vector_flyby = np.linspace(-true_anomaly_limit, true_anomaly_limit, flyby_number_of_points)
radius_vector_flyby = radius_from_true_anomaly(true_anomaly_vector_flyby, flyby_eccentricity, flyby_sma)
true_anomaly_plot_flyby = np.linspace(flyby_pericenter_angle_wrt_x_axis-true_anomaly_limit,flyby_pericenter_angle_wrt_x_axis+true_anomaly_limit, flyby_number_of_points)

x_flyby, y_flyby = cartesian_2d_from_polar(radius_vector_flyby, true_anomaly_plot_flyby)
z_flyby = np.zeros(len(x_flyby))

# Create matrix with point coordinates
flyby_states = np.vstack((x_flyby, y_flyby, z_flyby)).T

# Calculate rotation matrix from orbit frame to ecliptic frame
# flyby_rotation_axis = flyby_axis
flyby_inclination = np.arccos(flyby_angular_momentum_vector[2]/LA.norm(flyby_angular_momentum_vector))
flyby_rotation_matrix = rotation_matrix(flyby_axis, -flyby_inclination)

# Transform coordinates of first arc to be in ecliptic frame
flyby_rotated_states = rotate_vectors_by_given_matrix(flyby_rotation_matrix,flyby_states)
x_flyby = flyby_rotated_states[:,0] + np.ones(len(flyby_rotated_states[:,0])) * moon_flyby_state[0]
y_flyby = flyby_rotated_states[:,1] + np.ones(len(flyby_rotated_states[:,1])) * moon_flyby_state[1]
z_flyby = flyby_rotated_states[:,2] + np.ones(len(flyby_rotated_states[:,2])) * moon_flyby_state[2]


### DRAWING TRAJECTORY ARCS ############################################################################################

# Find true anomalies at the second arc boundaries
second_arc_arrival_true_anomaly = true_anomaly_from_radius(second_arc_arrival_radius, second_arc_eccentricity, second_arc_semimajor_axis)
second_arc_departure_true_anomaly = true_anomaly_from_radius(second_arc_departure_radius, second_arc_eccentricity, second_arc_semimajor_axis)

# Swap true anomaly signs if the trajectory is counterclockwise
if np.dot(second_arc_angular_momentum_vector, z_axis) > 0:
    second_arc_arrival_true_anomaly = - second_arc_arrival_true_anomaly
    second_arc_departure_true_anomaly = - second_arc_departure_true_anomaly

# Calculate phase angle of second arc
second_arc_phase_angle = second_arc_arrival_true_anomaly - second_arc_departure_true_anomaly

# Find true anomalies at the first arc boundaries
first_arc_arrival_true_anomaly = - true_anomaly_from_radius(first_arc_arrival_radius, first_arc_eccentricity, first_arc_semimajor_axis)
first_arc_departure_true_anomaly = - true_anomaly_from_radius(first_arc_departure_radius, first_arc_eccentricity, first_arc_semimajor_axis)

# NOT NEEDED: Swap signs of true anomalies if needed
if...:
    ...

# Calculate phase angle of first arc
first_arc_phase_angle = first_arc_arrival_true_anomaly - first_arc_departure_true_anomaly

# Now end and start conditions w.r.t. x axis are different
first_arc_final_angle_wrt_x_axis = np.arccos(np.dot(unit_vector(first_arc_arrival_position),x_axis))
second_arc_initial_angle_wrt_x_axis = np.arccos(np.dot(unit_vector(second_arc_departure_position),x_axis))

# Check if such angles are grater than np.pi or not, and set them accordingly
if np.dot(np.cross(x_axis, first_arc_arrival_position), z_axis) < 0:
    first_arc_final_angle_wrt_x_axis = - first_arc_final_angle_wrt_x_axis + 2 * np.pi
if np.dot(np.cross(x_axis, second_arc_departure_position), z_axis) < 0:
    second_arc_initial_angle_wrt_x_axis = - second_arc_initial_angle_wrt_x_axis + 2 * np.pi

# Calculate coordinate points of the first arc to be plotted
true_anomaly_vector_arc_1 = np.linspace(first_arc_departure_true_anomaly, first_arc_arrival_true_anomaly, first_arc_number_of_points)
radius_vector_arc_1 = radius_from_true_anomaly(true_anomaly_vector_arc_1, first_arc_eccentricity, first_arc_semimajor_axis)
true_anomaly_plot_arc_1 = np.linspace(first_arc_final_angle_wrt_x_axis-first_arc_phase_angle,first_arc_final_angle_wrt_x_axis, first_arc_number_of_points)

# Calculate first arc cartesian coordinates in trajectory frame
x_arc1, y_arc1 = cartesian_2d_from_polar(radius_vector_arc_1, true_anomaly_plot_arc_1)
z_arc1 = np.zeros(len(x_arc1))

# Calculate coordinate points of the second arc to be plotted
true_anomaly_vector_arc_2 = np.linspace(second_arc_departure_true_anomaly, second_arc_arrival_true_anomaly, second_arc_number_of_points)
radius_vector_arc_2 = radius_from_true_anomaly(true_anomaly_vector_arc_2, second_arc_eccentricity, second_arc_semimajor_axis)
true_anomaly_plot_arc_2 = np.linspace(second_arc_initial_angle_wrt_x_axis,second_arc_initial_angle_wrt_x_axis + second_arc_phase_angle, second_arc_number_of_points)

# Calculate second arc cartesian coordinates in trajectory frame
x_arc2, y_arc2 = cartesian_2d_from_polar(radius_vector_arc_2, true_anomaly_plot_arc_2)
z_arc2 = np.zeros(len(x_arc2))

# Create matrix with point coordinates
first_arc_states = np.vstack((x_arc1, y_arc1, z_arc1)).T

# Calculate rotation matrix from orbit frame to ecliptic frame
first_arc_rotation_axis = unit_vector(np.cross(first_arc_angular_momentum_vector,z_axis))
first_arc_inclination = np.arccos(first_arc_angular_momentum_vector[2]/LA.norm(first_arc_angular_momentum_vector))
first_arc_rotation_matrix = rotation_matrix(first_arc_rotation_axis, -first_arc_inclination)

# Transform coordinates of first arc to be in ecliptic frame
first_arc_rotated_states = rotate_vectors_by_given_matrix(first_arc_rotation_matrix,first_arc_states)
x_arc1 = first_arc_rotated_states[:,0]
y_arc1 = first_arc_rotated_states[:,1]
z_arc1 = first_arc_rotated_states[:,2]

# Create matrix with point coordinates
second_arc_states = np.vstack((x_arc2, y_arc2, z_arc2)).T

# Calculate rotation matrix from orbit frame to ecliptic frame
second_arc_rotation_axis = unit_vector(np.cross(second_arc_angular_momentum_vector,z_axis))
second_arc_inclination = np.arccos(second_arc_angular_momentum_vector[2]/LA.norm(second_arc_angular_momentum_vector))
second_arc_rotation_matrix = rotation_matrix(second_arc_rotation_axis, -second_arc_inclination)

# Transform coordinates of second arc to be in ecliptic frame
second_arc_rotated_states = rotate_vectors_by_given_matrix(second_arc_rotation_matrix,second_arc_states)
x_arc2 = second_arc_rotated_states[:,0]
y_arc2 = second_arc_rotated_states[:,1]
z_arc2 = second_arc_rotated_states[:,2]

# Plot 3-D Trajectory
fig = plt.figure()
ax = plt.axes(projection='3d')

# Plot flyby and second arc
ax.plot3D(x_arc2, y_arc2, z_arc2, 'gray')
ax.plot3D(x_flyby, y_flyby, z_flyby, 'gray')

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('Trajectory up to jupiter atmosphere')

xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
ax.set_xlim3d(XYZlim)
ax.set_ylim3d(XYZlim)
ax.set_zlim3d(XYZlim * 0.75)
ax.set_aspect('auto')

# Plot first arc (done afterwards so that boundaries don't include full arrival)
ax.plot3D(x_arc1, y_arc1, z_arc1, 'gray')

# draw jupiter
u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
x = jupiter_radius * np.cos(u) * np.sin(v)
y = jupiter_radius * np.sin(u) * np.sin(v)
z = jupiter_radius * np.cos(v)
ax.plot_wireframe(x, y, z, color="r")


# draw moon
u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
x_0 = moon_flyby_state[0]
y_0 = moon_flyby_state[1]
z_0 = moon_flyby_state[2]
x = x_0 + moon_radius * np.cos(u) * np.sin(v)
y = y_0 + moon_radius * np.sin(u) * np.sin(v)
z = z_0 + moon_radius * np.cos(v)
ax.plot_wireframe(x, y, z, color="b")

# moon_flyby_end_state = spice_interface.get_body_cartesian_state_at_epoch(
#     target_body_name=choose_flyby_moon,
#     observer_body_name="Jupiter",
#     reference_frame_name=global_frame_orientation,
#     aberration_corrections="NONE",
#     ephemeris_time=moon_epoch_of_flyby+delta_t_flyby)
#
# # draw moon
# u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
# x_0 = moon_flyby_end_state[0]
# y_0 = moon_flyby_end_state[1]
# z_0 = moon_flyby_end_state[2]
# x = x_0 + moon_radius * np.cos(u) * np.sin(v)
# y = y_0 + moon_radius * np.sin(u) * np.sin(v)
# z = z_0 + moon_radius * np.cos(v)
# ax.plot_wireframe(x, y, z, color="b")


# origin = np.array([[0, 0, 0],[0, 0, 0]])
# V = np.concatenate(())
#     np.array([[1,1], [-2,2], [4,-7]])
#
# ax.quiver(*origin,V[:,0], V[:,1], color=['r','b','g'] )

# vectors for debugging
ax.plot3D([0., second_arc_departure_position[0]], [0., second_arc_departure_position[1]], [0., second_arc_departure_position[2]], 'red')
ax.plot3D([0., first_arc_arrival_position[0]], [0., first_arc_arrival_position[1]], [0., first_arc_arrival_position[2]], 'firebrick')
ax.plot3D([moon_position[0], moon_position[0] + flyby_pericenter_position[0]], [moon_position[1], moon_position[1] + flyby_pericenter_position[1]], [moon_position[2], moon_position[2] + flyby_pericenter_position[2]], 'firebrick')

ax.plot3D([second_arc_departure_position[0], second_arc_departure_velocity_vector[0]*1e3+second_arc_departure_position[0]],
          [second_arc_departure_position[1], second_arc_departure_velocity_vector[1]*1e3+second_arc_departure_position[1]],
          [second_arc_departure_position[2], second_arc_departure_velocity_vector[2]*1e3+second_arc_departure_position[2]], 'blue')
ax.plot3D([first_arc_arrival_position[0], first_arc_arrival_velocity_vector[0]*1e3+first_arc_arrival_position[0]],
          [first_arc_arrival_position[1], first_arc_arrival_velocity_vector[1]*1e3+first_arc_arrival_position[1]],
          [first_arc_arrival_position[2], first_arc_arrival_velocity_vector[2]*1e3+first_arc_arrival_position[2]], 'green')

plt.show()


