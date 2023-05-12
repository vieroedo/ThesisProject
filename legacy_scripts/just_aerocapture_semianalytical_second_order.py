import warnings

from handle_functions import *

from second_order_equations_aerocapture import *
import CapsuleEntryUtilities as Util

"""
The orbit is planar and flybys occur at the center of mass of the moons
"""

do_regula_falsi_function_debugging = False

# PROBLEM PARAMETERS ###################################################################################################
# (Consider moving to related script in case other ones using this library need different parameters)

# Atmospheric entry conditions
arrival_pericenter_altitude = atmospheric_entry_altitude  # m (DO NOT CHANGE - consider changing only with valid and sound reasons)
flight_path_angle_at_atmosphere_entry = -3  # degrees

# Jupiter arrival conditions
interplanetary_arrival_velocity_in_jupiter_frame = 5600  # m/s

# Post-aerocapture flyby parameters
p_ae_flyby_moon = 'Io'

# Plotting parameters
number_of_epochs_to_plot = 200

########################################################################################################################


# Parameters reprocessing ##############################################################################################
flight_path_angle_at_atmosphere_entry = flight_path_angle_at_atmosphere_entry * np.pi / 180  # rad
first_arc_number_of_points = number_of_epochs_to_plot
second_arc_number_of_points = number_of_epochs_to_plot
########################################################################################################################

# Calculate first arc quantities and the initial state vector
pre_ae_departure_radius = jupiter_SOI_radius
pre_ae_departure_velocity_norm = interplanetary_arrival_velocity_in_jupiter_frame

pre_ae_orbital_energy = orbital_energy(pre_ae_departure_radius, pre_ae_departure_velocity_norm)

pre_ae_arrival_radius = jupiter_radius + arrival_pericenter_altitude
pre_ae_arrival_velocity_norm = velocity_from_energy(pre_ae_orbital_energy, pre_ae_arrival_radius)

pre_ae_angular_momentum_norm = pre_ae_arrival_radius * pre_ae_arrival_velocity_norm * np.cos(flight_path_angle_at_atmosphere_entry)
pre_ae_angular_momentum = z_axis * pre_ae_angular_momentum_norm

pre_ae_semilatus_rectum = pre_ae_angular_momentum_norm ** 2 / jupiter_gravitational_parameter
pre_ae_semimajor_axis = - jupiter_gravitational_parameter / (2 * pre_ae_orbital_energy)
pre_ae_eccentricity = np.sqrt(1 - pre_ae_semilatus_rectum / pre_ae_semimajor_axis)

pre_ae_arrival_position = x_axis * pre_ae_arrival_radius

initial_state_vector = Util.get_initial_state(flight_path_angle_at_atmosphere_entry, atmospheric_entry_altitude, z_axis,
                                              interplanetary_arrival_velocity_in_jupiter_frame, verbose=True)

# AEROCAPTURE ##########################################################################################################

# Entry conditions (from arcs 1 and 2)
atmospheric_entry_fpa = flight_path_angle_at_atmosphere_entry
atmospheric_entry_velocity_norm = pre_ae_arrival_velocity_norm
atmospheric_entry_altitude = arrival_pericenter_altitude
# atmospheric_entry_g_acc = jupiter_gravitational_parameter / (pre_ae_arrival_radius ** 2)
atmospheric_entry_initial_position = pre_ae_arrival_position

tau_entry, tau_minimum_altitude, tau_exit, a1, a2, a3 = calculate_tau_boundaries_second_order_equations(
                                                             atmospheric_entry_altitude+jupiter_radius,
                                                             atmospheric_entry_velocity_norm, atmospheric_entry_fpa,
                                                             K_hypersonic=Util.vehicle_hypersonic_K_parameter)

tau_linspace = np.linspace(tau_entry, tau_exit, 100)

# (radius, velocity, flight_path_angle, density, drag, lift, wall_heat_flux)
aerocapture_quantities = second_order_approximation_aerocapture(tau_linspace,tau_minimum_altitude, a1, a2, a3,
                                                                atmospheric_entry_altitude+jupiter_radius,
                                                                atmospheric_entry_velocity_norm, atmospheric_entry_fpa,
                                                                K_hypersonic=Util.vehicle_hypersonic_K_parameter)

ae_range_angles = tau_linspace * np.sqrt(
            jupiter_scale_height / (atmospheric_entry_altitude + jupiter_radius))
ae_radii = aerocapture_quantities[0]
ae_velocities = aerocapture_quantities[1]
ae_fpas = aerocapture_quantities[2]
ae_densities = aerocapture_quantities[3]
ae_drag = aerocapture_quantities[4]
ae_lift = aerocapture_quantities[5]
ae_wall_hfx =  aerocapture_quantities[6]

# Atmosphere exit fpa
atmospheric_exit_fpa = ae_fpas[-1]

# Atmosphere exit velocity
atmospheric_exit_velocity_norm = ae_velocities[-1]

# Minimum altitude
x_tau_min_alt = x_tau_function(tau_minimum_altitude, a1, a2, a3)
minimum_altitude = - jupiter_scale_height * x_tau_min_alt + atmospheric_entry_altitude


# Travelled distance (assumed at surface)
atmospheric_entry_phase_angle = ae_range_angles[-1]
final_distance_travelled = atmospheric_entry_phase_angle * jupiter_radius

atmosph_entry_rot_matrix = rotation_matrix(pre_ae_angular_momentum, atmospheric_entry_phase_angle)
atmospheric_entry_final_position = rotate_vectors_by_given_matrix(atmosph_entry_rot_matrix, pre_ae_arrival_position)


x_unal, y_unal, z_unal = cartesian_3d_from_polar(ae_radii,np.zeros(len(ae_radii)),ae_range_angles)

entry_position_states_unaligned = np.vstack((x_unal, y_unal, z_unal))

atmospheric_entry_reference_angle = np.arcsin(LA.norm(np.cross(x_axis, atmospheric_entry_initial_position))/LA.norm(atmospheric_entry_initial_position))
if np.dot(z_axis, np.cross(x_axis, atmospheric_entry_initial_position)) < 0:
    atmospheric_entry_reference_angle = atmospheric_entry_reference_angle + np.pi

entry_positions_rot_matrix = rotation_matrix(pre_ae_angular_momentum, atmospheric_entry_reference_angle)
entry_position_states = rotate_vectors_by_given_matrix(entry_positions_rot_matrix,
                                                       entry_position_states_unaligned)


entry_velocity_states =np.zeros(np.shape(entry_position_states))
if max(np.shape(entry_velocity_states)) <= 3:
    raise Exception('too few instances of the reentry trajectory (less or equal than 3)')
for i in range(max(np.shape(entry_velocity_states))):
    entry_velocities_rot_matrix = rotation_matrix(pre_ae_angular_momentum, np.pi/2 - ae_fpas[i])
    entry_velocity_states[i, :] = rotate_vectors_by_given_matrix(entry_velocities_rot_matrix, entry_position_states[i, :])

entry_cartesian_states = np.concatenate((entry_position_states, entry_velocity_states), axis=1)
dependent_variables = np.vstack((ae_fpas,ae_velocities,ae_radii,ae_densities, ae_drag, ae_lift, ae_wall_hfx)).T

# Calculate loads on the spacecraft
a_total_max_over_g = np.nan
# Heat loads
stagnation_point_heat_flux = ...
integrated_heat_load = ...


print('\n Atmosphere exit conditions:\n'
      f'- exit velocity: {atmospheric_exit_velocity_norm/1e3:.3f} km/s')

p_ae_departure_velocity_norm = atmospheric_exit_velocity_norm
p_ae_departure_fpa = atmospheric_exit_fpa
p_ae_departure_radius = pre_ae_arrival_radius
p_ae_departure_position = atmospheric_entry_final_position

p_ae_orbital_energy = orbital_energy(p_ae_departure_radius, p_ae_departure_velocity_norm)

p_ae_orbital_axis = z_axis

# Find the third_arc_departure_velocity
p_ae_velocity_rotation_matrix = rotation_matrix(p_ae_orbital_axis, np.pi / 2 - p_ae_departure_fpa)
p_ae_departure_velocity = rotate_vectors_by_given_matrix(p_ae_velocity_rotation_matrix, unit_vector(p_ae_departure_position)) * p_ae_departure_velocity_norm

p_ae_angular_momentum = np.cross(p_ae_departure_position, p_ae_departure_velocity)
p_ae_angular_momentum_norm = LA.norm(p_ae_angular_momentum)

# Calculate other orbital parameters of the second arc
p_ae_semilatus_rectum = p_ae_angular_momentum_norm ** 2 / jupiter_gravitational_parameter
p_ae_semimajor_axis = - jupiter_gravitational_parameter / (2 * p_ae_orbital_energy)
p_ae_eccentricity = np.sqrt(1 - p_ae_semilatus_rectum / p_ae_semimajor_axis)


p_ae_moon_fb_sma = galilean_moons_data[p_ae_flyby_moon]['SMA']
p_ae_moon_fb_velocity_norm = np.sqrt(jupiter_gravitational_parameter / p_ae_moon_fb_sma)

p_ae_apocenter = p_ae_semimajor_axis * (1 + p_ae_eccentricity)
p_ae_pericenter = p_ae_semimajor_axis * (1 - p_ae_eccentricity)

if p_ae_moon_fb_sma > abs(p_ae_apocenter):
    warnings.warn(f'orbit too low for post aerocapture flyby. apocenter - moon SMA: {(p_ae_apocenter-p_ae_moon_fb_sma)/1e3:.3f} km')
    p_ae_arrival_position = p_ae_departure_position
    final_orbit_dep_vel_w_fb = p_ae_departure_velocity
else:
    # third_arc_arrival_position = p_ae_moon_fb_position
    p_ae_arrival_radius = p_ae_moon_fb_sma
    p_ae_arrival_velocity_norm = np.sqrt(2 * (p_ae_orbital_energy + jupiter_gravitational_parameter / p_ae_arrival_radius))
    p_ae_arrival_fpa = np.arccos(p_ae_angular_momentum_norm / (p_ae_arrival_radius * p_ae_arrival_velocity_norm))
    # third_arc_arrival_fpa_s = (third_arc_arrival_fpa_sol, -third_arc_arrival_fpa_sol) # we check both options

    p_ae_departure_true_anomaly = true_anomaly_from_radius(p_ae_departure_radius, p_ae_eccentricity,
                                                           p_ae_semimajor_axis, True)
    p_ae_arrival_true_anomaly = true_anomaly_from_radius(p_ae_arrival_radius, p_ae_eccentricity, p_ae_semimajor_axis,
                                                         True)
    # third_arc_arrival_true_anomalies = (third_arc_arrival_true_anomaly_sol, -third_arc_arrival_true_anomaly_sol)

    p_ae_phase_angle = p_ae_arrival_true_anomaly - p_ae_departure_true_anomaly

    p_ae_moon_fb_position = rotate_vectors_by_given_matrix(rotation_matrix(p_ae_orbital_axis, p_ae_phase_angle), unit_vector(p_ae_departure_position)) * p_ae_moon_fb_sma
    p_ae_moon_fb_velocity = rotate_vectors_by_given_matrix(rotation_matrix(p_ae_orbital_axis, np.pi/2), unit_vector(p_ae_moon_fb_position)) * p_ae_moon_fb_velocity_norm

    p_ae_moon_fb_state = np.concatenate((p_ae_moon_fb_position, p_ae_moon_fb_velocity))

    p_ae_arrival_position = rotate_vectors_by_given_matrix(rotation_matrix(p_ae_orbital_axis, p_ae_phase_angle), unit_vector(p_ae_departure_position)) * p_ae_arrival_radius
    p_ae_arrival_velocity = rotate_vectors_by_given_matrix(rotation_matrix(p_ae_orbital_axis, np.pi / 2 - p_ae_arrival_fpa), unit_vector(p_ae_arrival_position)) * p_ae_arrival_velocity_norm

    ########################################################################################################################
    # CALCULATE POST AEROCAPTURE FLYBY  ####################################################################################
    ########################################################################################################################

    p_ae_moon_fb_mu = galilean_moons_data[p_ae_flyby_moon]['mu']
    p_ae_moon_fb_radius = galilean_moons_data[p_ae_flyby_moon]['Radius']
    p_ae_moon_fb_SOI_radius = galilean_moons_data[p_ae_flyby_moon]['SOI_Radius']

    p_ae_fb_initial_velocity = p_ae_arrival_velocity - p_ae_moon_fb_velocity

    # ILLINOIS METHOD ######################################################################################################
    interval_left_boundary_a = p_ae_moon_fb_radius
    interval_right_boundary_b = p_ae_moon_fb_SOI_radius

    desired_value = jupiter_radius + 2000e3

    # DEBUG #############
    if do_regula_falsi_function_debugging:
        radii = np.linspace(p_ae_moon_fb_radius, p_ae_moon_fb_SOI_radius, 500)
        function_values = np.zeros(len(radii))
        for i, chosen_radius in enumerate(radii):
            rp_function = calculate_orbit_pericenter_from_flyby_pericenter(flyby_rp=chosen_radius,
                                                                           arc_departure_position=p_ae_arrival_position,
                                                                           flyby_initial_velocity_vector=p_ae_fb_initial_velocity,
                                                                           mu_moon=p_ae_moon_fb_mu,
                                                                           moon_flyby_state=p_ae_moon_fb_state)

            f_function = rp_function - desired_value
            function_values[i] = f_function

        plt.axhline(y=0)
        plt.plot(radii, function_values)
        plt.show()
        quit()

    #####################


    a_int = interval_left_boundary_a
    b_int = interval_right_boundary_b

    tolerance = 1e-7
    max_iter = 1000

    c_point, f_c, i = regula_falsi_illinois((a_int, b_int), calculate_orbit_pericenter_from_flyby_pericenter,
                                            desired_value, tolerance, max_iter,
                                            arc_departure_position=p_ae_arrival_position,
                                            flyby_initial_velocity_vector=p_ae_fb_initial_velocity,
                                            mu_moon=p_ae_moon_fb_mu,
                                            moon_flyby_state=p_ae_moon_fb_state
                                            )

    # Found root
    flyby_pericenter = c_point
    orbit_pericenter = f_c + desired_value


    # Debugging
    print(f'\nNumber of iterations second time: {i}')
    print(f'Second flyby pericenter altitude: {(flyby_pericenter - p_ae_moon_fb_radius) / 1e3:.3f} km')
    print(f'Orbit pericenter altitude result of root finder: {(orbit_pericenter-jupiter_radius)/1e3:.3f} deg')

########################################################################################################################
# CALCULATE FINAL ORBIT  ###############################################################################################
########################################################################################################################


    # Calculate v_inf_t
    p_ae_fb_initial_velocity_norm = LA.norm(p_ae_fb_initial_velocity)

    # Calculate axis normal to flyby plane (based on assumption:flyby plane coincides with moon orbital plane)
    p_ae_fb_orbital_axis = unit_vector(np.cross(p_ae_moon_fb_state[0:3], p_ae_moon_fb_state[3:6]))

    # Calculate resulting flyby bending angle
    p_ae_fb_alpha_angle = 2 * np.arcsin(1 / (1 + flyby_pericenter * p_ae_fb_initial_velocity_norm ** 2 / p_ae_moon_fb_mu))

    # Calculate the v_inf_t_star
    p_ae_fb_final_velocity = (rotation_matrix(p_ae_fb_orbital_axis, p_ae_fb_alpha_angle) @
                                   p_ae_fb_initial_velocity.reshape(3, 1)).reshape(3)
    final_orbit_dep_vel_w_fb = p_ae_fb_final_velocity + p_ae_moon_fb_state[3:6]

# Get initial radius and position of post-aerocapture post-flyby arc
final_orbit_departure_position = p_ae_arrival_position
final_orbit_departure_radius = LA.norm(final_orbit_departure_position)

# Calculate post-aerocapture post-flyby arc departure velocity
final_orbit_departure_velocity = final_orbit_dep_vel_w_fb
final_orbit_departure_velocity_norm = LA.norm(final_orbit_departure_velocity)

# Calculate post-flyby arc departure flight path angle
final_orbit_departure_fpa = np.arcsin(
    np.dot(unit_vector(final_orbit_departure_position), unit_vector(final_orbit_departure_velocity)))

# Calculate post-flyby arc orbital energy
final_orbit_orbital_energy = final_orbit_departure_velocity_norm ** 2 / 2 - \
                             jupiter_gravitational_parameter / final_orbit_departure_radius

final_orbit_angular_momentum = final_orbit_departure_radius * final_orbit_departure_velocity_norm * np.cos(final_orbit_departure_fpa)

final_orbit_semilatus_rectum = final_orbit_angular_momentum ** 2 / jupiter_gravitational_parameter
final_orbit_semimajor_axis = - jupiter_gravitational_parameter / (2 * final_orbit_orbital_energy)
final_orbit_eccentricity = np.sqrt(1 - final_orbit_semilatus_rectum / final_orbit_semimajor_axis)

final_orbit_orbital_period = 2 * np.pi * np.sqrt(final_orbit_semimajor_axis ** 3 / jupiter_gravitational_parameter)

# atm entry prints
print('\nAtmospheric entry trajectory parameters:')
print(f'- Max acceleration on the spacecraft: {a_total_max_over_g:.3f} g  (1g = 9.81 m/s^2)')
print('- Stagnation point heat flux: ...')
print('- Integrated heat load: ...')
print(f'- Minimum altitude: {minimum_altitude/1e3:.3f} km')
print(f'- Horizontal distance travelled: {final_distance_travelled/1e3:.3f} km')


# Print fourth arc quantities for debugging
print(f'\n\nPost-aerocapture flyby moon: {p_ae_flyby_moon}')
if p_ae_moon_fb_sma < p_ae_apocenter:
    print(f'Post-aerocapture flyby pericenter altitude: {(flyby_pericenter-p_ae_moon_fb_radius)/1e3:.3f} km')
    print(f'Final orbit pericenter altitude: {(orbit_pericenter-jupiter_radius)/1e3:.3f} km')
print(f'Final orbit eccentricity: {p_ae_eccentricity:.5f}')


print(f'\nFinal orbit orbital period: {final_orbit_orbital_period/constants.JULIAN_DAY:.3f} days')

moons = ['Io', 'Europa', 'Ganymede', 'Callisto']
for moon in moons:
    moon_period = galilean_moons_data[moon]['Orbital_Period']
    print(f'- expressed in {moon}\'s period: {final_orbit_orbital_period / moon_period}')


# Post other quantities
print(f'\n\nPre arocapture orbital energy: {pre_ae_orbital_energy/1e3:.3f} kJ')
print(f'Post aerocapture orbital energy: {p_ae_orbital_energy/1e3:.3f} kJ')
print(f'Final orbit orbital energy: {final_orbit_orbital_energy / 1e3:.3f} kJ')
print(f'pre/post a-e delta orbital energy: {abs(pre_ae_orbital_energy-p_ae_orbital_energy)/1e3:.3f} kJ')
print(f'pre-post flyby delta orbital energy: {abs(p_ae_orbital_energy - final_orbit_orbital_energy) / 1e3:.3f} kJ')


########################################################################################################################
# DRAW TRAJECTORY  #####################################################################################################
########################################################################################################################


# FIRST, SECOND, AND THIRD ARCS ########################################################################################

if p_ae_moon_fb_sma > p_ae_apocenter:
    arcs_dictionary = {
        'pre_ae': (pre_ae_departure_radius, pre_ae_arrival_position, pre_ae_eccentricity, pre_ae_semimajor_axis,
                   flight_path_angle_at_atmosphere_entry),
        # 'post_ae': (
        # p_ae_departure_radius, p_ae_arrival_position, p_ae_eccentricity, p_ae_semimajor_axis, p_ae_arrival_fpa),
        # 'Third': (third_arc_departure_radius, third_arc_arrival_position, third_arc_eccentricity, third_arc_semimajor_axis, third_arc_arrival_fpa),
    }
else:
    arcs_dictionary = {
        'pre_ae': (pre_ae_departure_radius, pre_ae_arrival_position, pre_ae_eccentricity, pre_ae_semimajor_axis, flight_path_angle_at_atmosphere_entry),
        'post_ae': (p_ae_departure_radius, p_ae_arrival_position, p_ae_eccentricity, p_ae_semimajor_axis, p_ae_arrival_fpa),
        # 'Third': (third_arc_departure_radius, third_arc_arrival_position, third_arc_eccentricity, third_arc_semimajor_axis, third_arc_arrival_fpa),
    }

arc_number_of_points = number_of_epochs_to_plot

# Plot 3-D Trajectory
fig = plt.figure()
ax = plt.axes(projection='3d')

for arc in arcs_dictionary.keys():

    arc_departure_radius = arcs_dictionary[arc][0]
    arc_arrival_position = arcs_dictionary[arc][1]
    arc_arrival_radius = LA.norm(arc_arrival_position)

    arc_eccentricity = arcs_dictionary[arc][2]
    arc_semimajor_axis = arcs_dictionary[arc][3]
    arc_arrival_fpa = arcs_dictionary[arc][4]

    # Find true anomalies at the first arc boundaries
    arc_arrival_true_anomaly = np.sign(arc_arrival_fpa) * true_anomaly_from_radius(arc_arrival_radius, arc_eccentricity,
                                                                                   arc_semimajor_axis, True)
    arc_departure_true_anomaly = np.sign(arc_arrival_fpa) * true_anomaly_from_radius(arc_departure_radius,
                                                                                     arc_eccentricity,
                                                                                     arc_semimajor_axis, True)

    # Calculate phase angle of first arc
    arc_phase_angle = arc_arrival_true_anomaly - arc_departure_true_anomaly

    # End and start conditions w.r.t. x axis are the same, so we take the position at the node
    arc_arrival_position_angle_wrt_x_axis = np.arccos(np.dot(unit_vector(arc_arrival_position),x_axis))

    # Check if such angle is greater than np.pi or not, and set it accordingly
    if np.dot(np.cross(x_axis, arc_arrival_position), z_axis) < 0:
        arc_arrival_position_angle_wrt_x_axis = - arc_arrival_position_angle_wrt_x_axis + 2 * np.pi

    # Calculate coordinate points of the first arc to be plotted
    final_orbit_true_anomaly_vector = np.linspace(arc_departure_true_anomaly, arc_arrival_true_anomaly, arc_number_of_points)
    radius_vector = radius_from_true_anomaly(final_orbit_true_anomaly_vector, arc_eccentricity, arc_semimajor_axis)
    final_orbit_true_anomaly_plot = np.linspace(arc_arrival_position_angle_wrt_x_axis - arc_phase_angle,
                                                arc_arrival_position_angle_wrt_x_axis, arc_number_of_points)

    # Calculate first arc cartesian coordinates in trajectory frame
    x_arc, y_arc = cartesian_2d_from_polar(radius_vector, final_orbit_true_anomaly_plot)
    z_arc = np.zeros(len(x_arc))

    # Create matrix with point coordinates
    # first_arc_states = np.vstack((x_arc, y_arc, z_arc)).T

    ax.plot3D(x_arc, y_arc, z_arc, 'gray')


# FINAL ORBIT ##########################################################################################################

final_orbit_number_of_points = 2*number_of_epochs_to_plot

# final_orbit_eccentricity
# final_orbit_semimajor_axis
final_orbit_reference_position = final_orbit_departure_position
final_orbit_reference_velocity = final_orbit_departure_velocity

position_multiplier = LA.norm(final_orbit_reference_velocity) ** 2 - jupiter_gravitational_parameter / LA.norm(final_orbit_reference_position)
velocity_multiplier = np.dot(final_orbit_reference_position, final_orbit_reference_velocity)
eccentricity_vector = 1 / jupiter_gravitational_parameter * (position_multiplier * final_orbit_reference_position - velocity_multiplier * final_orbit_reference_velocity)


final_orbit_pericenter_angle_wrt_x_axis = np.arccos(np.dot(unit_vector(eccentricity_vector), x_axis))

# Check if such angle is greater than np.pi or not, and set it accordingly
if np.dot(np.cross(x_axis, eccentricity_vector), z_axis) < 0:
    final_orbit_pericenter_angle_wrt_x_axis = - final_orbit_pericenter_angle_wrt_x_axis + 2 * np.pi

final_orbit_true_anomaly_vector = np.linspace(-np.pi, np.pi, final_orbit_number_of_points)
final_orbit_radius_vector = radius_from_true_anomaly(final_orbit_true_anomaly_vector, final_orbit_eccentricity, final_orbit_semimajor_axis)
final_orbit_true_anomaly_plot = np.linspace(final_orbit_pericenter_angle_wrt_x_axis - np.pi,
                                            final_orbit_pericenter_angle_wrt_x_axis + np.pi, final_orbit_number_of_points)

# Calculate first arc cartesian coordinates in trajectory frame
x_final_orbit, y_final_orbit = cartesian_2d_from_polar(final_orbit_radius_vector, final_orbit_true_anomaly_plot)
z_final_orbit = np.zeros(len(x_final_orbit))

ax.plot3D(x_final_orbit, y_final_orbit, z_final_orbit, 'r')

# Plot entry trajectory
ax.plot3D(entry_cartesian_states[:,0], entry_cartesian_states[:,1], entry_cartesian_states[:,2])

# PLOT FIGURE ARRANGEMENT ##############################################################################################

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('Jupiter full arrival trajectory')

lines = ax.get_lines()
last_line = lines[-1]

# Get the data for the last line
line_data = last_line.get_data_3d()

# Get the data limits for the last line
x_data_limits = line_data[0].min(), line_data[0].max()
y_data_limits = line_data[1].min(), line_data[1].max()
z_data_limits = line_data[2].min(), line_data[2].max()

xyzlim = np.array([x_data_limits, y_data_limits, z_data_limits]).T
# xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
ax.set_xlim3d(XYZlim)
ax.set_ylim3d(XYZlim)
ax.set_zlim3d(XYZlim * 0.75)
ax.set_aspect('auto')

# draw jupiter
u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
x = jupiter_radius * np.cos(u) * np.sin(v)
y = jupiter_radius * np.sin(u) * np.sin(v)
z = jupiter_radius * np.cos(v)
ax.plot_wireframe(x, y, z, color="saddlebrown")

if p_ae_moon_fb_sma < p_ae_apocenter:
    # draw second moon
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x_0 = p_ae_moon_fb_state[0]
    y_0 = p_ae_moon_fb_state[1]
    z_0 = p_ae_moon_fb_state[2]
    x = x_0 + p_ae_moon_fb_radius * np.cos(u) * np.sin(v)
    y = y_0 + p_ae_moon_fb_radius * np.sin(u) * np.sin(v)
    z = z_0 + p_ae_moon_fb_radius * np.cos(v)
    ax.plot_wireframe(x, y, z, color="b")
else:
    # draw post-ae possibly flyby moon orbit
    for moon in galilean_moons_data.keys():
        moon_sma = galilean_moons_data[moon]['SMA']
        theta_angle = np.linspace(0, 2 * np.pi, 200)
        x_m = moon_sma * np.cos(theta_angle)
        y_m = moon_sma * np.sin(theta_angle)
        z_m = np.zeros(len(theta_angle))
        ax.plot3D(x_m, y_m, z_m, 'b')

########################################################################################################################
# RE-ENTRY PLOTS  ######################################################################################################
########################################################################################################################

fpa_vector = np.linspace(atmospheric_entry_fpa, atmospheric_exit_fpa, 200)

altitude_vector = ae_radii-jupiter_radius
    # atmospheric_entry_trajectory_altitude(fpa_vector, atmospheric_entry_fpa, density_at_atmosphere_entry,
    #                                                     reference_density, ballistic_coefficient_times_g_acc,
    #                                                     atmospheric_entry_g_acc, jupiter_beta_parameter)
downrange_vector = tau_linspace * np.sqrt(jupiter_scale_height/(atmospheric_entry_altitude+jupiter_radius)) * jupiter_radius
    # atmospheric_entry_trajectory_distance_travelled(fpa_vector, atmospheric_entry_fpa, effective_entry_fpa, scale_height)

fig2, ax2 = plt.subplots(figsize=(5,6))
ax2.plot(downrange_vector/1e3, altitude_vector/1e3)
ax2.set(xlabel='downrange [km]', ylabel='altitude [km]')

plt.show()