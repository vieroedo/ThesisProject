import warnings

from handle_functions import *
import first_order_equations_aerocapture as ae_first_order

"""
The orbit is planar and flybys occur at the center of mass of the moons
"""

do_regula_falsi_function_debugging = False

# PROBLEM PARAMETERS ###################################################################################################
# (Consider moving to related script in case other ones using this library need different parameters)

# Atmospheric entry conditions
arrival_pericenter_altitude = atmospheric_entry_altitude  # m
flight_path_angle_at_atmosphere_entry = -1.5  # degrees

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


moon_flyby_state = moon_circular_2d_state(epoch=213, choose_moon=p_ae_flyby_moon)

pre_ae_departure_radius = jupiter_SOI_radius
pre_ae_departure_velocity_norm = interplanetary_arrival_velocity_in_jupiter_frame

pre_ae_orbital_energy = orbital_energy(pre_ae_departure_radius, pre_ae_departure_velocity_norm)

pre_ae_arrival_radius = jupiter_radius + arrival_pericenter_altitude
pre_ae_arrival_velocity_norm = velocity_from_energy(pre_ae_orbital_energy, pre_ae_arrival_radius)

pre_ae_arrival_fpa = flight_path_angle_at_atmosphere_entry

pre_ae_angular_momentum_norm = pre_ae_arrival_radius * pre_ae_arrival_velocity_norm * np.cos(pre_ae_arrival_fpa)
pre_ae_angular_momentum = z_axis * pre_ae_angular_momentum_norm

pre_ae_semilatus_rectum = pre_ae_angular_momentum_norm ** 2 / jupiter_gravitational_parameter
pre_ae_semimajor_axis = - jupiter_gravitational_parameter / (2 * pre_ae_orbital_energy)
pre_ae_eccentricity = np.sqrt(1 - pre_ae_semilatus_rectum / pre_ae_semimajor_axis)

# pre_ae_arrival_radius = jupiter_radius + arrival_pericenter_altitude

pre_ae_arrival_position = x_axis * pre_ae_arrival_radius

circ_vel_at_atm_entry = np.sqrt(jupiter_gravitational_parameter / (jupiter_radius + arrival_pericenter_altitude))

print('\nAtmospheric entry (pre-aerocapture) conditions:\n'
      f'- altitude: {arrival_pericenter_altitude/1e3} km\n'
      f'- velocity: {pre_ae_arrival_velocity_norm / 1e3:.3f} km/s\n'
      f'- ref circular velocity: {circ_vel_at_atm_entry/1e3:.3f} km/s\n'
      f'- flight path angle: {pre_ae_arrival_fpa*180/np.pi:.3f} deg\n'
      f'- eccentricity: {pre_ae_eccentricity:.10f} ')

# Calculate initial state vector
pre_ae_departure_true_anomaly = true_anomaly_from_radius(pre_ae_departure_radius, pre_ae_eccentricity, pre_ae_semimajor_axis)
pre_ae_arrival_true_anomaly = true_anomaly_from_radius(pre_ae_arrival_radius, pre_ae_eccentricity, pre_ae_semimajor_axis)

delta_true_anomaly = pre_ae_arrival_true_anomaly - pre_ae_departure_true_anomaly

pos_rotation_matrix = rotation_matrix(z_axis, -delta_true_anomaly)
pre_ae_departure_position = rotate_vectors_by_given_matrix(pos_rotation_matrix, unit_vector(pre_ae_arrival_position)) * pre_ae_departure_radius

pre_ae_departure_fpa = - np.arccos(pre_ae_angular_momentum_norm / (pre_ae_departure_radius * pre_ae_departure_velocity_norm))

vel_rotation_matrix = rotation_matrix(z_axis, np.pi/2 - pre_ae_departure_fpa)
pre_ae_departure_velocity = rotate_vectors_by_given_matrix(vel_rotation_matrix, unit_vector(pre_ae_departure_position)) * pre_ae_departure_velocity_norm

print('\nDeparture state:')
print(f'{list(np.concatenate((pre_ae_departure_position, pre_ae_departure_velocity)))}')

# Entry conditions (from arcs 1 and 2)
atmospheric_entry_fpa = pre_ae_arrival_fpa
atmospheric_entry_velocity_norm = pre_ae_arrival_velocity_norm
atmospheric_entry_altitude = arrival_pericenter_altitude
atmospheric_entry_g_acc = jupiter_gravitational_parameter / (pre_ae_arrival_radius ** 2)

# # Atmosphere model coefficients (exponential atmosphere)
jupiter_beta_parameter = 1 / jupiter_scale_height
# density_at_atmosphere_entry = jupiter_1bar_density * np.exp(-atmospheric_entry_altitude*jupiter_beta_parameter) # assumed exp model
#
#
# # Capsule properties and coefficients
# #  make L/D = 0.5 -> https://www.researchgate.net/publication/238790363_Aerodynamic_Control_on_a_Lunar_Return_Capsule_using_Trim-Flaps
lift_over_drag_ratio = vehicle_cl/vehicle_cd


atmospheric_entry_initial_position = pre_ae_arrival_position

fpa_entry, fpa_minimum_altitude, fpa_exit = ae_first_order.calculate_fpa_boundaries(atmospheric_entry_fpa)
fpa_linspace = np.linspace(fpa_entry, fpa_exit, 100)

aerocapture_quantities, other_data = ae_first_order.first_order_approximation_aerocapture(
    fpa_linspace, fpa_entry, fpa_minimum_altitude,
    atmospheric_entry_altitude + jupiter_radius,
    atmospheric_entry_velocity_norm)

# Minimum altitude
minimum_altitude = other_data[0]

ae_radii = aerocapture_quantities[0]
ae_velocities = aerocapture_quantities[1]
ae_fpas = aerocapture_quantities[2]
ae_densities = aerocapture_quantities[3]
ae_drag = aerocapture_quantities[4]
ae_lift = aerocapture_quantities[5]
ae_wall_hfx = aerocapture_quantities[6]
ae_range_angles = aerocapture_quantities[7]

# Atmosphere exit fpa
atmospheric_exit_fpa = ae_fpas[-1]

# Atmosphere exit velocity
atmospheric_exit_velocity_norm = ae_velocities[-1]

# Travelled distance (assumed at surface)
atmospheric_entry_final_phase_angle = ae_range_angles[-1]
final_distance_travelled = atmospheric_entry_final_phase_angle * jupiter_radius

atmosph_entry_rot_matrix = rotation_matrix(pre_ae_angular_momentum, atmospheric_entry_final_phase_angle)
atmospheric_entry_final_position = rotate_vectors_by_given_matrix(atmosph_entry_rot_matrix,
                                                                  atmospheric_entry_initial_position)


fpa_max_acceleration, av_max_over_g, a_total_max_over_g = ae_first_order.calculate_max_loads_point(
                                        atmospheric_entry_fpa, atmospheric_entry_velocity_norm, lift_over_drag_ratio)


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

if p_ae_moon_fb_sma > p_ae_apocenter:
    warnings.warn(f'orbit too low for post aerocapture flyby. apocenter - moon SMA: {(p_ae_apocenter-p_ae_moon_fb_sma)/1e3:.3f} km')
    p_ae_arrival_position = p_ae_departure_position
    final_orbit_dep_vel_w_fb = p_ae_departure_velocity
else:
    # third_arc_arrival_position = p_ae_moon_fb_position
    p_ae_arrival_radius = p_ae_moon_fb_sma
    p_ae_arrival_velocity_norm = np.sqrt(2 * (p_ae_orbital_energy + jupiter_gravitational_parameter / p_ae_arrival_radius))
    p_ae_arrival_fpa = np.arccos(p_ae_angular_momentum_norm / (p_ae_arrival_radius * p_ae_arrival_velocity_norm))
    # third_arc_arrival_fpa_s = (third_arc_arrival_fpa_sol, -third_arc_arrival_fpa_sol) # we check both options

    p_ae_departure_true_anomaly = true_anomaly_from_radius(p_ae_departure_radius, p_ae_eccentricity, p_ae_semimajor_axis)
    p_ae_arrival_true_anomaly = true_anomaly_from_radius(p_ae_arrival_radius, p_ae_eccentricity, p_ae_semimajor_axis)
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
print(f'- Stagnation point peak heat flux: {max(ae_wall_hfx)}')
print('- Integrated heat load: ...')
print(f'- Minimum altitude: {minimum_altitude/1e3:.3f} km')
print(f'- Horizontal distance travelled: {final_distance_travelled/1e3:.3f} km')
# print(f'- Horizontal distance travelled check for debug: {final_downrange/1e3:.3f} km')


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
                   pre_ae_arrival_fpa),
        # 'post_ae': (
        # p_ae_departure_radius, p_ae_arrival_position, p_ae_eccentricity, p_ae_semimajor_axis, p_ae_arrival_fpa),
        # 'Third': (third_arc_departure_radius, third_arc_arrival_position, third_arc_eccentricity, third_arc_semimajor_axis, third_arc_arrival_fpa),
    }
else:
    arcs_dictionary = {
        'pre_ae': (pre_ae_departure_radius, pre_ae_arrival_position, pre_ae_eccentricity, pre_ae_semimajor_axis, pre_ae_arrival_fpa),
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
    arc_arrival_true_anomaly = np.sign(arc_arrival_fpa) * true_anomaly_from_radius(arc_arrival_radius, arc_eccentricity, arc_semimajor_axis)
    arc_departure_true_anomaly = np.sign(arc_arrival_fpa) * true_anomaly_from_radius(arc_departure_radius, arc_eccentricity, arc_semimajor_axis)

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

plt.show()