import numpy as np

from handle_functions import *

"""
The orbit is planar and flybys occur at the center of mass of the moons
"""


# known issues:
# none
do_regula_falsi_function_debugging = False


# PROBLEM PARAMETERS ###################################################################################################
# (Consider moving to related script in case other ones using this library need different parameters)

# Atmospheric entry conditions
arrival_pericenter_altitude = 400e3  # m (DO NOT CHANGE - consider changing only with valid and sound reasons)
flight_path_angle_at_atmosphere_entry = -0.2  # degrees

# Jupiter arrival conditions
interplanetary_arrival_velocity_in_jupiter_frame = 5600  # m/s
delta_angle_from_hohmann_trajectory = 1.02  # degrees

# First flyby parameters
choose_flyby_moon = 'Callisto'
safety_altitude_flyby = 0.

# Post-aerocapture flyby parameters
p_ae_moon_fb_name = 'Io'  # CANNOT be equal to the moon of first flyby

# Plotting parameters
number_of_epochs_to_plot = 200

########################################################################################################################


# Parameters reprocessing ##############################################################################################
flight_path_angle_at_atmosphere_entry = flight_path_angle_at_atmosphere_entry * np.pi / 180  # rad
delta_angle_from_hohmann_trajectory_to_plot = delta_angle_from_hohmann_trajectory  # deg
delta_angle_from_hohmann_trajectory = delta_angle_from_hohmann_trajectory * np.pi / 180  # rad
first_arc_number_of_points = number_of_epochs_to_plot
second_arc_number_of_points = number_of_epochs_to_plot
########################################################################################################################


moon_flyby_state = moon_circular_2d_state(epoch=213, choose_moon=choose_flyby_moon)


########################################################################################################################
# ORBIT DETERMINATION OF THE FIRST ARC #################################################################################
########################################################################################################################

# Calculate first arc orbital plane based on the one of the moon
first_arc_angular_momentum_cap = unit_vector(np.cross(moon_flyby_state[0:3], moon_flyby_state[3:6]))

# First arc initial known conditions
first_arc_departure_radius = jupiter_SOI_radius
first_arc_departure_velocity_norm = interplanetary_arrival_velocity_in_jupiter_frame
angle_between_departure_position_and_velocity = np.pi - delta_angle_from_hohmann_trajectory

# First arc final known conditions
first_arc_arrival_radius = LA.norm(moon_flyby_state[0:3])

# Calculate first arc orbital energy
first_arc_orbital_energy = first_arc_departure_velocity_norm ** 2 / 2 - central_body_gravitational_parameter / first_arc_departure_radius

# Calculate max allowed delta_hoh based on chosen initial velocity
frac_numerator = np.sqrt(2*first_arc_arrival_radius*(first_arc_orbital_energy*first_arc_arrival_radius + central_body_gravitational_parameter))
max_allowed_delta_hoh = np.arcsin(frac_numerator / (first_arc_departure_radius * first_arc_departure_velocity_norm))

# End script if the angle exceeds the max value
if delta_angle_from_hohmann_trajectory > max_allowed_delta_hoh:
    raise Exception(fr'The chosen delta_angle_from_hohmann_trajectory is too big, max value allowed is of {max_allowed_delta_hoh*180/np.pi} degrees with an initial velocity of {first_arc_departure_velocity_norm / 1e3} km/s')

# Calculate angular momentum magnitude and build the vector
first_arc_angular_momentum_norm = first_arc_departure_radius * first_arc_departure_velocity_norm * np.sin(angle_between_departure_position_and_velocity)
first_arc_angular_momentum = first_arc_angular_momentum_norm * first_arc_angular_momentum_cap

# Calculate p, sma, and e of the first arc
first_arc_semilatus_rectum = first_arc_angular_momentum_norm ** 2 / central_body_gravitational_parameter
first_arc_semimajor_axis = - central_body_gravitational_parameter / (2 * first_arc_orbital_energy)
first_arc_eccentricity = np.sqrt(1 - first_arc_semilatus_rectum/ first_arc_semimajor_axis)


# ASSUMPTION: Arrival position is set to the moon's center, from that arrival velocity is determined
first_arc_arrival_position = moon_flyby_state[0:3]
first_arc_arrival_velocity_norm = np.sqrt(2 * (first_arc_orbital_energy + central_body_gravitational_parameter / first_arc_arrival_radius))

# Calculate arrival fpa of the first arc. It's negative since the trajectory is counterclockwise.
first_arc_arrival_fpa = -np.arccos(np.clip(first_arc_angular_momentum_norm / (first_arc_arrival_radius * first_arc_arrival_velocity_norm), -1, 1))

# Find the first_arc_arrival_velocity_vector
velocity_rotation_matrix = rotation_matrix(first_arc_angular_momentum, np.pi / 2 - first_arc_arrival_fpa)
first_arc_arrival_velocity = rotate_vectors_by_given_matrix(velocity_rotation_matrix,unit_vector(first_arc_arrival_position)) * first_arc_arrival_velocity_norm


########################################################################################################################
# ROOT FINDING FOR PERICENTER FLYBY RADIUS #############################################################################
########################################################################################################################

# Retrieve flyby moon data: Radius, mu, velocity vector
moon_radius = galilean_moons_data[choose_flyby_moon]['Radius']
moon_SOI_radius = galilean_moons_data[choose_flyby_moon]['SOI_Radius']
mu_moon = galilean_moons_data[choose_flyby_moon]['mu']
moon_velocity = moon_flyby_state[3:6]

# Calculate flyby incoming v_infinity
flyby_initial_velocity = first_arc_arrival_velocity - moon_velocity

# Calculate second arc departure position (assumed flyby to happen in a point in space, at the moon's centre)
second_arc_departure_position = first_arc_arrival_position

# Set the arrival radius to be at the edge of Jupiter's atmosphere
second_arc_arrival_radius = jupiter_radius + arrival_pericenter_altitude

# Sub-problem free parameter: flyby_pericenter_radius


# ILLINOIS METHOD ######################################################################################################
interval_left_boundary_a = moon_radius
interval_right_boundary_b = moon_SOI_radius

a_int = interval_left_boundary_a
b_int = interval_right_boundary_b

tolerance = 1e-7
max_iter = 1000


c_point, f_c, i = regula_falsi_illinois((a_int, b_int), calculate_fpa_from_flyby_pericenter,
                                        flight_path_angle_at_atmosphere_entry, tolerance, max_iter,
                                        arc_arrival_radius=second_arc_arrival_radius,
                                        arc_departure_position=second_arc_departure_position,
                                        flyby_initial_velocity_vector=flyby_initial_velocity,
                                        mu_moon=mu_moon,
                                        moon_in_plane_velocity=moon_velocity)

# Found root
flyby_pericenter = c_point
calculated_fpa = f_c + flight_path_angle_at_atmosphere_entry

if flyby_pericenter-moon_radius < 0:
    raise Warning('Moon impact: flyby failed.')


# Debugging
print(f'Number of iterations: {i}')
print(f'Flyby pericenter altitude: {(flyby_pericenter-moon_radius)/1e3:.3f} km')
print(f'f.p.a. result of root finder: {calculated_fpa*180/np.pi:.5f} deg')


########################################################################################################################
# CALCULATE FLYBY AND SECOND ARC WITH FOUND PERICENTER RADIUS  #########################################################
########################################################################################################################

# CALCULATE FLYBY WITH FOUND PERICENTER RADIUS #########################################################################

# Flyby moon data retrieved above -> Radius, SOI radius, mu, velocity vector
# flyby_pericenter -> found by root finder

# flyby_initial_velocity_vector -> took from above

# Calculate magnitude of the flyby v_inf_t
flyby_v_inf_t = LA.norm(flyby_initial_velocity)

# Calculate the axis normal to the flyby plane using v_inf_t and v_moon
flyby_orbital_plane_normal_axis = unit_vector(np.cross(moon_velocity,flyby_initial_velocity))

# Calculate the resulting bending angle on the velocity vector
flyby_alpha_angle = 2 * np.arcsin(1/(1+flyby_pericenter*flyby_v_inf_t**2/mu_moon))

# Rotate the incoming v_infinity by alpha to obtain the departing v_infinity
flyby_velocity_rotation_matrix = rotation_matrix(flyby_orbital_plane_normal_axis, flyby_alpha_angle)
flyby_final_velocity = rotate_vectors_by_given_matrix(flyby_velocity_rotation_matrix, flyby_initial_velocity)

# Calculate the delta v of the flyby
flyby_delta_v = LA.norm(flyby_final_velocity - flyby_initial_velocity)

# Calculate max bending angle and delta v
alpha_max = 2 * np.arcsin(1/(1+flyby_v_inf_t**2/(mu_moon/moon_radius)))
flyby_delta_v_max = 2 * flyby_v_inf_t * np.sin(alpha_max/2)

# Print flyby data
print(f'\n Moon of flyby: {choose_flyby_moon}')
print(f'Flyby alpha angle: {flyby_alpha_angle*180/np.pi:.5f} deg')
print(f'Flyby delta_v: {flyby_delta_v/1e3:.3f} km/s')
print(f'Max delta_v achievable for {choose_flyby_moon}: {flyby_delta_v_max/1e3:.3f} km/s')

# CALCULATE SECOND ARC WITH FOUND PERICENTER RADIUS ####################################################################

# Calculate the initial position and radius of the second arc
# second_arc_departure_position -> Second arc departure position calculated above
second_arc_departure_radius = LA.norm(second_arc_departure_position)

# Calculate the initial velocity vector and magnitude as a result of the flyby
second_arc_departure_velocity = flyby_final_velocity + moon_velocity
second_arc_departure_velocity_norm = LA.norm(second_arc_departure_velocity)

# Calculate the orbital energy of the second arc
second_arc_orbital_energy = second_arc_departure_velocity_norm ** 2 / 2 - central_body_gravitational_parameter / second_arc_departure_radius

# Calculate the departure flight path angel of the second arc
second_arc_departure_fpa = np.arcsin(np.dot(unit_vector(second_arc_departure_position), unit_vector(second_arc_departure_velocity)))

# Set the arrival radius to be at the edge of Jupiter's atmosphere
# second_arc_arrival_radius -> set above

# Calculate arrival velocity at Jupiter atmospheric entry
second_arc_arrival_velocity_norm = np.sqrt(2 * (second_arc_orbital_energy + central_body_gravitational_parameter / second_arc_arrival_radius))

# Pre-calculation for second arc arrival flight path angle
arccos_arg = second_arc_departure_radius / second_arc_arrival_radius * second_arc_departure_velocity_norm * np.cos(second_arc_departure_fpa) / second_arc_arrival_velocity_norm

# Calculate second arc arrival flight path angle (signed)
second_arc_arrival_fpa = - np.arccos(arccos_arg)

# Calculate second arc angular momentum
second_arc_angular_momentum_norm = second_arc_arrival_radius * second_arc_arrival_velocity_norm * np.cos(second_arc_arrival_fpa)
second_arc_angular_momentum = np.cross(second_arc_departure_position, second_arc_departure_velocity)

# Calculate other orbital parameters of the second arc
second_arc_semilatus_rectum = second_arc_angular_momentum_norm ** 2 / central_body_gravitational_parameter
second_arc_semimajor_axis = - central_body_gravitational_parameter / (2*second_arc_orbital_energy)
second_arc_eccentricity = np.sqrt(1-second_arc_semilatus_rectum/second_arc_semimajor_axis)

# Plot problem parameters
print('\nJupiter starting conditions:\n'
      f'- velocity: {interplanetary_arrival_velocity_in_jupiter_frame/1e3} km/s\n'
      f'- angle between velocity and radius: {delta_angle_from_hohmann_trajectory_to_plot} deg\n'
      f'- eccentricity: {first_arc_eccentricity:.10f}')
print('\nAtmospheric entry conditions:\n'
      f'- altitude: {arrival_pericenter_altitude/1e3} km\n'
      f'- velocity: {second_arc_arrival_velocity_norm/1e3:.3f} km/s\n'
      f'- flight path angle: {second_arc_arrival_fpa*180/np.pi:.3f} deg\n'
      f'- eccentricity: {second_arc_eccentricity:.10f} ')


# Find true anomalies at the second arc boundaries
second_arc_arrival_true_anomaly = true_anomaly_from_radius(second_arc_arrival_radius, second_arc_eccentricity, second_arc_semimajor_axis)
second_arc_departure_true_anomaly = true_anomaly_from_radius(second_arc_departure_radius, second_arc_eccentricity, second_arc_semimajor_axis)

# Swap true anomaly signs if the trajectory is counterclockwise
if np.dot(second_arc_angular_momentum, z_axis) > 0:
    second_arc_arrival_true_anomaly = - second_arc_arrival_true_anomaly
    second_arc_departure_true_anomaly = - second_arc_departure_true_anomaly

# Calculate phase angle of second arc
second_arc_phase_angle = second_arc_arrival_true_anomaly - second_arc_departure_true_anomaly

second_arc_arrival_position = rotate_vectors_by_given_matrix(rotation_matrix(second_arc_angular_momentum, second_arc_phase_angle), unit_vector(second_arc_departure_position)) * second_arc_arrival_radius

########################################################################################################################
# CALCULATE ATMOSPHERIC ENTRY ARC  #####################################################################################
########################################################################################################################

# Entry conditions (from arcs 1 and 2)
atmospheric_entry_fpa = second_arc_arrival_fpa
atmospheric_entry_velocity_norm = second_arc_arrival_velocity_norm
atmospheric_entry_altitude = arrival_pericenter_altitude
atmospheric_entry_g_acc = central_body_gravitational_parameter / (second_arc_arrival_radius ** 2)

# Atmosphere model coefficients (exponential atmosphere)
reference_density = jupiter_1bar_density
scale_height = jupiter_scale_height
beta_parameter = 1 / scale_height
density_at_atmosphere_entry = reference_density * np.exp(-atmospheric_entry_altitude*beta_parameter) # assumed exp model

# Capsule properties and coefficients
lift_coefficient = vehicle_cl  # set to 0.6 to make L/D = 0.5 -> https://www.researchgate.net/publication/238790363_Aerodynamic_Control_on_a_Lunar_Return_Capsule_using_Trim-Flaps
drag_coefficient = vehicle_cd
lift_over_drag_ratio = lift_coefficient/drag_coefficient
capsule_mass = vehicle_mass  # kg
capsule_surface = vehicle_reference_area  #12.5  # m^2

capsule_weight = capsule_mass * atmospheric_entry_g_acc

# Atmosphere exit fpa
atmospheric_exit_fpa = - atmospheric_entry_fpa

# Atmosphere exit velocity
exponential_argument = 2 * atmospheric_entry_fpa / lift_over_drag_ratio
atmospheric_exit_velocity_norm = atmospheric_entry_velocity_norm * np.exp(exponential_argument)

# Check vehicle doesn't go too deep into Jupiter atmosphere (not below zero-level altitude)
weight_over_surface_cl_coefficient = capsule_weight / (capsule_surface * lift_coefficient)
minimum_altitude_condition_value = atmospheric_entry_g_acc * reference_density \
                                   / (4 * beta_parameter * np.sin(atmospheric_entry_fpa/2)**2)
if weight_over_surface_cl_coefficient > minimum_altitude_condition_value:
    raise Warning('Minimum altitude is below zero!! Trajectory is theoretically possible, but heat loads can be prohibitive.')

# # Angle at a certain time idk
# gamma_angle = ...

effective_entry_fpa = - np.arccos(np.cos(atmospheric_entry_fpa) - density_at_atmosphere_entry * atmospheric_entry_g_acc / (2*beta_parameter) * 1 / weight_over_surface_cl_coefficient)

minimum_altitude = atmospheric_entry_trajectory_altitude(0., atmospheric_entry_fpa, density_at_atmosphere_entry,
                                                         reference_density, weight_over_surface_cl_coefficient,
                                                         atmospheric_entry_g_acc, beta_parameter)

pressure_at_minimum_altitude = ...


# Travelled distance (assumed at surface)
final_distance_travelled = atmospheric_entry_trajectory_distance_travelled(atmospheric_exit_fpa, atmospheric_entry_fpa,
                                                                           effective_entry_fpa, scale_height)
final_distance_travelled_check = -2*(atmospheric_entry_altitude-minimum_altitude)*(1/np.tan(effective_entry_fpa)) + 2/beta_parameter*(-atmospheric_entry_fpa+np.log(2))


atmospheric_entry_phase_angle = final_distance_travelled / jupiter_radius

atmosph_entry_rot_matrix = rotation_matrix(second_arc_angular_momentum, atmospheric_entry_phase_angle)
atmospheric_entry_final_position = rotate_vectors_by_given_matrix(atmosph_entry_rot_matrix, second_arc_arrival_position)


# Calculate loads on the spacecraft
term_1 = lift_over_drag_ratio / (2*(1 + np.cos(atmospheric_entry_fpa)))
term_2 = 1 - np.sqrt(1 + (4 * np.sin(atmospheric_entry_fpa)**2)/(lift_over_drag_ratio**2))
fpa_max_acceleration = 2 * np.arctan(term_1 * term_2)

g_earth = 9.81  # m/s^2
av_max_over_g = beta_parameter * atmospheric_entry_velocity_norm ** 2 / (g_earth * lift_over_drag_ratio) \
                  * (np.cos(fpa_max_acceleration) - np.cos(atmospheric_entry_fpa)) \
                  * np.exp(- 2 * (fpa_max_acceleration - atmospheric_entry_fpa) / lift_over_drag_ratio)

a_total_max_over_g = av_max_over_g * np.sqrt(1 + lift_over_drag_ratio**2)



# Heat loads
stagnation_point_heat_flux = ...
integrated_heat_load = ...



# atm entry prints
print('\nAtmospheric entry trajectory parameters:')
print(f'- Max acceleration on the spacecraft: {a_total_max_over_g:.3f} g  (1g = 9.81 m/s^2)')
print('- Stagnation point heat flux: ...')
print('- Integrated heat load: ...')
print(f'- Minimum altitude: {minimum_altitude/1e3:.3f} km')
print(f'- Horizontal distance travelled: {final_distance_travelled/1e3:.3f} km')

########################################################################################################################
# CALCULATE THIRD ORBITAL ARC  #########################################################################################
########################################################################################################################

third_arc_departure_velocity_norm = atmospheric_exit_velocity_norm
third_arc_departure_fpa = atmospheric_exit_fpa
third_arc_departure_radius = second_arc_arrival_radius
third_arc_departure_position = atmospheric_entry_final_position

third_arc_orbital_energy = orbital_energy(third_arc_departure_radius, third_arc_departure_velocity_norm)

third_arc_orbital_axis = unit_vector(second_arc_angular_momentum)

# Find the third_arc_departure_velocity
third_arc_velocity_rotation_matrix = rotation_matrix(third_arc_orbital_axis, np.pi / 2 - third_arc_departure_fpa)
third_arc_departure_velocity = rotate_vectors_by_given_matrix(third_arc_velocity_rotation_matrix,unit_vector(third_arc_departure_position)) * third_arc_departure_velocity_norm

third_arc_angular_momentum = np.cross(third_arc_departure_position, third_arc_departure_velocity)
third_arc_angular_momentum_norm = LA.norm(third_arc_angular_momentum)

# Calculate other orbital parameters of the second arc
third_arc_semilatus_rectum = third_arc_angular_momentum_norm ** 2 / central_body_gravitational_parameter
third_arc_semimajor_axis = - central_body_gravitational_parameter / (2*third_arc_orbital_energy)
third_arc_eccentricity = np.sqrt(1-third_arc_semilatus_rectum/third_arc_semimajor_axis)


# Print third arc departure conditions (atmospheric exit conditions)
print('\nAtmospheric exit (/third arc initial) conditions:\n'
      f'- altitude: {arrival_pericenter_altitude/1e3} km\n'
      f'- velocity: {third_arc_departure_velocity_norm/1e3:.3f} km/s\n'
      f'- flight path angle: {third_arc_departure_fpa*180/np.pi:.3f} deg\n'
      f'- eccentricity: {third_arc_eccentricity:.10f} ')


# CALCULATE POST-AEROCAPTURE MOON FLYBY ################################################################################

# POST-AEROCAPTURE MOON FLYBY == p_ae_moon_fb



p_ae_moon_fb_sma = galilean_moons_data[p_ae_moon_fb_name]['SMA']
p_ae_moon_fb_velocity_norm = np.sqrt(central_body_gravitational_parameter/p_ae_moon_fb_sma)
# p_ae_moon_fb_position = rotate_vectors_by_given_matrix(rotation_matrix(third_arc_orbital_axis, p_ae_moon_fb_phase_angle), unit_vector(third_arc_departure_position)) * p_ae_moon_fb_sma
# p_ae_moon_fb_velocity = rotate_vectors_by_given_matrix(rotation_matrix(third_arc_orbital_axis, np.pi/2), unit_vector(p_ae_moon_fb_position)) * p_ae_moon_fb_velocity_norm


# p_ae_moon_fb_state = np.concatenate((p_ae_moon_fb_position, p_ae_moon_fb_velocity))

third_arc_apocenter = third_arc_semimajor_axis * (1 + third_arc_eccentricity)
third_arc_pericenter = third_arc_semimajor_axis * (1 - third_arc_eccentricity)

if p_ae_moon_fb_sma > third_arc_apocenter:
    raise Exception('orbit too low for post aerocapture flyby')

# third_arc_arrival_position = p_ae_moon_fb_position
third_arc_arrival_radius = p_ae_moon_fb_sma
third_arc_arrival_velocity_norm = np.sqrt(2 * (third_arc_orbital_energy + central_body_gravitational_parameter / third_arc_arrival_radius))
third_arc_arrival_fpa = np.arccos(third_arc_angular_momentum_norm/(third_arc_arrival_radius*third_arc_arrival_velocity_norm))
# third_arc_arrival_fpa_s = (third_arc_arrival_fpa_sol, -third_arc_arrival_fpa_sol) # we check both options

third_arc_departure_true_anomaly = true_anomaly_from_radius(third_arc_departure_radius, third_arc_eccentricity, third_arc_semimajor_axis)
third_arc_arrival_true_anomaly = true_anomaly_from_radius(third_arc_arrival_radius, third_arc_eccentricity, third_arc_semimajor_axis)
# third_arc_arrival_true_anomalies = (third_arc_arrival_true_anomaly_sol, -third_arc_arrival_true_anomaly_sol)

third_arc_phase_angle = third_arc_arrival_true_anomaly - third_arc_departure_true_anomaly

p_ae_moon_fb_position = rotate_vectors_by_given_matrix(rotation_matrix(third_arc_orbital_axis, third_arc_phase_angle), unit_vector(third_arc_departure_position)) * p_ae_moon_fb_sma
p_ae_moon_fb_velocity = rotate_vectors_by_given_matrix(rotation_matrix(third_arc_orbital_axis, np.pi/2), unit_vector(p_ae_moon_fb_position)) * p_ae_moon_fb_velocity_norm

p_ae_moon_fb_state = np.concatenate((p_ae_moon_fb_position, p_ae_moon_fb_velocity))

third_arc_arrival_position = rotate_vectors_by_given_matrix(rotation_matrix(third_arc_orbital_axis, third_arc_phase_angle), unit_vector(third_arc_departure_position)) * third_arc_arrival_radius
third_arc_arrival_velocity = rotate_vectors_by_given_matrix(rotation_matrix(third_arc_orbital_axis, np.pi/2 - third_arc_arrival_fpa), unit_vector(third_arc_arrival_position)) * third_arc_arrival_velocity_norm

# DO FLYBY SO THAT PERICENTER IS AT LEAST AT h=2000km form 1bar level of Jupiter

########################################################################################################################
# CALCULATE POST AEROCAPTURE FLYBY  ####################################################################################
########################################################################################################################


p_ae_moon_fb_mu = galilean_moons_data[p_ae_moon_fb_name]['mu']
p_ae_moon_fb_radius = galilean_moons_data[p_ae_moon_fb_name]['Radius']
p_ae_moon_fb_SOI_radius = galilean_moons_data[p_ae_moon_fb_name]['SOI_Radius']

p_ae_fb_initial_velocity = third_arc_arrival_velocity - p_ae_moon_fb_velocity


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
                                                                       arc_departure_position=third_arc_arrival_position,
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
                                        arc_departure_position=third_arc_arrival_position,
                                        flyby_initial_velocity_vector=p_ae_fb_initial_velocity,
                                        mu_moon=p_ae_moon_fb_mu,
                                        moon_flyby_state=p_ae_moon_fb_state
                                        )

# Found root
second_flyby_pericenter = c_point
orbit_pericenter = f_c + desired_value

# Debugging
print(f'\nNumber of iterations second time: {i}')
print(f'Second flyby pericenter altitude: {(second_flyby_pericenter-p_ae_moon_fb_radius)/1e3:.3f} km')
print(f'Orbit pericenter altitude result of root finder: {(orbit_pericenter-jupiter_radius)/1e3:.3f} km')


########################################################################################################################
# CALCULATE FOURTH ORBITAL ARC  ########################################################################################
########################################################################################################################


# Calculate v_inf_t
p_ae_fb_initial_velocity_norm = LA.norm(p_ae_fb_initial_velocity)

# Calculate axis normal to flyby plane (based on assumption:flyby plane coincides with moon orbital plane)
p_ae_fb_orbital_axis = unit_vector(np.cross(p_ae_moon_fb_state[0:3], p_ae_moon_fb_state[3:6]))

# Calculate resulting flyby bending angle
p_ae_fb_alpha_angle = 2 * np.arcsin(1 / (1 + second_flyby_pericenter * p_ae_fb_initial_velocity_norm ** 2 / p_ae_moon_fb_mu))

# Calculate the v_inf_t_star
p_ae_fb_final_velocity = (rotation_matrix(p_ae_fb_orbital_axis, p_ae_fb_alpha_angle) @
                               p_ae_fb_initial_velocity.reshape(3, 1)).reshape(3)

# Get initial radius and position of post-aerocapture post-flyby arc
fourth_arc_departure_position = third_arc_arrival_position
fourth_arc_departure_radius = LA.norm(fourth_arc_departure_position)

# Calculate post-aerocapture post-flyby arc departure velocity
fourth_arc_departure_velocity = p_ae_fb_final_velocity + p_ae_moon_fb_state[3:6]
fourth_arc_departure_velocity_norm = LA.norm(fourth_arc_departure_velocity)

# Calculate post-flyby arc departure flight path angle
fourth_arc_departure_fpa = np.arcsin(
    np.dot(unit_vector(fourth_arc_departure_position), unit_vector(fourth_arc_departure_velocity)))

# Calculate post-flyby arc orbital energy
fourth_arc_orbital_energy = fourth_arc_departure_velocity_norm ** 2 / 2 - \
                            central_body_gravitational_parameter / fourth_arc_departure_radius

fourth_arc_angular_momentum = fourth_arc_departure_radius * fourth_arc_departure_velocity_norm * np.cos(fourth_arc_departure_fpa)

fourth_arc_semilatus_rectum = fourth_arc_angular_momentum ** 2 / central_body_gravitational_parameter
fourth_arc_semimajor_axis = - central_body_gravitational_parameter / (2 * fourth_arc_orbital_energy)
fourth_arc_eccentricity = np.sqrt(1 - fourth_arc_semilatus_rectum / fourth_arc_semimajor_axis)

fourth_arc_orbital_period = 2* np.pi * np.sqrt(fourth_arc_semimajor_axis**3/central_body_gravitational_parameter)

# Print fourth arc quantities for debugging
print(f'\n\nPost-aerocapture flyby moon: {p_ae_moon_fb_name}')
print(f'Post-aerocapture flyby pericenter altitude: {(second_flyby_pericenter-p_ae_moon_fb_radius)/1e3:.3f} km')
print(f'Final orbit pericenter altitude: {(orbit_pericenter-jupiter_radius)/1e3:.3f} km')
print(f'Final orbit eccentricity: {fourth_arc_eccentricity:.5f}')


print(f'\nFinal orbit orbital period: {fourth_arc_orbital_period/constants.JULIAN_DAY:.3f} days')

moons = ['Io', 'Europa', 'Ganymede', 'Callisto']
for moon in moons:
    moon_period = galilean_moons_data[moon]['Orbital_Period']
    print(f'- expressed in {moon}\'s period: {fourth_arc_orbital_period/moon_period}')


# Post other quantities
print(f'\n\nFirst arc orbital specific energy: {first_arc_orbital_energy/1e3:.3f} kJ/m')
print(f'Second arc orbital specific energy: {second_arc_orbital_energy/1e3:.3f} kJ/m')
print(f'Third arc orbital specific energy: {third_arc_orbital_energy/1e3:.3f} kJ/m')
print(f'Fourth arc orbital specific energy: {fourth_arc_orbital_energy/1e3:.3f} kJ/m')
print(f'Arcs 1-2 delta orbital specific energy: {abs(first_arc_orbital_energy-second_arc_orbital_energy)/1e3:.3f} kJ/m')
print(f'Arcs 2-3 delta orbital specific energy: {abs(second_arc_orbital_energy-third_arc_orbital_energy)/1e3:.3f} kJ/m')
print(f'Arcs 3-4 delta orbital specific energy: {abs(third_arc_orbital_energy-fourth_arc_orbital_energy)/1e3:.3f} kJ/m')


########################################################################################################################
# DRAW TRAJECTORY  #####################################################################################################
########################################################################################################################


# FIRST, SECOND, AND THIRD ARCS ########################################################################################

arcs_dictionary = {
    'First': (first_arc_departure_radius, first_arc_arrival_position, first_arc_eccentricity, first_arc_semimajor_axis, first_arc_arrival_fpa),
    'Second': (second_arc_departure_radius, second_arc_arrival_position, second_arc_eccentricity, second_arc_semimajor_axis, second_arc_arrival_fpa),
    'Third': (third_arc_departure_radius, third_arc_arrival_position, third_arc_eccentricity, third_arc_semimajor_axis, third_arc_arrival_fpa),
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

final_orbit_eccentricity = fourth_arc_eccentricity
final_orbit_semimajor_axis = fourth_arc_semimajor_axis
final_orbit_reference_position = fourth_arc_departure_position
final_orbit_reference_velocity = fourth_arc_departure_velocity

position_multiplier = LA.norm(final_orbit_reference_velocity) **2 - central_body_gravitational_parameter/LA.norm(final_orbit_reference_position)
velocity_multiplier = np.dot(final_orbit_reference_position, final_orbit_reference_velocity)
eccentricity_vector = 1/central_body_gravitational_parameter * (position_multiplier * final_orbit_reference_position - velocity_multiplier * final_orbit_reference_velocity)


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


# draw first moon
u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
x_0 = moon_flyby_state[0]
y_0 = moon_flyby_state[1]
z_0 = moon_flyby_state[2]
x = x_0 + moon_radius * np.cos(u) * np.sin(v)
y = y_0 + moon_radius * np.sin(u) * np.sin(v)
z = z_0 + moon_radius * np.cos(v)
ax.plot_wireframe(x, y, z, color="b")

# draw second moon
u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
x_0 = p_ae_moon_fb_state[0]
y_0 = p_ae_moon_fb_state[1]
z_0 = p_ae_moon_fb_state[2]
x = x_0 + p_ae_moon_fb_radius * np.cos(u) * np.sin(v)
y = y_0 + p_ae_moon_fb_radius * np.sin(u) * np.sin(v)
z = z_0 + p_ae_moon_fb_radius * np.cos(v)
ax.plot_wireframe(x, y, z, color="b")


########################################################################################################################
# RE-ENTRY PLOTS  ######################################################################################################
########################################################################################################################

fpa_vector = np.linspace(atmospheric_entry_fpa, atmospheric_exit_fpa, 200)

altitude_vector = atmospheric_entry_trajectory_altitude(fpa_vector, atmospheric_entry_fpa, density_at_atmosphere_entry,
                                                        reference_density, weight_over_surface_cl_coefficient,
                                                        atmospheric_entry_g_acc, beta_parameter)
downrange_vector = atmospheric_entry_trajectory_distance_travelled(fpa_vector, atmospheric_entry_fpa, effective_entry_fpa, scale_height)

fig2, ax2 = plt.subplots(figsize=(5,6))
ax2.plot(downrange_vector/1e3, altitude_vector/1e3)
ax2.set(xlabel='downrange [km]', ylabel='altitude [km]')


# ax2.plot(fpa_vector,downrange_vector)

# ax2.set_aspect('equal', 'box')
plt.show()
