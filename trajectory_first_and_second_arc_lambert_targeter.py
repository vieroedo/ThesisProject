import matplotlib.pyplot as plt
import numpy as np

from handle_functions import *

# For loop to find best initial epoch to start the trajectory (s.t. the start is consistent w Hohmann traj.)
departure_epoch_days = 11201.  # days

# COMPUTE FIRST ARC ###################################################################################################
reference_values = ...


########################################################################################################################

# first_arc_final_state = np.array([482836072.70040,479542929.83204,19715476.43289,-20008.94853,954.57201,-44.84575])


# COMPUTE FLYBY FROM FIRST TO SECOND ARC ###############################################################################
compute_precise_timing = False
# Current epoch at beginning of flyby
flyby_initial_epoch = 974937600.0  # s # equal to final epoch of arc 1

# Arrival state from first arc
first_arc_arrival_position = first_arc_final_state[0:3]
first_arc_arrival_velocity = first_arc_final_state[3:6]

# Retrieve moon state from SPICE kernels - NOTE: it is the same state that gets obtained at the end of arc 1 so consider substituting
initial_state_moon_center = spice_interface.get_body_cartesian_state_at_epoch(
    target_body_name=choose_flyby_moon,
    observer_body_name="Jupiter",
    reference_frame_name=global_frame_orientation,
    aberration_corrections="NONE",
    ephemeris_time=flyby_initial_epoch)

# Moon and flyby data
moon_SOI_radius = galilean_moons_data[choose_flyby_moon]['SOI_Radius']
moon_radius = galilean_moons_data[choose_flyby_moon]['Radius']
mu_moon = galilean_moons_data[choose_flyby_moon]['mu']
safety_altitude_flyby = 100e3  # m

# Moon state at beginning of flyby
moon_initial_position = initial_state_moon_center[0:3] # at moon SOI arrival epoch
moon_initial_velocity = initial_state_moon_center[3:6] # at moon SOI arrival epoch

# Flyby initial state
flyby_initial_position = first_arc_arrival_position - moon_initial_position  # set to be at the moon SOI edge
flyby_initial_velocity = first_arc_arrival_velocity - moon_initial_velocity

# Flyby parameters and angles
phi_2_flyby = np.arccos(np.dot(unit_vector(-moon_initial_velocity), unit_vector(flyby_initial_velocity)))
delta_angle_flyby = np.arccos(np.dot(unit_vector(-moon_initial_velocity), unit_vector(flyby_initial_position)))
impact_parameter_B_flyby = moon_SOI_radius * np.sin(phi_2_flyby - delta_angle_flyby)

# Flyby outcomes
recurring_term = (impact_parameter_B_flyby ** 2 * LA.norm(flyby_initial_velocity) ** 4 / mu_moon ** 2)
pericenter_radius_flyby = mu_moon / LA.norm(flyby_initial_velocity) ** 2 * (np.sqrt(1 + recurring_term) - 1)
if pericenter_radius_flyby < moon_radius + safety_altitude_flyby:
    warnings.warn('MOON IMPACT - FLYBY FAILED')
alpha_angle_flyby = 2 * np.arcsin(1 / np.sqrt(1 + recurring_term))
beta_angle_flyby = phi_2_flyby + alpha_angle_flyby/2 - np.pi/2

# Flyby final state
flyby_final_position = ccw_rotation_z(flyby_initial_position, 2 * (2 * np.pi - delta_angle_flyby + beta_angle_flyby))
flyby_final_velocity = ccw_rotation_z(flyby_initial_velocity, alpha_angle_flyby)

# Flyby final epoch calculation
flyby_eccentricity = - 1 / np.cos((np.pi + alpha_angle_flyby)/2)
flyby_sma = pericenter_radius_flyby / (1-flyby_eccentricity)
if compute_precise_timing:
    true_anomaly_flyby_end = np.arccos(np.dot(unit_vector(-moon_initial_velocity), unit_vector(flyby_final_position))) * 0.05
    eccentric_anomaly = 2 * np.arctanh(np.tan(true_anomaly_flyby_end/2)) * np.sqrt((flyby_eccentricity-1)/(flyby_eccentricity+1))
    delta_t_flyby = 2 * (np.sqrt((-flyby_sma**3)/mu_moon) * flyby_eccentricity * np.sinh(eccentric_anomaly) - eccentric_anomaly)
    flyby_final_epoch = flyby_initial_epoch + delta_t_flyby
else:
    flyby_final_epoch = flyby_initial_epoch

# Retrieve moon final state from SPICE kernels - NOTE: it is the same state that gets obtained at the end of arc 1 so consider substituting
final_state_moon_center = spice_interface.get_body_cartesian_state_at_epoch(
    target_body_name=choose_flyby_moon,
    observer_body_name="Jupiter",
    reference_frame_name=global_frame_orientation,
    aberration_corrections="NONE",
    ephemeris_time=flyby_final_epoch)

# Moon state at flyby end
moon_final_position = final_state_moon_center[0:3]
moon_final_velocity = final_state_moon_center[3:6]

# Departure state for the second arc
second_arc_departure_position = flyby_final_position + moon_final_position
second_arc_departure_velocity = flyby_final_velocity + moon_final_velocity
########################################################################################################################

# See flyby - not working
# flyby_initial_state = np.concatenate((flyby_initial_position, flyby_initial_velocity), axis=0)
# flyby_final_state = np.concatenate((flyby_final_position, flyby_final_velocity), axis=0)
# flyby_lambert_history = compute_lambert_targeter_state_history(flyby_initial_state, flyby_final_state,flyby_initial_epoch, flyby_final_epoch)
# plot_trajectory_arc(flyby_lambert_history)

# theta_max = np.arccos(-1/flyby_eccentricity)
# theta_angles = np.linspace(-theta_max,theta_max,300)
# orbit_radii = flyby_sma * (1- flyby_eccentricity**2)/(1 + flyby_eccentricity*np.cos(theta_angles))
# x, y = cartesian_2d_from_polar(orbit_radii, theta_angles)
# fig, ax = plt.subplots(1)
# ax.plot(x, y)
# moon_plot = plt.Circle((0.,0.),moon_radius)
# # ax.set_aspect( 1 )
# ax.add_artist(moon_plot)
# plt.axis('equal')
# plt.show()

# COMPUTE SECOND ARC ####################################################################################################


########################################################################################################################
