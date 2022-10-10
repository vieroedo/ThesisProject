from handle_functions import *


# Set flyby epoch -> edit this to be initial epoch!!
first_arc_arrival_epoch_days = 11293.  # days

# PROBLEM PARAMETERS ###################################################################################################
# (Consider moving to related script in case other ones using this library need different parameters)

# Atmospheric entry conditions
arrival_pericenter_altitude = 2000e3  # m (DO NOT CHANGE - consider changing only with valid and sound reasons)
flight_path_angle_at_atmosphere_entry = -3.  # degrees

# Jupiter arrival conditions
interplanetary_arrival_velocity_in_jupiter_frame = 5600 # m/s
delta_angle_from_hohmann_trajectory = 0.92  # degrees

# Trajectory geometry
choose_flyby_moon = 'Io'
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

# First arc initial known conditions
first_arc_departure_radius = jupiter_SOI_radius
first_arc_departure_velocity = interplanetary_arrival_velocity_in_jupiter_frame
radius_velocity_vectors_angle = np.pi - delta_angle_from_hohmann_trajectory

second_arc_arrival_radius = jupiter_radius + arrival_pericenter_altitude

moon_radius = galilean_moons_data[choose_flyby_moon]['Radius']
moon_SOI_radius = galilean_moons_data[choose_flyby_moon]['SOI_Radius']
mu_moon = galilean_moons_data[choose_flyby_moon]['mu']

sigma_angles = np.linspace(0,2*np.pi,400)
function_values = np.zeros(len(sigma_angles))

for i, sigma_angle in enumerate(sigma_angles):
    fpa_a = calculate_fpa_from_flyby_geometry(sigma_angle=sigma_angle,
                                              arc_1_initial_velocity=first_arc_departure_velocity,
                                              arc_1_initial_radius=first_arc_departure_radius,
                                              delta_hoh=delta_angle_from_hohmann_trajectory,
                                              arc_2_final_radius=second_arc_arrival_radius,
                                              mu_moon=mu_moon,
                                              moon_SOI=moon_SOI_radius,
                                              moon_state_at_flyby=moon_flyby_state)
    f_a = fpa_a - flight_path_angle_at_atmosphere_entry

    function_values[i] = f_a

plt.axhline(y=0)
plt.plot(sigma_angles*180/np.pi, function_values)
plt.show()