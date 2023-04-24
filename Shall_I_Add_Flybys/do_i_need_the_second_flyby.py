from JupiterTrajectory_GlobalParameters import *
# import CapsuleEntryUtilities as Util
from handle_functions import *
from tudatpy.kernel.interface import spice_interface
from class_AerocaptureSemianalyticalModel import *

# Load spice kernels
spice_interface.load_standard_kernels()
arrival_pericenter_altitude = atmospheric_entry_altitude


do_regula_falsi_function_debugging = False

choose_equations_order = 2 # always

# PROBLEM PARAMETERS ###################################################################################################
# (Consider moving to related script in case other ones using this library need different parameters)

# Atmospheric entry conditions
flight_path_angle_at_atmosphere_entry = np.deg2rad(-2.5)  # input value in: deg

# Jupiter arrival conditions
interplanetary_arrival_velocity_in_jupiter_frame = 5600  # m/s

# Post-aerocapture flyby parameters
p_ae_flyby_moon = 'Io'
MJD200_date = 62502  # 01/01/2030 62502

# Plotting parameters
number_of_epochs_to_plot = 200

########################################################################################################################

# Parameters reprocessing ##############################################################################################
first_arc_number_of_points = number_of_epochs_to_plot
second_arc_number_of_points = number_of_epochs_to_plot

J2000_date = MJD200_date - 51544
flyby_epoch = J2000_date * constants.JULIAN_DAY
########################################################################################################################

moons_and_energies = dict()

for p_ae_flyby_moon in galilean_moons_data.keys():

    moon_flyby_state = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name=p_ae_flyby_moon,
            observer_body_name="Jupiter",
            reference_frame_name=global_frame_orientation,
            aberration_corrections="NONE",
            ephemeris_time=flyby_epoch)


    # Calculate first arc quantities and the initial state vector
    pre_ae_departure_radius = jupiter_SOI_radius
    pre_ae_departure_velocity_norm = interplanetary_arrival_velocity_in_jupiter_frame

    pre_ae_orbital_energy = orbital_energy(pre_ae_departure_radius, pre_ae_departure_velocity_norm, jupiter_gravitational_parameter)

    pre_ae_arrival_radius = jupiter_radius + arrival_pericenter_altitude
    pre_ae_arrival_velocity_norm = velocity_from_energy(pre_ae_orbital_energy, pre_ae_arrival_radius, jupiter_gravitational_parameter)

    pre_ae_angular_momentum_norm = pre_ae_arrival_radius * pre_ae_arrival_velocity_norm * np.cos(flight_path_angle_at_atmosphere_entry)
    pre_ae_angular_momentum = z_axis * pre_ae_angular_momentum_norm

    pre_ae_semilatus_rectum = pre_ae_angular_momentum_norm ** 2 / jupiter_gravitational_parameter
    pre_ae_semimajor_axis = - jupiter_gravitational_parameter / (2 * pre_ae_orbital_energy)
    pre_ae_eccentricity = np.sqrt(1 - pre_ae_semilatus_rectum / pre_ae_semimajor_axis)

    pre_ae_arrival_position = x_axis * pre_ae_arrival_radius

    # initial_state_vector = Util.get_initial_state(flight_path_angle_at_atmosphere_entry, atmospheric_entry_altitude,
    #                                               interplanetary_arrival_velocity_in_jupiter_frame, verbose=True)

    # AEROCAPTURE ##########################################################################################################

    # Entry conditions (from arcs 1 and 2)
    atmospheric_entry_fpa = flight_path_angle_at_atmosphere_entry
    atmospheric_entry_velocity_norm = pre_ae_arrival_velocity_norm
    atmospheric_entry_altitude = arrival_pericenter_altitude
    # atmospheric_entry_g_acc = jupiter_gravitational_parameter / (pre_ae_arrival_radius ** 2)
    atmospheric_entry_initial_position = pre_ae_arrival_position


    aerocapture_analytical_problem = AerocaptureSemianalyticalModel([0,1000],atmospheric_entry_initial_position,number_of_epochs_to_plot,choose_equations_order)
    aerocapture_analytical_problem.fitness([interplanetary_arrival_velocity_in_jupiter_frame, atmospheric_entry_fpa])

    # aerocapture_parameters:
    # [atmospheric_exit_fpa, atmospheric_exit_velocity_norm, final_distance_travelled, minimum_altitude, atmospheric_entry_final_position]
    aerocapture_problem_parameters = aerocapture_analytical_problem.aerocapture_parameters_function()

    # Atmosphere exit fpa
    atmospheric_exit_fpa = aerocapture_problem_parameters[0]
    # Atmosphere exit velocity
    atmospheric_exit_velocity_norm = aerocapture_problem_parameters[1]
    # Minimum altitude
    minimum_altitude = aerocapture_problem_parameters[3]
    # Travelled distance (assumed at surface)
    final_distance_travelled = aerocapture_problem_parameters[2]
    # Final position after aerocapture
    atmospheric_entry_final_position = aerocapture_problem_parameters[4]

    # Aerocapture cartesian states and dependent variables
    entry_cartesian_states = np.vstack(list(aerocapture_analytical_problem.get_cartesian_state_history().values()))
    dependent_variables = np.vstack(list(aerocapture_analytical_problem.get_dependent_variables_history().values()))
    # [ae_fpas, ae_velocities, ae_radii, ae_densities, ae_drag, ae_lift, ae_wall_hfx, ae_range_angles]
    ae_fpas = dependent_variables[:,0]
    ae_velocities = dependent_variables[:,1]
    ae_radii = dependent_variables[:,2]
    ae_densities = dependent_variables[:,3]
    ae_drag = dependent_variables[:,4]
    ae_lift = dependent_variables[:,5]
    ae_wall_hfx = dependent_variables[:,6]
    ae_range_angles = dependent_variables[:,7]


    print('\n Atmosphere exit conditions:\n'
          f'- exit velocity: {atmospheric_exit_velocity_norm/1e3:.3f} km/s')

    p_ae_departure_velocity_norm = atmospheric_exit_velocity_norm
    p_ae_departure_fpa = atmospheric_exit_fpa
    p_ae_departure_radius = pre_ae_arrival_radius
    p_ae_departure_position = atmospheric_entry_final_position

    p_ae_orbital_energy = orbital_energy(p_ae_departure_radius, p_ae_departure_velocity_norm, jupiter_gravitational_parameter)

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
        # interval_left_boundary_a = p_ae_moon_fb_radius
        # interval_right_boundary_b = p_ae_moon_fb_SOI_radius
        #
        # desired_value = jupiter_radius + 2000e3
        #
        # # DEBUG #############
        # if do_regula_falsi_function_debugging:
        #     radii = np.linspace(p_ae_moon_fb_radius, p_ae_moon_fb_SOI_radius, 500)
        #     function_values = np.zeros(len(radii))
        #     for i, chosen_radius in enumerate(radii):
        #         rp_function = calculate_orbit_pericenter_from_flyby_pericenter(flyby_rp=chosen_radius,
        #                                                                        arc_departure_position=p_ae_arrival_position,
        #                                                                        flyby_initial_velocity_vector=p_ae_fb_initial_velocity,
        #                                                                        mu_moon=p_ae_moon_fb_mu,
        #                                                                        moon_flyby_state=p_ae_moon_fb_state)
        #
        #         f_function = rp_function - desired_value
        #         function_values[i] = f_function
        #
        #     plt.axhline(y=0)
        #     plt.plot(radii, function_values)
        #     plt.show()
        #     quit()
        #
        # #####################


        # a_int = interval_left_boundary_a
        # b_int = interval_right_boundary_b
        #
        # tolerance = 1e-7
        # max_iter = 1000
        #
        # c_point, f_c, i = regula_falsi_illinois((a_int, b_int), calculate_orbit_pericenter_from_flyby_pericenter,
        #                                         desired_value, tolerance, max_iter,
        #                                         arc_departure_position=p_ae_arrival_position,
        #                                         flyby_initial_velocity_vector=p_ae_fb_initial_velocity,
        #                                         mu_moon=p_ae_moon_fb_mu,
        #                                         moon_flyby_state=p_ae_moon_fb_state
        #                                         )

        # Found root
        # flyby_pericenter = c_point
        flyby_pericenter = p_ae_moon_fb_radius
        # orbit_pericenter = f_c + desired_value


        # Debugging
        # print(f'\nNumber of iterations second time: {i}')
        print(f'Second flyby pericenter altitude: {(flyby_pericenter - p_ae_moon_fb_radius) / 1e3:.3f} km')
        # print(f'Orbit pericenter altitude result of root finder: {(orbit_pericenter-jupiter_radius)/1e3:.3f} deg')

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




        moon_radius = galilean_moons_data[p_ae_flyby_moon]['Radius']
        moon_SOI_radius = galilean_moons_data[p_ae_flyby_moon]['SOI_Radius']
        mu_moon = galilean_moons_data[p_ae_flyby_moon]['mu']
        moon_velocity = moon_flyby_state[3:6]

        flyby_v_inf_t = p_ae_fb_initial_velocity_norm

        v_c_surf = np.sqrt(mu_moon / moon_radius)

        delta_energy_max = 2 * LA.norm(moon_velocity) * flyby_v_inf_t / (1 + flyby_v_inf_t ** 2 / v_c_surf ** 2)
        moons_and_energies[p_ae_flyby_moon] = delta_energy_max

        final_arc_minimum_orbital_energy = p_ae_orbital_energy - delta_energy_max




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
    # print('\nAtmospheric entry trajectory parameters:')
    # print(f'- Max acceleration on the spacecraft: {0.000:.3f} g  (1g = 9.81 m/s^2)')
    # print(f'- Stagnation point peak heat flux: {max(ae_wall_hfx)}')
    # print('- Integrated heat load: ...')
    # print(f'- Minimum altitude: {minimum_altitude/1e3:.3f} km')
    # print(f'- Horizontal distance travelled: {final_distance_travelled/1e3:.3f} km')


    # Print fourth arc quantities for debugging
    print(f'\n\nPost-aerocapture flyby moon: {p_ae_flyby_moon}')
    if p_ae_moon_fb_sma < p_ae_apocenter:
        print(f'Post-aerocapture flyby pericenter altitude: {(flyby_pericenter-p_ae_moon_fb_radius)/1e3:.3f} km')
        # print(f'Final orbit pericenter altitude: {(orbit_pericenter-jupiter_radius)/1e3:.3f} km')
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
    print(f'Final orbit orbital energy CHECKCKCK: {final_arc_minimum_orbital_energy / 1e3:.3f} kJ')
    # print(f'pre/post a-e delta orbital energy: {abs(pre_ae_orbital_energy-p_ae_orbital_energy)/1e3:.3f} kJ')
    # print(f'pre-post flyby delta orbital energy: {abs(p_ae_orbital_energy - final_orbit_orbital_energy) / 1e3:.3f} kJ')
    print(f'Maximum delta energy achievable with {p_ae_flyby_moon}: {delta_energy_max/1e3:.3f} kJ/kg')


print('\n\n')
print(moons_and_energies)
print('\n')
print(p_ae_orbital_energy)
