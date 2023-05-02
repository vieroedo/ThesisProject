import numpy as np
from JupiterTrajectory_GlobalParameters import *
import class_AerocaptureSemianalyticalModel as ae_analytical
from handle_functions import *

# Load spice kernels
spice_interface.load_standard_kernels()


def get_initial_state_with_flyby(atmosphere_entry_fpa: float,
                                 atmosphere_entry_altitude: float,
                                 B_parameter,
                                 flyby_moon: str,
                                 flyby_epoch: float,
                                 jupiter_arrival_v_inf: float,
                                 perturb_fpa: float = 0.,
                                 perturb_entry_velocity_magnitude: float = 0.,
                                 start_at_entry_interface: bool = False,
                                 # start_at_exit_interface: bool = False,
                                 verbose: bool = False) -> np.ndarray:
    """
    Calculates the initial state given the initial distance and speed, and the arrival fpa.

    The initial state is expressed in Jupiter-centered cartesian coordinates.

    Parameters
    ----------
    atmosphere_entry_fpa : float
        Flight path angle of the spacecraft at the atmosphere interface.
    atmosphere_entry_altitude : float
        Altitude of the spacecraft at the atmosphere interface.
    jupiter_arrival_v_inf : float
        Interplanetary excess velocity in Jupiter frame.
    verbose: bool
        Choose whether to plot analytical arc info or not.

    Returns
    -------
    initial_state_vector : np.ndarray
        The initial state of the vehicle expressed in inertial coordinates.
        :param flyby_moon:
    """

    if perturb_entry_velocity_magnitude != 0. and not start_at_entry_interface:
        warnings.warn('Warning: you are calculating a wrong initial state!'
                      'Entry velocity magnitude perturation might not work well under these conditions.')

    if B_parameter < 0:
        prograde_flyby = False
    else:
        prograde_flyby = True
    B_parameter = abs(B_parameter)

    flyby_moon_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name=flyby_moon,
        observer_body_name="Jupiter",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=flyby_epoch)
    moon_position = flyby_moon_state[0:3]
    moon_velocity = flyby_moon_state[3:6]
    moon_orbital_axis = unit_vector(np.cross(moon_position, moon_velocity))
    moon_eccentricity_vector = eccentricity_vector_from_cartesian_state(flyby_moon_state)
    # moon_orbital_energy = orbital_energy(LA.norm(moon_position), LA.norm(moon_velocity), jupiter_gravitational_parameter)
    moon_sma = galilean_moons_data[flyby_moon]['SMA']
    moon_period = galilean_moons_data[flyby_moon]['Orbital_Period']
    moon_SOI = galilean_moons_data[flyby_moon]['SOI_Radius']
    moon_radius = galilean_moons_data[flyby_moon]['Radius']
    mu_moon = galilean_moons_data[flyby_moon]['mu']



    # epoch_interval = [flyby_epoch-moon_period/2, flyby_epoch+moon_period/2]
    orbital_axis = moon_orbital_axis



    arrival_fpa_deg = atmosphere_entry_fpa + perturb_fpa
    # Problem parameters
    arrival_fpa = np.deg2rad(arrival_fpa_deg)  # rad
    departure_radius = jupiter_SOI_radius
    departure_velocity_norm = jupiter_arrival_v_inf

    # Calculate orbital energy
    initial_orbital_energy = orbital_energy(departure_radius, departure_velocity_norm, jupiter_gravitational_parameter)

    # Calculate arrival radius and speed
    arrival_radius = jupiter_radius + atmosphere_entry_altitude
    arrival_velocity_norm = velocity_from_energy(initial_orbital_energy, arrival_radius, jupiter_gravitational_parameter) + perturb_entry_velocity_magnitude

    # Calculate angular momentum
    angular_momentum_norm = arrival_radius * arrival_velocity_norm * np.cos(arrival_fpa)
    # angular_momentum = z_axis * angular_momentum_norm

    # Calculate other orbit elements
    semilatus_rectum = angular_momentum_norm ** 2 / jupiter_gravitational_parameter
    semimajor_axis = - jupiter_gravitational_parameter / (2 * initial_orbital_energy)
    eccentricity = np.sqrt(1 - semilatus_rectum / semimajor_axis)

    # Set arrival position on x axis
    arrival_position = x_axis * arrival_radius

    # Calculate arrival velocity for the atmospheric entry interface start case
    arr_vel_rotation_matrix = rotation_matrix(orbital_axis, np.pi / 2 - arrival_fpa)
    arrival_velocity = arrival_velocity_norm * \
                       rotate_vectors_by_given_matrix(arr_vel_rotation_matrix, unit_vector(arrival_position))

    aerocapture_analytical_problem = ae_analytical.AerocaptureSemianalyticalModel([0.,0.],
                                                                       atmospheric_entry_initial_position=arrival_position,
                                                                       orbit_datapoints=200,
                                                                       equations_order=2)
    orbital_parameters = [departure_velocity_norm,arrival_fpa]
    aerocapture_analytical_problem.fitness(orbital_parameters)

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
    # Phase angle of the atmospehric entry
    atmospheric_entry_final_phase_angle = aerocapture_problem_parameters[5]

    # second_arc_angular_momentum = atmospheric_exit_velocity_norm * arrival_radius * np.cos(atmospheric_exit_fpa)
    second_arc_angular_momentum = angular_momentum(arrival_radius,atmospheric_exit_velocity_norm,atmospheric_exit_fpa)
    second_arc_orbital_energy = orbital_energy(arrival_radius,atmospheric_exit_velocity_norm,jupiter_gravitational_parameter)

    second_arc_semilatus_rectum = second_arc_angular_momentum ** 2 / jupiter_gravitational_parameter
    second_arc_semimajor_axis = - jupiter_gravitational_parameter / (2 * second_arc_orbital_energy)
    second_arc_eccentricity = np.sqrt(1 - second_arc_semilatus_rectum / second_arc_semimajor_axis)

    second_arc_arrival_velocity_norm = velocity_from_energy(second_arc_orbital_energy,moon_sma,jupiter_gravitational_parameter)

    # >0 cs its going away
    second_arc_arrival_fpa = np.arccos(second_arc_angular_momentum/(moon_sma*second_arc_arrival_velocity_norm))

    rot_matrix = rotation_matrix(orbital_axis,np.pi/2-second_arc_arrival_fpa)
    second_arc_arrival_velocity = rotate_vectors_by_given_matrix(rot_matrix,unit_vector(moon_position)) * second_arc_arrival_velocity_norm

    flyby_initial_velocity_vector = second_arc_arrival_velocity - moon_velocity
    flyby_v_inf_t = LA.norm(flyby_initial_velocity_vector)

    if prograde_flyby:
        flyby_axis = orbital_axis
    else:
        flyby_axis = - orbital_axis

    phi_2_angle = np.arccos(np.dot(unit_vector(-moon_velocity), unit_vector(flyby_initial_velocity_vector)))
    if np.dot(np.cross(-moon_velocity, flyby_initial_velocity_vector), flyby_axis) < 0:
        phi_2_angle = - phi_2_angle + 2 * np.pi

    delta_angle = phi_2_angle - (np.pi - np.arcsin(B_parameter/moon_SOI))
    # delta_angle = (phi_2_angle - np.arcsin(B_parameter/moon_SOI))

    rot_matrix = rotation_matrix(flyby_axis,delta_angle)
    flyby_initial_position = rotate_vectors_by_given_matrix(rot_matrix,unit_vector(-moon_velocity))*moon_SOI



    flyby_alpha_angle = 2 * np.arcsin(1 / np.sqrt(1 + (B_parameter ** 2 * flyby_v_inf_t ** 4) / mu_moon ** 2))
    beta_angle = phi_2_angle + flyby_alpha_angle / 2 - np.pi / 2

    position_rot_angle = 2 * (- delta_angle + beta_angle)
    # position_rot_angle = np.pi - phi_2_angle + beta_angle - np.arcsin(B_parameter/moon_SOI)

    flyby_final_position = rotate_vectors_by_given_matrix(rotation_matrix(flyby_axis, position_rot_angle),
                                                          flyby_initial_position)

    flyby_final_velocity_vector = rotate_vectors_by_given_matrix(rotation_matrix(flyby_axis, flyby_alpha_angle),
                                                                 flyby_initial_velocity_vector)

    # flyby_pericenter = mu_moon / (flyby_v_inf_t ** 2) * (
    #         np.sqrt(1 + (B_parameter ** 2 * flyby_v_inf_t ** 4) / (mu_moon ** 2)) - 1)
    #
    # flyby_orbital_energy = orbital_energy(LA.norm(flyby_initial_position), flyby_v_inf_t, mu_parameter=mu_moon)
    # flyby_sma = - mu_moon / (2 * flyby_orbital_energy)
    # flyby_eccentricity = 1 - flyby_pericenter / flyby_sma
    #
    # true_anomaly_boundary = 2 * np.pi - delta_angle + beta_angle
    # true_anomaly_boundary = true_anomaly_boundary if true_anomaly_boundary < 2 * np.pi else true_anomaly_boundary - 2 * np.pi
    # true_anomaly_range = np.array([-true_anomaly_boundary, true_anomaly_boundary])
    # flyby_elapsed_time = delta_t_from_delta_true_anomaly(true_anomaly_range,
    #                                                      eccentricity=flyby_eccentricity,
    #                                                      semi_major_axis=flyby_sma,
    #                                                      mu_parameter=mu_moon)
    # flyby_final_epoch = flyby_epoch + flyby_elapsed_time
    # flyby_moon_state = spice_interface.get_body_cartesian_state_at_epoch(
    #     target_body_name=flyby_moon,
    #     observer_body_name="Jupiter",
    #     reference_frame_name=global_frame_orientation,
    #     aberration_corrections="NONE",
    #     ephemeris_time=flyby_final_epoch)
    # moon_position = flyby_moon_state[0:3]
    # moon_velocity = flyby_moon_state[3:6]

    fourth_arc_departure_position = flyby_final_position + moon_position
    fourth_arc_departure_velocity = flyby_final_velocity_vector + moon_velocity
    fourth_arc_departure_velocity_norm = LA.norm(fourth_arc_departure_velocity)

    # Calculate post-flyby arc departure flight path angle

    fourth_arc_departure_fpa = np.arcsin(
        np.dot(unit_vector(fourth_arc_departure_position), unit_vector(fourth_arc_departure_velocity)))

    # Calculate post-flyby arc orbital energy
    fourth_arc_orbital_energy = fourth_arc_departure_velocity_norm ** 2 / 2 - \
                                jupiter_gravitational_parameter / LA.norm(fourth_arc_departure_position)

    fourth_arc_angular_momentum = LA.norm(fourth_arc_departure_position) * fourth_arc_departure_velocity_norm * np.cos(
        fourth_arc_departure_fpa)

    fourth_arc_semilatus_rectum = fourth_arc_angular_momentum ** 2 / jupiter_gravitational_parameter
    fourth_arc_semimajor_axis = - jupiter_gravitational_parameter / (2 * fourth_arc_orbital_energy)
    fourth_arc_eccentricity = np.sqrt(1 - fourth_arc_semilatus_rectum / fourth_arc_semimajor_axis)

    second_arc_final_position = moon_position + flyby_initial_position

    # Calculate delta true anomaly spanned by the spacecraft
    first_arc_departure_true_anomaly = - true_anomaly_from_radius(departure_radius, eccentricity, semimajor_axis)
    first_arc_arrival_true_anomaly = - true_anomaly_from_radius(arrival_radius, eccentricity,semimajor_axis)
    first_arc_delta_true_anomaly = first_arc_arrival_true_anomaly - first_arc_departure_true_anomaly

    aerocapture_delta_phase_angle = atmospheric_entry_final_phase_angle

    second_arc_departure_true_anomaly = true_anomaly_from_radius(arrival_radius,second_arc_eccentricity,second_arc_semimajor_axis)
    second_arc_arrival_true_anomaly = true_anomaly_from_radius(moon_sma, second_arc_eccentricity, second_arc_semimajor_axis)
    second_arc_delta_true_anomaly = second_arc_arrival_true_anomaly - second_arc_departure_true_anomaly

    delta_true_anomaly = first_arc_delta_true_anomaly + aerocapture_delta_phase_angle + second_arc_delta_true_anomaly


    second_arc_initial_position = rotate_vectors_by_given_matrix(rotation_matrix(orbital_axis,-second_arc_delta_true_anomaly),unit_vector(second_arc_final_position)) * arrival_radius

    first_arc_final_position =  rotate_vectors_by_given_matrix(rotation_matrix(orbital_axis,-aerocapture_delta_phase_angle),unit_vector(second_arc_initial_position)) * arrival_radius

    first_arc_initial_position = rotate_vectors_by_given_matrix(rotation_matrix(orbital_axis,-first_arc_delta_true_anomaly),unit_vector(first_arc_final_position)) * jupiter_SOI_radius

    # circ_vel_at_atm_entry = np.sqrt(
    #     jupiter_gravitational_parameter / (jupiter_radius + atmosphere_entry_altitude))
    #
    # # Prints for debugging
    # if verbose:
    #     print('\nAtmospheric entry (pre-aerocapture) analytical conditions:\n'
    #           f'- altitude: {atmosphere_entry_altitude / 1e3} km\n'
    #           f'- velocity: {arrival_velocity_norm / 1e3:.3f} km/s\n'
    #           f'- ref circular velocity: {circ_vel_at_atm_entry / 1e3:.3f} km/s\n'
    #           f'- flight path angle: {atmosphere_entry_fpa} deg\n'
    #           f'- eccentricity: {eccentricity:.10f} ')
    #
    # # Calculate delta true anomaly spanned by the spacecraft
    # departure_true_anomaly = true_anomaly_from_radius(departure_radius, eccentricity, semimajor_axis)
    # arrival_true_anomaly = true_anomaly_from_radius(arrival_radius, eccentricity,semimajor_axis)
    # delta_true_anomaly = arrival_true_anomaly - departure_true_anomaly

    # Calculate the departure position of the spacecraft
    pos_rotation_matrix = rotation_matrix(orbital_axis, -delta_true_anomaly)
    departure_position = rotate_vectors_by_given_matrix(pos_rotation_matrix, unit_vector(
        moon_position)) * departure_radius

    # Calculate departure fpa (useful to obtain departure velocity)
    departure_fpa = - np.arccos(
        angular_momentum_norm / (departure_radius * departure_velocity_norm))

    # Calculate departure velocity
    vel_rotation_matrix = rotation_matrix(orbital_axis, np.pi / 2 - departure_fpa)
    departure_velocity = rotate_vectors_by_given_matrix(vel_rotation_matrix, unit_vector(
        departure_position)) * departure_velocity_norm

    # Build the initial state vector
    initial_state_vector = np.concatenate((departure_position, departure_velocity))

    # Build the atmospheric entry interface state vector
    if start_at_entry_interface:
        initial_state_vector = np.concatenate((arrival_position, arrival_velocity))

    # Print the state vector for debugging
    if verbose:
        print('\nDeparture state:')
        print(f'{list(initial_state_vector)}')
    # return initial_state_vector


    arcs_dictionary = {
        'First': (
        jupiter_SOI_radius, first_arc_final_position, eccentricity, semimajor_axis,
        arrival_fpa),
        'Second': (
        LA.norm(atmospheric_entry_final_position), second_arc_final_position, second_arc_eccentricity, second_arc_semimajor_axis,
        second_arc_arrival_fpa),
        # 'Third': (
        # moon_sma, third_arc_final_position, third_arc_eccentricity, third_arc_semimajor_axis,
        # third_arc_arrival_fpa),
    }
    number_of_epochs_to_plot = 200
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
        arc_arrival_true_anomaly = np.sign(arc_arrival_fpa) * true_anomaly_from_radius(arc_arrival_radius,
                                                                                       arc_eccentricity,
                                                                                       arc_semimajor_axis)
        arc_departure_true_anomaly = np.sign(arc_arrival_fpa) * true_anomaly_from_radius(arc_departure_radius,
                                                                                         arc_eccentricity,
                                                                                         arc_semimajor_axis)

        # Calculate phase angle of first arc
        arc_phase_angle = arc_arrival_true_anomaly - arc_departure_true_anomaly

        # End and start conditions w.r.t. x axis are the same, so we take the position at the node
        arc_arrival_position_angle_wrt_x_axis = np.arccos(np.dot(unit_vector(arc_arrival_position), x_axis))

        # Check if such angle is greater than np.pi or not, and set it accordingly
        if np.dot(np.cross(x_axis, arc_arrival_position), z_axis) < 0:
            arc_arrival_position_angle_wrt_x_axis = - arc_arrival_position_angle_wrt_x_axis + 2 * np.pi

        # Calculate coordinate points of the first arc to be plotted
        final_orbit_true_anomaly_vector = np.linspace(arc_departure_true_anomaly, arc_arrival_true_anomaly,
                                                      arc_number_of_points)
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

    final_orbit_number_of_points = 2 * number_of_epochs_to_plot

    final_orbit_eccentricity = fourth_arc_eccentricity
    final_orbit_semimajor_axis = fourth_arc_semimajor_axis
    final_orbit_reference_position = moon_position
    final_orbit_reference_velocity = fourth_arc_departure_velocity

    position_multiplier = LA.norm(final_orbit_reference_velocity) ** 2 - jupiter_gravitational_parameter / LA.norm(
        final_orbit_reference_position)
    velocity_multiplier = np.dot(final_orbit_reference_position, final_orbit_reference_velocity)
    eccentricity_vector = 1 / jupiter_gravitational_parameter * (
                position_multiplier * final_orbit_reference_position - velocity_multiplier * final_orbit_reference_velocity)

    final_orbit_pericenter_angle_wrt_x_axis = np.arccos(np.dot(unit_vector(eccentricity_vector), x_axis))

    # Check if such angle is greater than np.pi or not, and set it accordingly
    if np.dot(np.cross(x_axis, eccentricity_vector), z_axis) < 0:
        final_orbit_pericenter_angle_wrt_x_axis = - final_orbit_pericenter_angle_wrt_x_axis + 2 * np.pi

    # if escape_orbit or flyby_induced_escape:
    #     true_anomaly_limit = np.arccos(-1 / final_orbit_eccentricity) - 0.1
    #     initial_true_anomaly = 0.
    #     if flyby_induced_escape:
    #         initial_true_anomaly = true_anomaly_from_radius(LA.norm(p_ae_moon_fb_position), final_orbit_eccentricity,
    #                                                         final_orbit_semimajor_axis)
    #     final_orbit_true_anomaly_vector = np.linspace(initial_true_anomaly, true_anomaly_limit,
    #                                                   final_orbit_number_of_points)
    # else:
    final_orbit_true_anomaly_vector = np.linspace(-np.pi, np.pi, final_orbit_number_of_points)

    final_orbit_radius_vector = radius_from_true_anomaly(final_orbit_true_anomaly_vector, final_orbit_eccentricity,
                                                         final_orbit_semimajor_axis)
    final_orbit_true_anomaly_plot = np.linspace(
        final_orbit_pericenter_angle_wrt_x_axis + final_orbit_true_anomaly_vector[0],
        final_orbit_pericenter_angle_wrt_x_axis + final_orbit_true_anomaly_vector[-1], final_orbit_number_of_points)

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

    # draw second moon
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x_0 = flyby_moon_state[0]
    y_0 = flyby_moon_state[1]
    z_0 = flyby_moon_state[2]
    x = x_0 + moon_radius * np.cos(u) * np.sin(v)
    y = y_0 + moon_radius * np.sin(u) * np.sin(v)
    z = z_0 + moon_radius * np.cos(v)
    ax.plot_wireframe(x, y, z, color="b")

    ########################################################################################################################
    # RE-ENTRY PLOTS  ######################################################################################################
    ########################################################################################################################

    # fpa_vector = np.linspace(arrival_fpa, atmospheric_exit_fpa, 200)
    #
    # altitude_vector = ae_radii - jupiter_radius
    # # atmospheric_entry_trajectory_altitude(fpa_vector, atmospheric_entry_fpa, density_at_atmosphere_entry,
    # #                                                     reference_density, ballistic_coefficient_times_g_acc,
    # #                                                     atmospheric_entry_g_acc, jupiter_beta_parameter)
    # downrange_vector = tau_linspace * np.sqrt(
    #     jupiter_scale_height / (atmospheric_entry_altitude + jupiter_radius)) * jupiter_radius
    # # atmospheric_entry_trajectory_distance_travelled(fpa_vector, atmospheric_entry_fpa, effective_entry_fpa, scale_height)
    #
    # fig2, ax2 = plt.subplots(figsize=(5, 6))
    # ax2.plot(downrange_vector / 1e3, altitude_vector / 1e3)
    # ax2.set(xlabel='downrange [km]', ylabel='altitude [km]')

    # ax2.plot(fpa_vector,downrange_vector)

    # ax2.set_aspect('equal', 'box')
    plt.show()
