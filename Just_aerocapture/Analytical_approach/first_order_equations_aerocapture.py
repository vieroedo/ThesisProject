from handle_functions import *

def calculate_max_loads_point(entry_fpa, entry_velocity, lift_over_drag_ratio):
    # Calculate loads on the spacecraft
    term_1 = lift_over_drag_ratio / (2 * (1 + np.cos(entry_fpa)))
    term_2 = 1 - np.sqrt(1 + (4 * np.sin(entry_fpa) ** 2) / (lift_over_drag_ratio ** 2))
    fpa_max_acceleration = 2 * np.arctan(term_1 * term_2)

    g_earth = 9.81  # m/s^2
    av_max_over_g = 1/jupiter_scale_height * entry_velocity ** 2 / (
                g_earth * lift_over_drag_ratio) \
                    * (np.cos(fpa_max_acceleration) - np.cos(entry_fpa)) \
                    * np.exp(- 2 * (fpa_max_acceleration - entry_fpa) / lift_over_drag_ratio)

    a_total_max_over_g = av_max_over_g * np.sqrt(1 + lift_over_drag_ratio ** 2)

    return fpa_max_acceleration, av_max_over_g, a_total_max_over_g


def calculate_fpa_boundaries(entry_fpa):
    exit_fpa = -entry_fpa
    minimum_altitude_fpa = 0.
    return entry_fpa, minimum_altitude_fpa, exit_fpa


def first_order_approximation_aerocapture(current_fpa, entry_fpa, minimum_altitude_fpa, entry_radius, entry_velocity,
                                          C_L = vehicle_cl, C_D = vehicle_cd, capsule_mass = vehicle_mass,
                                          capsule_radius = vehicle_radius, capsule_area = vehicle_reference_area):

    entry_altitude = entry_radius - jupiter_radius
    entry_g_acc = jupiter_gravitational_parameter / (entry_radius ** 2)

    # Atmosphere model coefficients (exponential atmosphere)
    jupiter_atmosphere_beta_parameter = 1 / jupiter_scale_height
    entry_density = jupiter_atmosphere_exponential(entry_altitude)

    # Capsule properties and coefficients
    #  make L/D = 0.5 -> https://www.researchgate.net/publication/238790363_Aerodynamic_Control_on_a_Lunar_Return_Capsule_using_Trim-Flaps
    lift_over_drag_ratio = C_L / C_D
    capsule_weight = capsule_mass * entry_g_acc

    # # Atmosphere exit fpa
    # exit_fpa = - entry_fpa

    # # Atmosphere exit velocity
    # exit_velocity = entry_velocity * np.exp(2 * entry_fpa / lift_over_drag_ratio)


    # Check vehicle doesn't go too deep into Jupiter atmosphere (not below zero-level altitude)
    ballistic_coefficient_times_g_acc = capsule_weight / (capsule_area * C_L)
    minimum_altitude_condition_value = entry_g_acc * jupiter_1bar_density \
                                       / (4 * jupiter_atmosphere_beta_parameter * np.sin(entry_fpa / 2) ** 2)
    if ballistic_coefficient_times_g_acc > minimum_altitude_condition_value:
        raise Warning(
            'Minimum altitude is below zero!! Trajectory is theoretically possible, but heat loads can be prohibitive.')

    # # Angle at a certain time idk
    # gamma_angle = ...

    effective_entry_fpa = - np.arccos(
        np.cos(entry_fpa) - entry_density * entry_g_acc / (
                2 * jupiter_atmosphere_beta_parameter) * 1 / ballistic_coefficient_times_g_acc)

    minimum_altitude = atmospheric_entry_trajectory_altitude(minimum_altitude_fpa, entry_fpa, entry_density,
                                                             jupiter_1bar_density, ballistic_coefficient_times_g_acc,
                                                             entry_g_acc, jupiter_atmosphere_beta_parameter)
    # pressure_at_minimum_altitude = ...

    # Travelled distance (assumed at surface)
    distance_travelled = atmospheric_entry_trajectory_distance_travelled(current_fpa,
                                                                         entry_fpa,
                                                                         effective_entry_fpa,
                                                                         jupiter_scale_height)


    range_angle = distance_travelled / jupiter_radius

    # atmosph_entry_rot_matrix = rotation_matrix(orbit_axis, range_angle)
    # atmospheric_entry_final_position = rotate_vectors_by_given_matrix(atmosph_entry_rot_matrix, pre_ae_arrival_position)

    # fpa_max_acceleration,  av_max_over_g, a_total_max_over_g = calculate_max_loads_point(
    #     entry_fpa, entry_velocity, lift_over_drag_ratio)


    current_altitude = atmospheric_entry_trajectory_altitude(current_fpa,entry_fpa,entry_density,
                                                             jupiter_1bar_density,ballistic_coefficient_times_g_acc,
                                                             entry_g_acc,jupiter_atmosphere_beta_parameter)
    radius = jupiter_radius + current_altitude

    velocity = entry_velocity * np.exp(- (current_fpa - entry_fpa) / lift_over_drag_ratio)

    flight_path_angle = current_fpa

    density = jupiter_atmosphere_exponential(current_altitude)

    g_earth = 9.81
    drag_acc = 1 / jupiter_scale_height * entry_velocity ** 2 / (lift_over_drag_ratio) \
               * (np.cos(current_fpa) - np.cos(entry_fpa)) * np.exp(- 2 * (current_fpa - entry_fpa) / lift_over_drag_ratio)

    lift_acc = lift_over_drag_ratio * drag_acc
    # Heat loads
    wall_heat_flux = 0.6556E-8 * (density/capsule_radius)**0.5 * velocity**3

    return_values = (radius, velocity, flight_path_angle, density, drag_acc, lift_acc, wall_heat_flux, range_angle)
    additional_values = (minimum_altitude,)

    return return_values, additional_values
