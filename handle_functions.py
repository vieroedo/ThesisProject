import numpy as np
import math
import numpy.linalg as LA
from matplotlib import pyplot as plt
import warnings
from typing import Callable

# TUDAT imports
from tudatpy.kernel.astro import two_body_dynamics
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.astro import element_conversion

# Load spice kernels
spice_interface.load_standard_kernels()
# spice_interface.get_average_radius('Jupiter')
# Ref system axes
x_axis = np.array([1., 0., 0.])
y_axis = np.array([0., 1., 0.])
z_axis = np.array([0., 0., 1.])

# Global values of the problem
jupiter_scale_height = 27e3  # m      https://web.archive.org/web/20111013042045/http://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html
jupiter_1bar_density = 0.16  # kg/m^3
jupiter_mass = 1898.6e24  # kg
jupiter_radius = spice_interface.get_average_radius('Jupiter')  #69911e3  # m
jupiter_SOI_radius = 48.2e9  # m
central_body_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Jupiter')
global_frame_orientation = 'ECLIPJ2000'

# orbital and physical data of galilean moons in SI units
# (entries: 'SMA' 'Mass' 'Radius' 'Orbital_Period' 'SOI_Radius' 'mu' 'g_0')
galilean_moons_data = {
                       'Io': {'SMA': 421.77e6,
                              'Mass': 893.3e20,
                              'Radius': 1821.3e3,
                              'Orbital_Period': 1.769138 * constants.JULIAN_DAY,
                              'SOI_Radius': 7836e3,
                              'mu': 893.3e20 * constants.GRAVITATIONAL_CONSTANT,
                              'g_0': 893.3e20 * constants.GRAVITATIONAL_CONSTANT / (1821.3e3**2)},
                       'Europa': {'SMA': 671.08e6,
                                  'Mass': 479.7e20,
                                  'Radius': 1565e3,
                                  'Orbital_Period': 3.551810 * constants.JULIAN_DAY,
                                  'SOI_Radius': 9723e3,
                                  'mu': 479.7e20 * constants.GRAVITATIONAL_CONSTANT,
                                  'g_0': 479.7e20 * constants.GRAVITATIONAL_CONSTANT / (1565e3**2)},
                       'Ganymede': {'SMA': 1070.4e6,
                                    'Mass': 1482e20,
                                    'Radius': 2634e3,
                                    'Orbital_Period': 7.154553 * constants.JULIAN_DAY,
                                    'SOI_Radius': 24351e3,
                                    'mu': 1482e20 * constants.GRAVITATIONAL_CONSTANT,
                                    'g_0': 1482e20 * constants.GRAVITATIONAL_CONSTANT / (2634e3**2)},
                       'Callisto': {'SMA': 1882.8e6,
                                    'Mass': 1076e20,
                                    'Radius': 2403e3,
                                    'Orbital_Period': 16.689018 * constants.JULIAN_DAY,
                                    'SOI_Radius': 37684e3,
                                    'mu': 1076e20 * constants.GRAVITATIONAL_CONSTANT,
                                    'g_0': 1076e20 * constants.GRAVITATIONAL_CONSTANT / (2403e3**2)}
                       }

# Jupiter atmosphere exponential layered model
# (T_0, h_0, rho_0, alpha) from lower to higher
jupiter_atmosphere_model = {
    'layer_1': [425.0, -132e3, 1.5, -0.56e3],
    'layer_2': [100.0,   50e3, 2e-2, 2.7e3],
    'layer_3': [200.0,  320e3, 2e-7, 0.9067e3],
    'layer_4': [950.0, 1000e3, 3e-11, np.inf]
}

# for moon in galilean_moons_data.keys():
#     g_0 = 'g_0'
#     print(f'g_0 of {moon}: {galilean_moons_data[moon][g_0]:.3f} m/s^2')


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    vec_shape = np.shape(vector)
    if len(vec_shape) == 2:
        norms = np.linalg.norm(vector, axis=1)
        final_vector = np.zeros(vec_shape)
        for i in range(len(vector[:,0])):
            final_vector[i,:] = vector[i,:]/norms[i]
        return final_vector
    return vector / np.linalg.norm(vector)


def rotation_matrix(axis, theta):
    """
    Obtained from https://stackoverflow.com/questions/6802577/rotation-of-3d-vector

    Returns the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate_vectors_by_given_matrix(rot_matrix: np.ndarray, vector: np.ndarray):
    """Returns a vector (or array of vectors) rotated according to the chosen rotation matrix"""

    vector_shape = np.shape(vector)

    if len(vector_shape) == 1:
        if len(vector) != 3:
            raise Exception('Vector for rotation must have three components!')
        return (rot_matrix @ vector.reshape(3, 1)).reshape(3)

    elif len(vector_shape) == 2:
        vector = vector.T  # makes the vector with shape (3,n)
        if len(vector[:, 0]) != 3:
            raise Exception('Vector for rotation must have three components!')

        number_of_vectors = len(vector[0, :])
        rotated_vectors = np.zeros((3, number_of_vectors))

        for i in range(number_of_vectors):
            rotated_vectors[:, i] = rot_matrix @ vector[:, i]
        rotated_vectors = rotated_vectors.T  # shape of rotated_vectors is now (n,3)
        return rotated_vectors

    else:
        raise Exception('Wrong input vector dimensions. '
                        'Vector must be either a 1-D array with 3 entries, '
                        'or a 2-D array of vectors with shape (n,3) where "n" is the number of vectors')


def cartesian_2d_from_polar(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


########################################################################################################################
# ASTRODYNAMICS HELPER FUNCTIONS #######################################################################################
########################################################################################################################


def orbital_energy(radius: float, velocity: float):
    return velocity**2/2 - central_body_gravitational_parameter/radius


def velocity_from_energy(energy: float, radius: float):
    return np.sqrt(2*(energy + central_body_gravitational_parameter/radius))


def true_anomaly_from_radius(radius,eccentricity,sma):
    """ WARNING: the solutions of ths function are 2! +theta and -theta"""
    theta = np.arccos(np.clip(1/eccentricity * (sma*(1-eccentricity**2)/radius - 1), -1, 1))
    return theta


def radius_from_true_anomaly(true_anomaly, eccentricity, sma, planet_SoI = jupiter_SOI_radius):
    e = eccentricity
    theta = true_anomaly
    if e>1:
        # radius = np.where(theta < np.arccos(-1/e), sma * (1 - e**2) / (1 + e*np.cos(theta)), planet_SoI)
        radius = sma * (1 - e**2) / (1 + e*np.cos(theta))

    else:
        radius = sma * (1 - e**2) / (1 + e*np.cos(theta))
    return radius


def delta_t_from_delta_true_anomaly(true_anomaly_range: np.ndarray,
                                    eccentricity: float,
                                    semi_major_axis: float,
                                    mu_parameter: float) -> float:
    """

    :param true_anomaly_range: vector containing initial and final true anomalies (interval width must be < 2*pi)
    :param eccentricity: eccentricity of the orbit
    :param semi_major_axis: sma of the orbit. If the orbit is parabolic, it indicates the p parameter
    :param mu_parameter: central body grav parameter
    :return: delta t between the two true anomalies
    """
    if not len(true_anomaly_range) == 2:
        raise Exception('The true_anomaly_range must be a ndarray with size (2,)')

    if eccentricity > 1:
        arctanh_argument = np.tan(true_anomaly_range/2) * np.sqrt((eccentricity - 1)/(eccentricity + 1))
        eccentric_anomaly_range = 2 * np.arctanh(arctanh_argument)
        mean_anomaly_range = eccentricity * np.sinh(eccentric_anomaly_range) - eccentric_anomaly_range
        angular_velocity = np.sqrt(mu_parameter/-semi_major_axis**3)
    elif eccentricity == 1:
        p_parameter = semi_major_axis
        mean_anomaly_range = 1/2 * (np.tan(true_anomaly_range/2) + np.tan(true_anomaly_range/2)**3/3)
        angular_velocity =  np.sqrt(mu_parameter/p_parameter**3)
        # warnings.warn('Not implemented for parabolic orbits lol')
    else:
        arctan_argument = np.tan(true_anomaly_range / 2) * np.sqrt((1-eccentricity) / (1+eccentricity))
        eccentric_anomaly_range = 2 * np.arctan(arctan_argument)
        mean_anomaly_range = eccentric_anomaly_range - eccentricity * np.sin(eccentric_anomaly_range)
        angular_velocity = np.sqrt(mu_parameter/semi_major_axis**3)

    time_range = mean_anomaly_range / angular_velocity

    return time_range[1] - time_range[0]


def moon_circular_2d_state(epoch: float, choose_moon: str) -> np.ndarray((6,)):
    orbital_radius = galilean_moons_data[choose_moon]['SMA']
    orbital_period = galilean_moons_data[choose_moon]['Orbital_Period']

    true_anomaly = epoch * 2 * np.pi / orbital_period
    k_value = int(true_anomaly/2*np.pi)
    true_anomaly = true_anomaly - 2*k_value*np.pi

    x_pos, y_pos = cartesian_2d_from_polar(orbital_radius, true_anomaly)
    z_pos = 0.

    moon_velocity = np.sqrt(central_body_gravitational_parameter/orbital_radius)

    x_vel, y_vel = cartesian_2d_from_polar(moon_velocity, true_anomaly+np.pi/2)
    z_vel = 0.

    return np.array([x_pos, y_pos, z_pos, x_vel, y_vel, z_vel])


def compute_lambert_targeter_state_history(
        initial_state,
        final_state,
        departure_epoch,
        arrival_epoch,
        central_body_grav_parameter=central_body_gravitational_parameter,
        number_of_epochs: int = 200):
    """
    Computes the lambert problem and returns the history of states
    :param initial_state:
    :param final_state:
    :param departure_epoch:
    :param arrival_epoch:
    :param central_body_grav_parameter:
    :param number_of_epochs:
    :return:
    """
    # Lambert targeter initialisation
    lambertTargeter = two_body_dynamics.LambertTargeterIzzo(
        initial_state[:3], final_state[:3], arrival_epoch - departure_epoch, central_body_grav_parameter)

    # lambertTargeter = two_body_dynamics.LambertTargeterGooding(
    #     initial_state[:3], final_state[:3], arrival_epoch - departure_epoch, central_body_gravitational_parameter)

    # Compute initial Cartesian state of Lambert arc
    lambert_arc_initial_state = initial_state
    lambert_arc_initial_state[3:] = lambertTargeter.get_departure_velocity()

    # Compute Keplerian state of Lambert arc
    lambert_arc_keplerian_elements = element_conversion.cartesian_to_keplerian(lambert_arc_initial_state,
                                                                               central_body_gravitational_parameter)

    # Setup Keplerian ephemeris model that describes the Lambert arc
    lambert_arc_ephemeris = environment_setup.create_body_ephemeris(
        environment_setup.ephemeris.keplerian(lambert_arc_keplerian_elements, departure_epoch,
                                              central_body_gravitational_parameter), "")

    # Selected epochs to plot
    epoch_list = np.linspace(departure_epoch, arrival_epoch, number_of_epochs)

    # Building lambert arc history dictionary
    lambert_arc_history = dict()
    for state in epoch_list:
        lambert_arc_history[state] = lambert_arc_ephemeris.cartesian_state(state)

    return lambert_arc_history

########################################################################################################################

########################################################################################################################


def calculate_fpa_from_flyby_pericenter(flyby_rp: float,
                                        flyby_initial_velocity_vector: np.ndarray,
                                        arc_departure_position: np.ndarray,
                                        arc_arrival_radius: float,
                                        mu_moon: float,
                                        moon_in_plane_velocity: np.ndarray,
                                        verbose: bool=True
                                        ) -> float:
    """
    Function to calculate the arrival f.p.a. for the post-flyby arc that ends up at Jupiter's atmosphere.
    All units are in I.S.
    The function returns error code 1000 if the lowest altitude of the second arc doesn't intersect Jupiter's atmosphere

    :param flyby_rp: pericenter radius of flyby
    :param flyby_initial_velocity_vector: arrival v infinite in moon's frame
    :param arc_departure_position: departure position vector of the post-flyby arc
    :param arc_arrival_radius: arrival radius of the post-flyby arc
    :param mu_moon: gravitational parameter of the flyby moon
    :param moon_radius: radius of the flyby moon
    :param moon_SOI_radius: SOI radius of the flyby moon
    :param moon_in_plane_velocity: velocity vector of the flyby moon in the flyby plane (here assumed to be coincident)

    :return: arrival f.p.a. of the post-flyby arc, which lies in the interval [0, np.pi]
    """

    # Calculate v_inf_t
    flyby_initial_velocity = LA.norm(flyby_initial_velocity_vector)

    # Calculate axis normal to flyby plane (based on assumption:flyby plane coincides with moon orbital plane)
    flyby_orbital_plane_normal_axis = unit_vector(np.cross(moon_in_plane_velocity, flyby_initial_velocity_vector))

    # Calculate resulting flyby bending angle
    flyby_alpha_angle = 2 * np.arcsin(1 / (1 + flyby_rp * flyby_initial_velocity ** 2 / mu_moon))

    # Calculate the v_inf_t_star
    flyby_final_velocity_vector = (rotation_matrix(flyby_orbital_plane_normal_axis,flyby_alpha_angle) @
                                   flyby_initial_velocity_vector.reshape(3, 1)).reshape(3)

    # Get initial radius of post-flyby arc
    arc_departure_radius = LA.norm(arc_departure_position)

    # Calculate post-flyby arc departure velocity
    arc_departure_velocity_vector = flyby_final_velocity_vector + moon_in_plane_velocity
    arc_departure_velocity = LA.norm(arc_departure_velocity_vector)

    # Calculate post-flyby arc departure flight path angle
    arc_departure_fpa = np.arcsin(
        np.dot(unit_vector(arc_departure_position), unit_vector(arc_departure_velocity_vector)))

    # Calculate post-flyby arc orbital energy
    arc_orbital_energy = arc_departure_velocity ** 2 / 2 - \
        central_body_gravitational_parameter / arc_departure_radius

    # Calculate post-flyby arc arrival velocity
    arc_arrival_velocity = np.sqrt(
        2 * (arc_orbital_energy + central_body_gravitational_parameter / arc_arrival_radius))

    # Pre-calculation for post-flyby arc arrival flight path angle
    # (arccos argument will be clipped at [-1,1] for cases where orbit doesn't intersect Jup atmosphere)
    arccos_argument = arc_departure_radius / arc_arrival_radius * arc_departure_velocity * np.cos(arc_departure_fpa) / arc_arrival_velocity

    # Post-flyby arc arrival flight path angle
    arc_arrival_fpa = - np.arccos(np.clip(arccos_argument, -1, 1))

    # if arccos_argument > 1:
    #     if verbose:
    #         print('The second arc of the trajectory doesn\'t intersect Jupiter\'s atmosphere!   Function error code: 1000')
    #     return arccos_argument

    return arc_arrival_fpa  # non-error values are between 0 and np.pi


def calculate_fpa_from_flyby_geometry(sigma_angle: float,
                                      arc_1_initial_velocity: float,
                                      arc_1_initial_radius: float,
                                      delta_hoh: float,
                                      arc_2_final_radius: float,
                                      mu_moon: float,
                                      moon_SOI: float,
                                      moon_state_at_flyby: np.ndarray,
                                      moon_radius: float = 0.,
                                      ) -> float:

    moon_position = moon_state_at_flyby[0:3]
    moon_velocity = moon_state_at_flyby[3:6]

    orbit_axis = unit_vector(np.cross(moon_position, moon_velocity))
    flyby_initial_position = rotate_vectors_by_given_matrix(rotation_matrix(orbit_axis, -sigma_angle), unit_vector(moon_velocity)) * moon_SOI

    h_arc_1 = arc_1_initial_radius * arc_1_initial_velocity * np.sin(delta_hoh)
    energy_arc_1 = arc_1_initial_velocity**2/2 - central_body_gravitational_parameter / arc_1_initial_radius

    arc_1_final_position = moon_position + flyby_initial_position
    arc_1_final_radius = LA.norm(arc_1_final_position)
    arc_1_final_velocity = np.sqrt(2 * (energy_arc_1 + central_body_gravitational_parameter/arc_1_final_radius))

    arc_1_final_fpa = - np.arccos(h_arc_1/(arc_1_final_radius*arc_1_final_velocity))
    arc_1_final_velocity_vector = rotate_vectors_by_given_matrix(rotation_matrix(orbit_axis, np.pi / 2 - arc_1_final_fpa), unit_vector(arc_1_final_position)) * arc_1_final_velocity

    flyby_initial_velocity_vector = arc_1_final_velocity_vector - moon_velocity

    flyby_v_inf_t = LA.norm(flyby_initial_velocity_vector)

    # phi_2_angle = np.arccos(np.dot(unit_vector(-moon_velocity),unit_vector(flyby_initial_velocity_vector)))
    #
    # delta_angle = 2 * np.pi - np.arccos(np.dot(unit_vector(-moon_velocity), unit_vector(flyby_initial_position)))

    flyby_axis = unit_vector(np.cross(flyby_initial_position, flyby_initial_velocity_vector))

    # We dont want clockwise orbits
    if np.dot(flyby_axis, orbit_axis) < 0:
        return -1

    phi_2_angle = np.arccos(np.dot(unit_vector(-moon_velocity), unit_vector(flyby_initial_velocity_vector)))
    if np.dot(np.cross(-moon_velocity, flyby_initial_velocity_vector), flyby_axis) < 0:
        phi_2_angle = - phi_2_angle + 2 * np.pi

    # delta_minus_2pi = np.arccos(np.dot(unit_vector(-moon_velocity), unit_vector(flyby_initial_position)))
    # if np.dot(np.cross(-moon_velocity, flyby_initial_position), flyby_axis) > 0:
    #     delta_minus_2pi = - delta_minus_2pi + 2 * np.pi
    # delta_angle = 2 * np.pi - delta_minus_2pi

    delta_angle = np.arccos(np.dot(unit_vector(-moon_velocity), unit_vector(flyby_initial_position)))
    if np.dot(np.cross(-moon_velocity, flyby_initial_position), flyby_axis) < 0:
        delta_angle = 2 * np.pi - delta_angle

    # Sometimes it needs a minus sign but we work around it with abs value. Only its squared value is needed so no issue
    B_parameter = abs(moon_SOI * np.sin(phi_2_angle - delta_angle))

    alpha_angle = 2 * np.arcsin(1/np.sqrt(1 + (B_parameter**2 * flyby_v_inf_t**4) / mu_moon **2))
    if alpha_angle < 0:
        warnings.warn('Alpha angle negative!!!!')

    beta_angle = phi_2_angle + alpha_angle /2 - np.pi /2


    position_rot_angle = 2 * (- delta_angle + beta_angle)
    # if position_rot_angle > 2*np.pi:
    #     position_rot_angle = position_rot_angle - 2 * np.pi
    flyby_final_position = rotate_vectors_by_given_matrix(rotation_matrix(flyby_axis, position_rot_angle), flyby_initial_position)

    flyby_final_velocity_vector = rotate_vectors_by_given_matrix(rotation_matrix(flyby_axis, alpha_angle), flyby_initial_velocity_vector)

    arc_2_departure_position = moon_position + flyby_final_position
    arc_2_departure_velocity_vector = moon_velocity + flyby_final_velocity_vector

    arc_2_departure_radius = LA.norm(arc_2_departure_position)
    arc_2_departure_velocity = LA.norm(arc_2_departure_velocity_vector)

    arc_2_h = LA.norm(np.cross(arc_2_departure_position, arc_2_departure_velocity_vector))
    arc_2_energy = arc_2_departure_velocity**2/2 - central_body_gravitational_parameter / arc_2_departure_radius

    arc_2_arrival_velocity = np.sqrt(2 * (arc_2_energy + central_body_gravitational_parameter / arc_2_final_radius))

    arc_2_final_fpa = - np.arccos(np.clip(arc_2_h/(arc_2_final_radius * arc_2_arrival_velocity), -1, 1))

    root_argument = 1 + (B_parameter ** 2 * flyby_v_inf_t ** 4) / (mu_moon ** 2)
    flyby_pericenter = mu_moon / (flyby_v_inf_t ** 2) * (np.sqrt(root_argument) - 1)
    flyby_altitude = flyby_pericenter - moon_radius
    if flyby_altitude < 0:
        print(f'Flyby impact! Altitude: {flyby_altitude/1e3} km     Sigma: {sigma_angle*180/np.pi} deg')
        return -1
    #     arc_2_final_fpa = arc_2_final_fpa + 1000

    return arc_2_final_fpa

def calculate_fpa_from_flyby_geometry_simplified(pericenter_radius: float,
                                                 arc_1_initial_velocity: float,
                                                 arc_1_initial_radius: float,
                                                 delta_hoh: float,
                                                 arc_2_final_radius: float,
                                                 mu_moon: float,
                                                 moon_SOI: float,
                                                 moon_state_at_flyby: np.ndarray,
                                                 moon_radius: float = 0.,
                                                 ) -> (float, float):
    moon_position = moon_state_at_flyby[0:3]
    moon_velocity = moon_state_at_flyby[3:6]
    orbit_axis = unit_vector(np.cross(moon_position, moon_velocity))

    # beta_angle = ...

    ## secant method ###

    max_iter = 100
    # curr_iter = 0
    tolerance = 1e-5
    function_values = np.zeros(max_iter+1)
    x_values = np.zeros(max_iter+1)

    x_values[0] = 0.
    function_values[0] = beta_angle_function(beta_angle_guess=x_values[0],
                                             pericenter_radius=pericenter_radius,
                                             arc_1_initial_velocity=arc_1_initial_velocity,
                                             arc_1_initial_radius=arc_1_initial_radius,
                                             delta_hoh=delta_hoh,
                                             mu_moon=mu_moon,
                                             moon_SOI=moon_SOI,
                                             moon_state_at_flyby=moon_state_at_flyby,
                                             moon_radius=moon_radius)
    save_x = 0.
    x_values[1] = np.pi
    for curr_iter in range(1,max_iter):
        if curr_iter == 1:
            x_values[curr_iter] = np.pi
        function_values[curr_iter] = beta_angle_function(beta_angle_guess=x_values[curr_iter],
                                                         pericenter_radius=pericenter_radius,
                                                         arc_1_initial_velocity=arc_1_initial_velocity,
                                                         arc_1_initial_radius=arc_1_initial_radius,
                                                         delta_hoh=delta_hoh,
                                                         mu_moon=mu_moon,
                                                         moon_SOI=moon_SOI,
                                                         moon_state_at_flyby=moon_state_at_flyby,
                                                         moon_radius=moon_radius)

        x_values[curr_iter+1] = x_values[curr_iter] - function_values[curr_iter] *\
                                       (x_values[curr_iter]-x_values[curr_iter-1]) / (function_values[curr_iter]-function_values[curr_iter-1])

        if abs(x_values[curr_iter+1] - x_values[curr_iter]) < tolerance:
            save_x = x_values[curr_iter+1]
            break
    beta_angle = save_x

    ####################

    flyby_pericenter_position = rotate_vectors_by_given_matrix(rotation_matrix(orbit_axis, beta_angle), unit_vector(-moon_velocity)) * pericenter_radius

    h_arc_1 = arc_1_initial_radius * arc_1_initial_velocity * np.sin(delta_hoh)
    energy_arc_1 = arc_1_initial_velocity**2/2 - central_body_gravitational_parameter / arc_1_initial_radius

    arc_1_final_position = moon_position + flyby_pericenter_position
    arc_1_final_radius = LA.norm(arc_1_final_position)
    arc_1_final_velocity = np.sqrt(2 * (energy_arc_1 + central_body_gravitational_parameter/arc_1_final_radius))

    arc_1_final_fpa = - np.arccos(h_arc_1/(arc_1_final_radius*arc_1_final_velocity))
    arc_1_final_velocity_vector = rotate_vectors_by_given_matrix(rotation_matrix(orbit_axis, np.pi / 2 - arc_1_final_fpa), unit_vector(arc_1_final_position)) * arc_1_final_velocity

    flyby_initial_velocity_vector = arc_1_final_velocity_vector - moon_velocity

    flyby_v_inf_t = LA.norm(flyby_initial_velocity_vector)

    # phi_2_angle = np.arccos(np.dot(unit_vector(-moon_velocity),unit_vector(flyby_initial_velocity_vector)))
    #
    # delta_angle = 2 * np.pi - np.arccos(np.dot(unit_vector(-moon_velocity), unit_vector(flyby_initial_position)))

    flyby_axis = unit_vector(np.cross(flyby_pericenter_position, flyby_initial_velocity_vector))

    # We dont want clockwise orbits
    # if np.dot(flyby_axis, orbit_axis) < 0:
    #     return -1000

    phi_2_angle = np.arccos(np.dot(unit_vector(-moon_velocity), unit_vector(flyby_initial_velocity_vector)))
    if np.dot(np.cross(-moon_velocity, flyby_initial_velocity_vector), flyby_axis) < 0:
        phi_2_angle = - phi_2_angle + 2 * np.pi

    # delta_minus_2pi = np.arccos(np.dot(unit_vector(-moon_velocity), unit_vector(flyby_initial_position)))
    # if np.dot(np.cross(-moon_velocity, flyby_initial_position), flyby_axis) > 0:
    #     delta_minus_2pi = - delta_minus_2pi + 2 * np.pi
    # delta_angle = 2 * np.pi - delta_minus_2pi

    # delta_angle = np.arccos(np.dot(unit_vector(-moon_velocity), unit_vector(flyby_initial_position)))
    # if np.dot(np.cross(-moon_velocity, flyby_initial_position), flyby_axis) < 0:
    #     delta_angle = 2 * np.pi - delta_angle

    # Sometimes it needs a minus sign but we work around it with abs value. Only its squared value is needed so no issue
    # B_parameter = abs(moon_SOI * np.sin(phi_2_angle - delta_angle))

    alpha_angle = 2 * np.arcsin(1/(1 + pericenter_radius * flyby_v_inf_t**2/mu_moon))
    if alpha_angle < 0:
        warnings.warn('Alpha angle negative!!!!')

    # function_beta_angle = phi_2_angle + alpha_angle / 2 - np.pi / 2 - beta_angle


    # position_rot_angle = 2 * (- delta_angle + beta_angle)
    # if position_rot_angle > 2*np.pi:
    #     position_rot_angle = position_rot_angle - 2 * np.pi

    flyby_final_velocity_vector = rotate_vectors_by_given_matrix(rotation_matrix(flyby_axis, alpha_angle), flyby_initial_velocity_vector)

    arc_2_departure_position = moon_position + flyby_pericenter_position
    arc_2_departure_velocity_vector = moon_velocity + flyby_final_velocity_vector

    arc_2_departure_radius = LA.norm(arc_2_departure_position)
    arc_2_departure_velocity = LA.norm(arc_2_departure_velocity_vector)

    arc_2_h = LA.norm(np.cross(arc_2_departure_position, arc_2_departure_velocity_vector))
    arc_2_energy = arc_2_departure_velocity**2/2 - central_body_gravitational_parameter / arc_2_departure_radius

    arc_2_arrival_velocity = np.sqrt(2 * (arc_2_energy + central_body_gravitational_parameter / arc_2_final_radius))

    arc_2_final_fpa = - np.arccos(np.clip(arc_2_h/(arc_2_final_radius * arc_2_arrival_velocity), -1, 1))

    # flyby_altitude = pericenter_radius - moon_radius
    # if flyby_altitude < 0:
    #     print(f'Flyby impact! Altitude: {flyby_altitude/1e3} km')
    #     return -1
    #     arc_2_final_fpa = arc_2_final_fpa + 1000

    return arc_2_final_fpa, beta_angle


def beta_angle_function(beta_angle_guess: float,
                        pericenter_radius:float,
                        arc_1_initial_velocity: float,
                        arc_1_initial_radius: float,
                        delta_hoh: float,
                        mu_moon: float,
                        moon_state_at_flyby: np.ndarray,
                        moon_SOI: float,
                        moon_radius: float = 0.,
                        ) -> float:
    moon_position = moon_state_at_flyby[0:3]
    moon_velocity = moon_state_at_flyby[3:6]
    orbit_axis = unit_vector(np.cross(moon_position, moon_velocity))

    flyby_pericenter_position = rotate_vectors_by_given_matrix(rotation_matrix(orbit_axis, beta_angle_guess),
                                                               unit_vector(-moon_velocity)) * pericenter_radius

    h_arc_1 = arc_1_initial_radius * arc_1_initial_velocity * np.sin(delta_hoh)
    energy_arc_1 = arc_1_initial_velocity ** 2 / 2 - central_body_gravitational_parameter / arc_1_initial_radius

    arc_1_final_position = moon_position + flyby_pericenter_position
    arc_1_final_radius = LA.norm(arc_1_final_position)
    arc_1_final_velocity = np.sqrt(2 * (energy_arc_1 + central_body_gravitational_parameter / arc_1_final_radius))

    arc_1_final_fpa = - np.arccos(h_arc_1 / (arc_1_final_radius * arc_1_final_velocity))
    arc_1_final_velocity_vector = rotate_vectors_by_given_matrix(
        rotation_matrix(orbit_axis, np.pi / 2 - arc_1_final_fpa),
        unit_vector(arc_1_final_position)) * arc_1_final_velocity

    flyby_initial_velocity_vector = arc_1_final_velocity_vector - moon_velocity

    flyby_v_inf_t = LA.norm(flyby_initial_velocity_vector)

    flyby_axis = unit_vector(np.cross(flyby_pericenter_position, flyby_initial_velocity_vector))

    # We dont want clockwise orbits
    # if np.dot(flyby_axis, orbit_axis) < 0:
    #     return 1

    phi_2_angle = np.arccos(np.dot(unit_vector(-moon_velocity), unit_vector(flyby_initial_velocity_vector)))
    if np.dot(np.cross(-moon_velocity, flyby_initial_velocity_vector), flyby_axis) < 0:
        phi_2_angle = - phi_2_angle + 2 * np.pi

    alpha_angle = 2 * np.arcsin(1 / (1 + pericenter_radius * flyby_v_inf_t ** 2 / mu_moon))
    if alpha_angle < 0:
        warnings.warn('Alpha angle negative!!!!')

    function_beta_angle = phi_2_angle + alpha_angle / 2 - np.pi / 2 - beta_angle_guess

    return function_beta_angle


def calculate_orbit_pericenter_from_flyby_pericenter(flyby_rp: float,
                                                     flyby_initial_velocity_vector: np.ndarray,
                                                     arc_departure_position: np.ndarray,
                                                     mu_moon: float,
                                                     moon_flyby_state: np.ndarray,
                                                     verbose: bool=True
                                                     ) -> float:
    """
    Function to calculate the orbit pericenter for the post-aerocapture post-flyby arc that ends up at a
    closed orbit around Jupiter.
    All units are in I.S.
    The function returns error code 1000 if the lowest altitude of the second arc doesn't intersect Jupiter's atmosphere

    :param flyby_rp: pericenter radius of flyby
    :param flyby_initial_velocity_vector: arrival v infinite in moon's frame
    :param arc_departure_position: departure position vector of the post-flyby arc
    :param arc_arrival_radius: arrival radius of the post-flyby arc
    :param mu_moon: gravitational parameter of the flyby moon
    :param moon_radius: radius of the flyby moon
    :param moon_SOI_radius: SOI radius of the flyby moon
    :param moon_in_plane_velocity: velocity vector of the flyby moon in the flyby plane (here assumed to be coincident)

    :return: arrival f.p.a. of the post-flyby arc, which lies in the interval [0, np.pi]
    """

    # Calculate v_inf_t
    flyby_initial_velocity = LA.norm(flyby_initial_velocity_vector)

    # Calculate axis normal to flyby plane (based on assumption:flyby plane coincides with moon orbital plane)
    flyby_orbital_axis = unit_vector(np.cross(moon_flyby_state[0:3], moon_flyby_state[3:6]))

    # Calculate resulting flyby bending angle
    flyby_alpha_angle = 2 * np.arcsin(1 / (1 + flyby_rp * flyby_initial_velocity ** 2 / mu_moon))

    # Calculate the v_inf_t_star
    flyby_final_velocity_vector = (rotation_matrix(flyby_orbital_axis, flyby_alpha_angle) @
                                   flyby_initial_velocity_vector.reshape(3, 1)).reshape(3)

    # Get initial radius of post-flyby arc
    arc_departure_radius = LA.norm(arc_departure_position)

    # Calculate post-flyby arc departure velocity
    arc_departure_velocity_vector = flyby_final_velocity_vector + moon_flyby_state[3:6]
    arc_departure_velocity = LA.norm(arc_departure_velocity_vector)

    # Calculate post-flyby arc departure flight path angle
    arc_departure_fpa = np.arcsin(
        np.dot(unit_vector(arc_departure_position), unit_vector(arc_departure_velocity_vector)))

    # Calculate post-flyby arc orbital energy
    arc_orbital_energy = arc_departure_velocity ** 2 / 2 - \
        central_body_gravitational_parameter / arc_departure_radius

    arc_angular_momentum = arc_departure_radius * arc_departure_velocity * np.cos(arc_departure_fpa)

    arc_semilatus_rectum = arc_angular_momentum ** 2 / central_body_gravitational_parameter
    arc_semimajor_axis = - central_body_gravitational_parameter / (2 * arc_orbital_energy)
    arc_eccentricity = np.sqrt(1 - arc_semilatus_rectum / arc_semimajor_axis)

    arc_pericenter_radius = arc_semimajor_axis * (1-arc_eccentricity)

    return arc_pericenter_radius


########################################################################################################################
# ATMOSPHERIC ENTRY HELPER FUNCTIONS ###################################################################################
########################################################################################################################


def atmospheric_entry_trajectory_distance_travelled(fpa_angle: np.ndarray,
                                                    atmospheric_entry_fpa: float,
                                                    effective_entry_fpa: float,
                                                    scale_height: float) -> np.ndarray:

    tangent_sum_fpa_angle = np.tan(effective_entry_fpa / 2) + np.tan(fpa_angle / 2)
    tangent_subtraction_fpa_angle = np.tan(effective_entry_fpa / 2) - np.tan(fpa_angle / 2)

    tangent_sum_fpa_entry = np.tan(effective_entry_fpa / 2) + np.tan(atmospheric_entry_fpa / 2)
    tangent_subtraction_fpa_entry = np.tan(effective_entry_fpa / 2) - np.tan(atmospheric_entry_fpa / 2)

    logarithm_arg = (tangent_sum_fpa_angle / tangent_subtraction_fpa_angle) * (
                tangent_subtraction_fpa_entry / tangent_sum_fpa_entry)

    horizontal_distance_traveled = scale_height * (
                fpa_angle - atmospheric_entry_fpa + 1 / np.tan(effective_entry_fpa) * np.log(logarithm_arg))

    return horizontal_distance_traveled


def atmospheric_entry_trajectory_altitude(fpa_angle: np.ndarray,
                                          fpa_entry: float,
                                          density_entry: float,
                                          density_0: float,
                                          weight_over_surface_cl_coeff: float,
                                          g_acceleration: float,
                                          beta_parameter: float,
                                          ) -> np.ndarray:
    current_density = (np.cos(fpa_angle) - np.cos(fpa_entry)) * 2*beta_parameter/g_acceleration * weight_over_surface_cl_coeff + density_entry
    current_altitude = - 1/beta_parameter * np.log(current_density/density_0)

    return current_altitude


# def entry_coords_to_polar_2d():


def atmospheric_pressure_given_altitude(altitude,
                                        surface_acceleration: float,
                                        b_curvature: float,
                                        gas_constant: float,
                                        layers: dict,
                                        input_density: bool = False):

    # layer dictionary key content: (T_0, h_0, p_0, alpha) from lower to higher

    # number_of_layers = len(layers.keys())
    location_layer = layers[list(layers.keys())[0]]

    for layer in layers.keys():
        h_0 = layers[layer][1]
        T_0 = layers[layer][0]

        if altitude >= h_0:
            location_layer = layer
        if input_density:
            layers[layer][2] = layers[layer][2] * gas_constant * T_0

    T_0 = layers[location_layer][0]
    h_0 = layers[location_layer][1]
    val_0 = layers[location_layer][2]
    alpha = layers[location_layer][3]

    # temperature = T_0 + (altitude-h_0) / alpha

    # g_acc = surface_acceleration * (1 - b_curvature * altitude)
    g_0 = surface_acceleration
    R_gas = gas_constant

    if alpha == np.inf:
        first_term = g_0 * (altitude-h_0) / (R_gas * T_0)
        second_term = 1 - b_curvature/2 * (altitude - h_0)
        return val_0 * np.exp(- first_term * second_term)

    a_term = (altitude-h_0)/(alpha*T_0) + 1
    b_term = -((g_0*alpha)/R_gas * (1+b_curvature*(T_0*alpha - h_0)))
    c_term = g_0 * b_curvature * alpha / R_gas * (altitude-h_0)
    pressure = val_0 * (a_term**b_term) * np.exp(c_term)

    return pressure


# def entry_model_A_matrix(v_i, v_f):
#     A_matrix = np.array([[1, v_i, v_i**2, v_i**3, v_i**4],
#                         [1, v_f, v_f**2, v_f**3, v_f**4],
#                         [0, 1, 2*v_i, 3*v_i**2, 4*v_i**3],
#                         [0, 1, 2*v_f, 3*v_f**2, 4*v_f**3],
#                         [v_f-v_i, (v_f**2-v_i**2)/2, (v_f**3-v_i**3)/3, (v_f**4-v_i**4)/4, (v_f**5-v_i**5)/5]
#                         ])
#     return A_matrix
#
# def entry_model_b_vector():
#     b_vec = np.array([[],
#                       [],
#                       [],
#                       [],
#                       []
#                       ])

########################################################################################################################
# ZERO-FINDING METHODS #################################################################################################
########################################################################################################################


def regula_falsi_illinois(interval_boundaries: tuple,
                          function: Callable,
                          zero_value: float = 0.,
                          tolerance: float = 1e-15,
                          max_iter: int = 1000,
                          **kwargs) -> tuple:
    a_int = interval_boundaries[0]
    b_int = interval_boundaries[1]

    f_a = function(a_int, **kwargs) - zero_value
    f_b = function(b_int, **kwargs) - zero_value

    if f_a * f_b > 0:
        raise Exception('The selected interval has either none or multiple zeroes.')

    ass_a = False
    ass_b = False
    i = 0
    c_point = -1
    f_c = -1

    for i in range(max_iter):

        c_point = (a_int * f_b - b_int * f_a) / (f_b - f_a)

        f_c = function(c_point, **kwargs) - zero_value

        if abs(f_c) < tolerance:
            # Root found
            break

        if f_c < 0:
            if ass_a:
                # m_ab = 1 - f_c / f_a
                # if m_ab < 0:
                m_ab = 0.5
                f_b = f_b * m_ab
            a_int = c_point
            f_a = f_c
            ass_a = True
            ass_b = False

        if f_c > 0:
            if ass_b:
                # m_ab = 1-f_c/f_b
                # if m_ab < 0:
                m_ab = 0.5
                f_a = f_a * m_ab
            b_int = c_point
            f_b = f_c
            ass_a = False
            ass_b = True

    if i == max_iter:
        raise Warning('Regula falsi hasn\'t converged: max number of iterations reached.')

    return c_point, f_c, i

# add secant method
