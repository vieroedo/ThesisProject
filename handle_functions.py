import numpy as np
import scipy as sp
import math
import numpy.linalg as LA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os
from typing import Callable

# TUDAT imports
from tudatpy.kernel.astro import two_body_dynamics
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.math import interpolators


from JupiterTrajectory_GlobalParameters import *

handle_functions_directory = os.path.dirname(__file__)

# Ref system axes
x_axis = np.array([1., 0., 0.])
y_axis = np.array([0., 1., 0.])
z_axis = np.array([0., 0., 1.])


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
    """
    Returns a vector (or array of vectors) rotated according to the chosen rotation matrix.
    In case of an array of vectors, the shape of the object is (n,3), where n is the number of vectors.
    """
    # vector_got_transposed = False
    vector_shape = np.shape(vector)

    if len(vector_shape) == 1:
        if len(vector) != 3:
            raise Exception('Vector for rotation must have three components!')
        return (rot_matrix @ vector.reshape(3, 1)).reshape(3)

    elif len(vector_shape) == 2:
        if vector_shape[0] != 3:
            vector = vector.T  # makes the vector with shape (3,n)
            # vector_got_transposed = True
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


def rotate_vector(vector, axis, angle):
    vector_shape = np.shape(vector)
    # angle_len = len(angle) if type(angle) is not np.float64 else 0
    try:
        angle_len = len(angle)
    except:
        angle_len = 0

    if angle_len == 0:
        rot_matrix = rotation_matrix(axis, angle)
        rotated_vector = rotate_vectors_by_given_matrix(rot_matrix, vector)
        return rotated_vector
    elif len(vector_shape) == 1 and angle_len > 1:
        rotated_vectors = np.zeros((angle_len,3))
        for i in range(angle_len):
            rot_matrix = rotation_matrix(axis, angle[i])
            rotated_vectors[i,:] = rotate_vectors_by_given_matrix(rot_matrix, vector)
        return rotated_vectors
    elif len(vector_shape)==2 and angle_len>1:
        rotated_vectors = np.zeros((angle_len, 3))
        if vector_shape[1] != 3:
            vector = vector.T # the vector has shape (n,3) now any case
        if len(vector[:,0]) != angle_len:
            raise Exception("number of vectors and values for rotationa angles must coincide!")
        for i in range(angle_len):
            rot_matrix = rotation_matrix(axis, angle[i])
            rotated_vectors[i, :] = rotate_vectors_by_given_matrix(rot_matrix, vector[i,:])
        return rotated_vectors
    else:
        raise Exception('wrong shape for angles or vectors')


def velocity_vector_from_position(position, axis, fpa, velocity_mag):
    if len(np.shape(position)) == 2:
        try:
            velocity_mag = velocity_mag.reshape((len(velocity_mag),1))
        except:
            raise Exception('idk something is wrong in here')
    velocity_unit_vector = rotate_vector(unit_vector(position), axis, np.pi / 2 - fpa)
    velocity_vector = velocity_unit_vector * velocity_mag
    return velocity_vector


def cartesian_2d_from_polar(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def cartesian_3d_from_polar(r, latitude, longitude):
    # singularity per latitude 90 degrees
    x = r * np.cos(latitude) * np.cos(longitude)
    y = r * np.cos(latitude) * np.sin(longitude)
    z = r * np.sin(latitude)
    return x, y, z


def longitude_from_cartesian(position_state: np.ndarray, verbose: bool = True):
    if len(position_state) != 3:
        raise Exception("wrong vector length")
    if position_state[2] != 0:
        position_state[2] = 0
        if verbose:
            warnings.warn("z component adjusted, you didnt input an equatorial vector")
    vector_normal_axis = np.cross(x_axis, position_state)
    is_positive = True if np.dot(vector_normal_axis,z_axis) > 0 else False
    longitude = np.arccos(np.dot(x_axis, unit_vector(position_state)))
    if is_positive:
        return longitude
    else:
        return - longitude


########################################################################################################################
# ASTRODYNAMICS HELPER FUNCTIONS #######################################################################################
########################################################################################################################
CARTESIAN_STATE_NUMBER_OF_ENTRIES = 6
PARABOLA_ECCENTRICITY = 1


def kepler_elements_from_cartesian_state(cartesian_state: np.ndarray, mu_parameter: float):

    if len(cartesian_state) != CARTESIAN_STATE_NUMBER_OF_ENTRIES:
        raise ValueError('Unexpected value for the cartesian state inserted')

    state_position = cartesian_state[0:3]
    state_velocity = cartesian_state[3:6]

    eccentricity_vector = eccentricity_vector_from_cartesian_state(cartesian_state)
    angular_momentum_vector = np.cross(state_position,state_velocity)
    node_vector = np.cross(z_axis,angular_momentum_vector)
    E_orbital_energy = orbital_energy(LA.norm(state_position), LA.norm(state_velocity), mu_parameter)

    eccentricity = LA.norm(eccentricity_vector)

    if eccentricity == PARABOLA_ECCENTRICITY:
        semimajor_axis = np.inf
    else:
        semimajor_axis = - mu_parameter/(2*E_orbital_energy)

    inclination = np.arccos(angular_momentum_vector[2] / LA.norm(angular_momentum_vector))
    longitude_of_ascending_node = np.arccos(node_vector[0]/LA.norm(node_vector))
    argument_of_periapsis = np.arccos(np.dot(unit_vector(node_vector), unit_vector(eccentricity_vector)))

    true_anomaly = np.arccos(np.dot(unit_vector(eccentricity_vector), unit_vector(state_position)))

    if eccentricity_vector[2] < 0:
        argument_of_periapsis = 2*np.pi - argument_of_periapsis

    if np.dot(state_position,state_velocity) < 0:
        true_anomaly = 2*np.pi - true_anomaly

    kepler_elements_list = [semimajor_axis,eccentricity,inclination,argument_of_periapsis,longitude_of_ascending_node,true_anomaly]
    kepler_elements = np.array(kepler_elements_list)
    return kepler_elements


def orbital_energy(radius: float, velocity: float, mu_parameter:float):
    return velocity ** 2 / 2 - mu_parameter / radius


def angular_momentum(radius:float, velocity:float, flight_path_angle:float):
    return radius * velocity * np.cos(flight_path_angle)


def orbital_period(semimajor_axis:float, mu_parameter:float):
    return 2*np.pi*np.sqrt(semimajor_axis**3/mu_parameter)


def velocity_from_energy(energy: float, radius: float, mu_parameter:float):
    return np.sqrt(2 * (energy + mu_parameter / radius))


def fpa_from_angular_momentum(angular_momentum, radius, velocity, is_fpa_positive):
    if is_fpa_positive:
        return np.arccos(angular_momentum/(radius*velocity))
    else:
        return - np.arccos(angular_momentum / (radius * velocity))


def fpa_from_cartesian_state(position_vec, velocity_vec):
    # returns values within [-np.pi/2, np.pi/2]
    return np.arcsin(np.dot(unit_vector(position_vec), unit_vector(velocity_vec)))


def true_anomaly_from_radius(radius,eccentricity,sma, return_positive, silence=True):
    """ WARNING: the solutions of ths function are 2! +theta and -theta, but only +theta is retrieved
    :param return_positive:
    """
    theta = np.arccos(np.clip(1/eccentricity * (sma*(1-eccentricity**2)/radius - 1), -1, 1))
    if not silence:
        warnings.warn('The solutions of this function are 2! +theta and -theta, but only +theta is returned')
    if return_positive:
        return theta
    else:
        return - theta


def radius_from_true_anomaly(true_anomaly, eccentricity, sma, planet_SoI = jupiter_SOI_radius):
    e = eccentricity
    theta = true_anomaly
    if e>1:
        # radius = np.where(theta < np.arccos(-1/e), sma * (1 - e**2) / (1 + e*np.cos(theta)), planet_SoI)
        radius = sma * (1 - e**2) / (1 + e*np.cos(theta))
    else:
        radius = sma * (1 - e**2) / (1 + e*np.cos(theta))
    return radius


def eccentric_to_true_anomaly(anomaly, eccentricity):
    """anomaly can be eccentric or hyperbolic anomaly"""
    if eccentricity < 1:
        return 2 * np.arctan(np.sqrt((1 + eccentricity) / (1 - eccentricity)) * np.tan(anomaly / 2))
    if eccentricity == 1:
        raise ValueError('no parabola implemented. eccentricity must not be equal to 1')
    if eccentricity > 1:
        return 2 * np.arctan(np.sqrt((eccentricity + 1) / (eccentricity - 1)) * np.tanh(anomaly / 2))
    raise ValueError('Unexpected value of eccentricity')


def delta_t_from_delta_true_anomaly(true_anomaly_range: np.ndarray,
                                    eccentricity: float,
                                    semi_major_axis: float,
                                    mu_parameter: float,
                                    verbose:bool = False) -> float:
    """

    :param true_anomaly_range: vector containing initial and final true anomalies
    :param eccentricity: eccentricity of the orbit
    :param semi_major_axis: sma of the orbit. If the orbit is parabolic, it indicates the p parameter
    :param mu_parameter: central body grav parameter
    :return: delta t between the two true anomalies
    """
    if not len(true_anomaly_range) == 2:
        raise Exception('The true_anomaly_range must be a ndarray with size (2,)')
    if eccentricity < 0 or mu_parameter < 0:
        raise Exception('invalid eccentricity or gravitational parameter')

    if true_anomaly_range[1]-true_anomaly_range[0] > 0:
        forward_calculation = True
    elif true_anomaly_range[1]-true_anomaly_range[0] == 0:
        return 0.
    else:
        forward_calculation = False

    if eccentricity > 1:
        theta_extreme = np.arccos(-1/eccentricity)
        if forward_calculation:
            if true_anomaly_range[0] < -theta_extreme or true_anomaly_range[1] > theta_extreme:
                raise Exception('Invalid true anomaly range for the hyperbola')
        else:
            if true_anomaly_range[1] < -theta_extreme or true_anomaly_range[0] > theta_extreme:
                raise Exception('Invalid true anomaly range for the hyperbola')

        arctanh_argument = np.tan(true_anomaly_range/2) * np.sqrt((eccentricity - 1)/(eccentricity + 1))
        eccentric_anomaly_range = 2 * np.arctanh(arctanh_argument)
        mean_anomaly_range = eccentricity * np.sinh(eccentric_anomaly_range) - eccentric_anomaly_range
        angular_velocity = np.sqrt(mu_parameter/-semi_major_axis**3)

        addition_to_elapsed_time = 0.
        revolutions_no = 0
    elif eccentricity == 1:
        if forward_calculation:
            if true_anomaly_range[0] < -np.pi or true_anomaly_range[1] > np.pi:
                raise Exception('Invalid true anomaly range for the parabola')
        else:
            if true_anomaly_range[0] < -np.pi or true_anomaly_range[1] > np.pi:
                raise Exception('Invalid true anomaly range for the parabola')
        p_parameter = semi_major_axis
        mean_anomaly_range = 1/2 * (np.tan(true_anomaly_range/2) + np.tan(true_anomaly_range/2)**3/3)
        angular_velocity = np.sqrt(mu_parameter/p_parameter**3)
        # warnings.warn('Not implemented for parabolic orbits lol')
        addition_to_elapsed_time = 0.
        revolutions_no = 0
    else:
        if abs(true_anomaly_range[1] - true_anomaly_range[0]) == 2 * np.pi:
            return orbital_period(semi_major_axis, mu_parameter)

        revolutions_no = int((true_anomaly_range[1] - true_anomaly_range[0]) / (2 * np.pi))
        if forward_calculation:
            true_anomaly_range[1] = true_anomaly_range[1] - revolutions_no * 2 * np.pi
        else:
            true_anomaly_range[0] = true_anomaly_range[0] - revolutions_no * 2 * np.pi

        arctan_argument = np.tan(true_anomaly_range / 2) * np.sqrt((1-eccentricity) / (1+eccentricity))
        eccentric_anomaly_range = 2 * np.arctan(arctan_argument)
        mean_anomaly_range = eccentric_anomaly_range - eccentricity * np.sin(eccentric_anomaly_range)
        angular_velocity = np.sqrt(mu_parameter/semi_major_axis**3)

        addition_to_elapsed_time = orbital_period(semi_major_axis,mu_parameter)*revolutions_no

    time_range = mean_anomaly_range / angular_velocity

    single_orbit_arc_elapsed_time = (time_range[1] - time_range[0])

    if forward_calculation and single_orbit_arc_elapsed_time < 0:
        single_orbit_arc_elapsed_time = orbital_period(semi_major_axis,mu_parameter) - abs(single_orbit_arc_elapsed_time)

    elapsed_time = single_orbit_arc_elapsed_time + addition_to_elapsed_time

    if verbose:
        print(f'Number of revolutions: {revolutions_no}     Elapsed time: {elapsed_time} s      Orbital period: {orbital_period(semi_major_axis,mu_parameter)} s')
        print(f'ohwell {(time_range[1] - time_range[0])}')
    return elapsed_time


def true_anomaly_from_delta_t(time_vector_pericenter_referenced: np.ndarray,
                              eccentricity: float,
                              semi_major_axis: float,
                              mu_parameter: float,
                              verbose:bool = False) -> float:

    if eccentricity == 1:
        raise ValueError('parabolas? we dont do that here. invalid value for eccentricity (e != 1)')

    angular_velocity = np.sqrt(mu_parameter / abs(semi_major_axis) ** 3)
    mean_anomaly_range = time_vector_pericenter_referenced * angular_velocity

    eccentric_anomaly_initial_guesses = kepler_equation_initial_guess(mean_anomaly_range, eccentricity)
    kepler_equation_solution = sp.optimize.fsolve(kepler_equation_zeroed, eccentric_anomaly_initial_guesses, (eccentricity, mean_anomaly_range))
    eccentric_anomaly_range = kepler_equation_solution

    # eccentric_anomaly_range_test = np.zeros(2)
    # for i in range(2):
    #     sol = newton_rhapson(kepler_equation, kepler_equation_derivative, eccentric_anomaly_initial_guesses[i],
    #                          mean_anomaly_range[i], eccentricity=eccentricity)
    #     eccentric_anomaly_range_test[i] = sol[0]

    true_anomaly_range = eccentric_to_true_anomaly(eccentric_anomaly_range, eccentricity)
    delta_true_anomaly = true_anomaly_range[1] - true_anomaly_range[0]

    if delta_true_anomaly < 0:
        warnings.warn('true anomaly spanned is negative. did you put a negative time range?')

    return delta_true_anomaly


def kepler_equation(anomaly, eccentricity):
    """anomaly can be eccentric or hyperbolic anomaly, depending on the eccentricity"""
    if eccentricity < 0:
        raise ValueError('Eccentricity cannot go below zero')
    if 0 <= eccentricity < 1:
        return anomaly - eccentricity * np.sin(anomaly)
    if eccentricity == 1:
        raise Exception('parabola not implemented')
    if eccentricity > 1:
        return eccentricity * np.sinh(anomaly) - anomaly
    raise ValueError('Unexpected value for eccentricity')


def kepler_equation_derivative(eccentric_anomaly, eccentricity):
    if eccentricity < 0:
        raise ValueError('Eccentricity cannot go below zero')
    if 0 <= eccentricity < 1:
        return 1 - eccentricity * np.cos(eccentric_anomaly)
    if eccentricity == 1:
        raise Exception('parabola not implemented')
    if eccentricity > 1:
        return eccentricity * np.cosh(eccentric_anomaly) - 1
    raise ValueError('Unexpected value for eccentricity')


def kepler_equation_initial_guess(mean_anomaly_range, eccentricity: float):
    if type(mean_anomaly_range) in [float, np.float64]:
        mean_anomaly_range = np.array([mean_anomaly_range])

    number_of_guesses = len(mean_anomaly_range)

    eccentric_anomaly_initial_guesses = np.zeros(number_of_guesses)
    for i in range(number_of_guesses):
        # mean_anomaly_range[i] = mean_anomaly_range[i] + 2*np.pi if mean_anomaly_range[i] < 0 else mean_anomaly_range[i]
        if 0 < mean_anomaly_range[i] % (2 * np.pi) < np.pi or -2*np.pi < mean_anomaly_range[i] % (2 * np.pi) < -np.pi:
            eccentric_anomaly_initial_guesses[i] = mean_anomaly_range[i] + eccentricity / 2
        else:
            eccentric_anomaly_initial_guesses[i] = mean_anomaly_range[i] - eccentricity / 2

    return eccentric_anomaly_initial_guesses

def kepler_equation_zeroed(eccentric_anomaly, eccentricity, mean_anomaly_range):
    return kepler_equation(eccentric_anomaly, eccentricity) - mean_anomaly_range

def true_anomaly_from_referenced_delta_t(absolute_epochs_interval: np.ndarray,
                                         initial_true_anomaly_reference_point: float,
                                         eccentricity: float,
                                         semi_major_axis: float,
                                         mu_parameter: float,
                                         verbose:bool = False) -> float:

    delta_t_from_pericenter = delta_t_from_delta_true_anomaly(np.array([0., initial_true_anomaly_reference_point]),
                                                              eccentricity, semi_major_axis, mu_parameter)
    relative_epochs_interval = absolute_epochs_interval - absolute_epochs_interval[0]
    pericenter_referenced_epochs = relative_epochs_interval + delta_t_from_pericenter

    delta_true_anomaly = true_anomaly_from_delta_t(pericenter_referenced_epochs,
                                                   eccentricity,semi_major_axis,mu_parameter)
    return delta_true_anomaly


def eccentricity_vector_from_cartesian_state(cartesian_state: np.ndarray) -> np.ndarray:
    if cartesian_state.shape[0] != 6 or len(cartesian_state.shape) != 1:
        raise Exception('Unexpected vector entered as cartesian state.')
    state_position = cartesian_state[0:3]
    state_velocity = cartesian_state[3:6]

    position_multiplier = LA.norm(state_velocity) ** 2 \
                          - jupiter_gravitational_parameter / LA.norm(state_position)
    velocity_multiplier = np.dot(state_position, state_velocity)
    eccentricity_vector = 1 / jupiter_gravitational_parameter * (
                position_multiplier * state_position - velocity_multiplier * state_velocity)
    return eccentricity_vector


def moon_circular_2d_state(epoch: float, choose_moon: str) -> np.ndarray((6,)):
    orbital_radius = galilean_moons_data[choose_moon]['SMA']
    orbital_period = galilean_moons_data[choose_moon]['Orbital_Period']

    true_anomaly = epoch * 2 * np.pi / orbital_period
    k_value = int(true_anomaly/2*np.pi)
    true_anomaly = true_anomaly - 2*k_value*np.pi

    x_pos, y_pos = cartesian_2d_from_polar(orbital_radius, true_anomaly)
    z_pos = 0.

    moon_velocity = np.sqrt(jupiter_gravitational_parameter / orbital_radius)

    x_vel, y_vel = cartesian_2d_from_polar(moon_velocity, true_anomaly+np.pi/2)
    z_vel = 0.

    return np.array([x_pos, y_pos, z_pos, x_vel, y_vel, z_vel])


def compute_lambert_targeter_state_history(
        initial_state,
        final_state,
        departure_epoch,
        arrival_epoch,
        central_body_grav_parameter=jupiter_gravitational_parameter,
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
                                                                               jupiter_gravitational_parameter)

    # Setup Keplerian ephemeris model that describes the Lambert arc
    lambert_arc_ephemeris = environment_setup.create_body_ephemeris(
        environment_setup.ephemeris.keplerian(lambert_arc_keplerian_elements, departure_epoch,
                                              jupiter_gravitational_parameter), "")

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
                         jupiter_gravitational_parameter / arc_departure_radius

    # Calculate post-flyby arc arrival velocity
    arc_arrival_velocity = np.sqrt(
        2 * (arc_orbital_energy + jupiter_gravitational_parameter / arc_arrival_radius))

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
                                      flyby_moon,
                                      flyby_epoch,
                                      equatorial_approximation: bool = False
                                      # mu_moon: float,
                                      # moon_SOI: float,
                                      # moon_state_at_flyby: np.ndarray,
                                      # moon_radius: float = 0.,
                                      ) -> float:

    moon_flyby_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name=flyby_moon,
        observer_body_name="Jupiter",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=flyby_epoch)

    moon_position = moon_flyby_state[0:3]
    moon_velocity = moon_flyby_state[3:6]

    if equatorial_approximation:
        moon_position[2] = 0.
        moon_velocity[2] = 0.

    mu_moon = galilean_moons_data[flyby_moon]['mu']
    moon_SOI = galilean_moons_data[flyby_moon]['SOI_Radius']
    moon_radius = galilean_moons_data[flyby_moon]['Radius']

    orbit_axis = unit_vector(np.cross(moon_position, moon_velocity))
    flyby_initial_position = rotate_vectors_by_given_matrix(rotation_matrix(orbit_axis, -sigma_angle), unit_vector(moon_velocity)) * moon_SOI

    h_arc_1 = arc_1_initial_radius * arc_1_initial_velocity * np.sin(delta_hoh)
    energy_arc_1 = arc_1_initial_velocity ** 2 / 2 - jupiter_gravitational_parameter / arc_1_initial_radius

    arc_1_final_position = moon_position + flyby_initial_position
    arc_1_final_radius = LA.norm(arc_1_final_position)
    arc_1_final_velocity = np.sqrt(2 * (energy_arc_1 + jupiter_gravitational_parameter / arc_1_final_radius))

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

    delta_angle = np.arccos(np.clip(np.dot(unit_vector(-moon_velocity), unit_vector(flyby_initial_position)), -1, 1))
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

    flyby_pericenter = mu_moon / (flyby_v_inf_t ** 2) * (
                np.sqrt(1 + (B_parameter ** 2 * flyby_v_inf_t ** 4) / (mu_moon ** 2)) - 1)
    flyby_altitude = flyby_pericenter - moon_radius
    if flyby_altitude < 0:
        print(f'Flyby impact! Altitude: {flyby_altitude / 1e3} km     Sigma: {sigma_angle * 180 / np.pi} deg')
        return -1
    #     arc_2_final_fpa = arc_2_final_fpa + 1000
    flyby_orbital_energy = orbital_energy(LA.norm(flyby_initial_position), flyby_v_inf_t, mu_parameter=mu_moon)
    flyby_sma = - mu_moon/(2*flyby_orbital_energy)
    flyby_eccentricity = 1 - flyby_pericenter/flyby_sma

    true_anomaly_boundary = 2 * np.pi - delta_angle + beta_angle
    true_anomaly_boundary = true_anomaly_boundary if true_anomaly_boundary < 2* np.pi else true_anomaly_boundary - 2*np.pi
    true_anomaly_range = np.array([-true_anomaly_boundary, true_anomaly_boundary])
    flyby_elapsed_time = delta_t_from_delta_true_anomaly(true_anomaly_range,
                                                         eccentricity=flyby_eccentricity,
                                                         semi_major_axis=flyby_sma,
                                                         mu_parameter=mu_moon)
    flyby_final_epoch = flyby_epoch + flyby_elapsed_time
    moon_flyby_final_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name=flyby_moon,
        observer_body_name="Jupiter",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=flyby_final_epoch)
    moon_final_position = moon_flyby_final_state[0:3]
    moon_final_velocity = moon_flyby_final_state[3:6]
    if equatorial_approximation:
        moon_final_position[2] = 0.
        moon_final_velocity[2] = 0.

    arc_2_departure_position = moon_final_position + flyby_final_position
    arc_2_departure_velocity_vector = moon_final_velocity + flyby_final_velocity_vector

    arc_2_departure_radius = LA.norm(arc_2_departure_position)
    arc_2_departure_velocity = LA.norm(arc_2_departure_velocity_vector)

    arc_2_h = LA.norm(np.cross(arc_2_departure_position, arc_2_departure_velocity_vector))
    arc_2_energy = arc_2_departure_velocity ** 2 / 2 - jupiter_gravitational_parameter / arc_2_departure_radius

    arc_2_arrival_velocity = np.sqrt(2 * (arc_2_energy + jupiter_gravitational_parameter / arc_2_final_radius))

    arc_2_final_fpa = - np.arccos(np.clip(arc_2_h/(arc_2_final_radius * arc_2_arrival_velocity), -1, 1))

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
    energy_arc_1 = arc_1_initial_velocity ** 2 / 2 - jupiter_gravitational_parameter / arc_1_initial_radius

    arc_1_final_position = moon_position + flyby_pericenter_position
    arc_1_final_radius = LA.norm(arc_1_final_position)
    arc_1_final_velocity = np.sqrt(2 * (energy_arc_1 + jupiter_gravitational_parameter / arc_1_final_radius))

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
    arc_2_energy = arc_2_departure_velocity ** 2 / 2 - jupiter_gravitational_parameter / arc_2_departure_radius

    arc_2_arrival_velocity = np.sqrt(2 * (arc_2_energy + jupiter_gravitational_parameter / arc_2_final_radius))

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
    energy_arc_1 = arc_1_initial_velocity ** 2 / 2 - jupiter_gravitational_parameter / arc_1_initial_radius

    arc_1_final_position = moon_position + flyby_pericenter_position
    arc_1_final_radius = LA.norm(arc_1_final_position)
    arc_1_final_velocity = np.sqrt(2 * (energy_arc_1 + jupiter_gravitational_parameter / arc_1_final_radius))

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
                         jupiter_gravitational_parameter / arc_departure_radius

    arc_angular_momentum = arc_departure_radius * arc_departure_velocity * np.cos(arc_departure_fpa)

    arc_semilatus_rectum = arc_angular_momentum ** 2 / jupiter_gravitational_parameter
    arc_semimajor_axis = - jupiter_gravitational_parameter / (2 * arc_orbital_energy)
    arc_eccentricity = np.sqrt(1 - arc_semilatus_rectum / arc_semimajor_axis)

    arc_pericenter_radius = arc_semimajor_axis * (1-arc_eccentricity)

    return arc_pericenter_radius


########################################################################################################################
# ATMOSPHERIC ENTRY HELPER FUNCTIONS ###################################################################################
########################################################################################################################

# Define some global objects for data interpolation in functions
jup_atm_number_of_entries = len(jupiter_altitude_atmosphere_values_seiff1998)
jup_atm_density_values_reshaped = np.array(list(jupiter_density_atmosphere_values_seiff1998)).reshape((jup_atm_number_of_entries, 1))
handle_density_altitude_dictionary = dict(zip(jupiter_altitude_atmosphere_values_seiff1998, jup_atm_density_values_reshaped))
handle_density_value_interpolator = interpolators.create_one_dimensional_vector_interpolator(handle_density_altitude_dictionary,
                                                                                             global_interpolator_settings)

galileo_velocity_values_reshaped = np.array(list(galileo_flight_velocity)).reshape((len(galileo_flight_velocity), 1))
handle_velocity_altitude_dictionary = dict(zip(galileo_flight_altitude, galileo_velocity_values_reshaped))
handle_velocity_value_interpolator = interpolators.create_one_dimensional_vector_interpolator(handle_velocity_altitude_dictionary,
                                                                                              global_interpolator_settings)


def galileo_velocity_from_altitude(h):
    # h in meters
    velocities = np.zeros(len(h))
    for i in range(len(h)):
        if galileo_flight_altitude_boundaries[0] < h[i] < galileo_flight_altitude_boundaries[1]:
            velocities[i] = handle_velocity_value_interpolator.interpolate(h[i])
    return velocities


def jupiter_atmosphere_exponential(altitude):
    """

    :param altitude:
        entry altitude in metres
    :return:
        atmospheric density in kg/m^3
    """
    density = jupiter_1bar_density * np.exp(-altitude/jupiter_scale_height)
    return density


def jupiter_atmosphere_density_model(h: np.ndarray):
    selected_altitude_km = h/1e3  # km

    # if selected_altitude_km > interpolation_boundaries[1]:
    #     ...  # exponential
    # if selected_altitude_km < interpolation_boundaries[0]:
    #     ...  # exponential

    if type(selected_altitude_km) == np.float64 or type(selected_altitude_km) == float:
        if not atmosphere_altitude_values_boundaries_seiff1998[0] < selected_altitude_km < atmosphere_altitude_values_boundaries_seiff1998[1]:
            # use altitude in meters since scale height is in meters too
            return jupiter_1bar_density * np.exp(-h/jupiter_scale_height)
        density_interpolated = handle_density_value_interpolator.interpolate(selected_altitude_km)
        return density_interpolated
    elif type(selected_altitude_km) == np.ndarray:
        density_values = np.zeros(len(h))
        for i in range(len(h)):
            if atmosphere_altitude_values_boundaries_seiff1998[0] < selected_altitude_km[i] < atmosphere_altitude_values_boundaries_seiff1998[1]:
                density_values[i] = handle_density_value_interpolator.interpolate(selected_altitude_km[i])
            else:
                density_values[i] = jupiter_1bar_density * np.exp(-h[i] / jupiter_scale_height)
        return density_values


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
                                          ballistic_coefficient_times_g_acc: float,
                                          g_acceleration: float,
                                          beta_parameter: float,
                                          ) -> np.ndarray:
    current_density = (np.cos(fpa_angle) - np.cos(fpa_entry)) * 2 * beta_parameter / g_acceleration * ballistic_coefficient_times_g_acc + density_entry
    current_altitude = - 1/beta_parameter * np.log(current_density/density_0)

    return current_altitude


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
        # if input_density:
        #     layers[layer][2] = layers[layer][2] * gas_constant * T_0

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


def atmospheric_entry_heat_loads_correlations(density, velocity, nose_radius):
    '''
    obtained from https://adsabs.harvard.edu/pdf/2006ESASP.631E...6R

    :returns
        convective and radiative heat fluxes in W/m^2
    '''

    convective_heat_flux = 2004.2526 * 1/(np.sqrt(2*nose_radius/0.6091)) * (density/1.22522)**(0.4334341) * (velocity/3048) ** (2.9978867)
    # convective_heat_flux = 3.6380163716698004e-08 / np.sqrt(nose_radius) * density**0.4334341 * velocity ** 2.9978867

    radiative_heat_flux = 9.7632379e-40 * (2* nose_radius)**(-0.17905) * density ** 1.763827469 * velocity ** 10.993852
    # radiative_heat_flux = 8.623716107859813e-40 * nose_radius**(-0.17905) * density ** 1.763827469 * velocity**10.993852

    zero_cells = np.where(density == 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gamma_factor = 4 * radiative_heat_flux*1e3 / (density * velocity**3)
        radiation_with_blockage = radiative_heat_flux*1e3 / (1 + 3 * gamma_factor**0.7)
    radiation_with_blockage[zero_cells] = 0
    return convective_heat_flux*1e3, radiative_heat_flux*1e3, radiation_with_blockage


def heat_flux_with_blockage_from_blowing(total_wall_hfx, single_hf, density, velocity):
    # chapter 5.1.1 of https://doi.org/10.1016/j.actaastro.2012.06.016

    k = 2.7659985259098415e-28 # from interpolation
    dm_dt = k * density * velocity ** 6.9

    # hot wall correction can be neglected in hypersonic flow (T_inf >> T_wall)
    B = dm_dt * velocity**2 / (2 * total_wall_hfx)

    investigated_hfx_w_blockage = (1 - 0.72 * B + 0.13 * B**2) * single_hf
    return investigated_hfx_w_blockage


def ablation_blockage(incident_hfx, density, velocity, total_wall_hfx):
    # chapter 2 of https://doi.org/10.1016/j.actaastro.2012.06.016

    k = 2.7659985259098415e-28  # from interpolation
    dm_dt = k * density * velocity ** 6.9

    # hot wall correction can be neglected in hypersonic flow (T_inf >> T_wall)
    # if type(total_wall_hfx) == np.ndarray:
    zero_cells = np.where(total_wall_hfx == 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        blowing_coefficient = dm_dt * velocity ** 2 / (2 * total_wall_hfx)
    # else:
    #     zero_cells = np.where(convective_hfx == 0)
    #     blowing_coefficient = dm_dt * velocity ** 2 / (2 * convective_hfx)
    blowing_coefficient[zero_cells] = 0

    b_coeff_null_cells = np.where(blowing_coefficient == 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        blockage_factor = ( 2.344/blowing_coefficient * (np.sqrt(blowing_coefficient+1)-1) ) **1.063
    blockage_factor[b_coeff_null_cells] = 1.18378  # value of the function for blowing_coefficient = 0
    incident_hfx_w_blockage = blockage_factor * incident_hfx
    return incident_hfx_w_blockage, blowing_coefficient


def custom_atm_convective_hfx_correlation(density, velocity, radius, return_scaling_vector = False):
    # correlation obtained by interpolation of rarefield and low altitudes heat flux data
    c1, m, n = [4.13377754, 0.51467325, 5.66475942]
    scaling_vector = np.array([1.58227848e+00, 4.92610837e+02, 2.09863589e-02, 6.49350649e-06])

    heat_flux = c1 * (radius ** -0.5) * density ** m * velocity ** n
    # hot wall correction can be neglected in hypersonic flow (T_inf >> T_wall)
    if return_scaling_vector:
        return heat_flux, scaling_vector
    return heat_flux


def galileo_heat_fluxes_park(entry_altitudes):
    # Data from https://arc-aiaa-org.tudelft.idm.oclc.org/doi/pdf/10.2514/1.38712
    '''

    :param entry_altitudes:
    :return:
    q_r boundary-layer edge, q_r wall, q_c wall
    '''
    galileo_heat_fluxes = np.loadtxt(handle_functions_directory + '/Just_aerocapture/Numerical_approach/GalileoMission/heat_fluxes_galileo_low_altitudes.txt')
    qr_boundary_layer_edge = galileo_heat_fluxes[:, 4] * 1e7  # W/m^2
    qr_wall = galileo_heat_fluxes[:, 5] * 1e7  # W/m^2
    qc_wall = galileo_heat_fluxes[:, 6] * 1e7  # W/m^2
    heat_fluxes_altitudes = galileo_heat_fluxes[:, 1] * 1e3  # m

    altitude_boundaries = [heat_fluxes_altitudes[-1], heat_fluxes_altitudes[0]]

    # entries_length = len(heat_fluxes_altitudes)
    heat_fluxes_vector = np.vstack((qr_boundary_layer_edge, qr_wall, qc_wall)).T  # .reshape((entries_length, 3))
    interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation=interpolators.extrapolate_at_boundary)
    heatflux_altitude_dictionary = dict(zip(heat_fluxes_altitudes, heat_fluxes_vector))
    heatflux_value_interpolator = interpolators.create_one_dimensional_vector_interpolator(heatflux_altitude_dictionary,
                                                                                           interpolator_settings)

    flight_heat_fluxes = np.zeros((len(entry_altitudes), 3))
    for i, curr_altitude in enumerate(entry_altitudes):
        if altitude_boundaries[0] < curr_altitude < altitude_boundaries[1]:
            flight_heat_fluxes[i, :] = heatflux_value_interpolator.interpolate(curr_altitude)

    return flight_heat_fluxes


def convective_heat_flux_girija(density, velocity, radius):
    # from https://arc.aiaa.org/doi/pdf/10.2514/1.A35214?src=getftr
    k_constant = 0.6556E-8
    convective_hfx = k_constant * (density/radius)**0.5 * velocity**3
    return convective_hfx


def entry_velocity_from_interplanetary(interplanetary_arrival_velocity_in_jupiter_frame):
    pre_ae_departure_radius = jupiter_SOI_radius
    pre_ae_departure_velocity_norm = interplanetary_arrival_velocity_in_jupiter_frame
    pre_ae_orbital_energy = orbital_energy(pre_ae_departure_radius, pre_ae_departure_velocity_norm)
    pre_ae_arrival_radius = jupiter_radius + atmospheric_entry_altitude
    pre_ae_arrival_velocity_norm = velocity_from_energy(pre_ae_orbital_energy, pre_ae_arrival_radius)
    return pre_ae_arrival_velocity_norm

########################################################################################################################
# ZERO-FINDING METHODS #################################################################################################
########################################################################################################################


def regula_falsi_illinois(interval_boundaries: tuple,
                          function: Callable,
                          zero_value: float = 0.,
                          tolerance: float = 1e-15,
                          max_iter: int = 1000,
                          illinois_addition: bool = True,
                          **kwargs) -> tuple:

    a_int, b_int = interval_boundaries[0], interval_boundaries[1]

    f_a = function(a_int, **kwargs) - zero_value
    f_b = function(b_int, **kwargs) - zero_value

    if f_a * f_b > 0:
        raise Exception('The selected interval has either none or multiple zeroes.')

    ass_a, ass_b = False, False
    i, c_point, f_c = 0, -1, -1

    for i in range(max_iter):

        c_point = (a_int * f_b - b_int * f_a) / (f_b - f_a)
        f_c = function(c_point, **kwargs) - zero_value

        if abs(f_c) < tolerance:
            # Root found
            break

        if f_a * f_c > 0:
            if ass_a and illinois_addition:
                # m_ab = 1 - f_c / f_a
                # if m_ab < 0:
                m_ab = 0.5
                f_b = f_b * m_ab
            a_int = c_point
            f_a = f_c
            ass_a, ass_b = True, False

        if f_b * f_c > 0:
            if ass_b and illinois_addition:
                # m_ab = 1-f_c/f_b
                # if m_ab < 0:
                m_ab = 0.5
                f_a = f_a * m_ab
            b_int = c_point
            f_b = f_c
            ass_a, ass_b = False, True

    if i == max_iter:
        raise Warning('Regula falsi hasn\'t converged: max number of iterations reached.')

    return c_point, f_c, i


def secant_method(function: Callable,
                  x_1: float,
                  x_2: float,
                  zero_value: float = 0.,
                  tolerance: float = 1e-15,
                  max_iter: int = 1000,
                  **kwargs) -> tuple:
    i = 0
    f__x_1 = -1

    for i in range(max_iter):

        f__x_1 = function(x_1, **kwargs) - zero_value
        f__x_2 = function(x_2, **kwargs) - zero_value

        if abs(f__x_1) < tolerance:
            # Root found
            break

        x_temp = x_1
        x_1 = x_1 - (x_1 - x_2) * f__x_1 / (f__x_1 - f__x_2)
        x_2 = x_temp

    if i == max_iter:
        raise Warning('Secant method has not converged: max number of iterations reached.')

    return x_1, f__x_1, i


def newton_rhapson(function: Callable,
                   derivative_function: Callable,
                   initial_guess: float,
                   zero_value: float=0,
                   tolerance: float = 1e-15,
                   max_iter: int = 1000,
                   **kwargs):

    x_value = initial_guess
    i = 0

    for i in range(max_iter):
        x_new = x_value - (function(x_value, **kwargs) - zero_value) / derivative_function(x_value, **kwargs)
        if abs(x_new - x_value) < tolerance:
            x_value = x_new
            break
        x_value = x_new

    if i == max_iter:
        raise Warning('Newthon method has not converged: max number of iterations reached.')

    return x_value, function(x_value, **kwargs), i

########################################################################################################################
# PLOTTING FUNCTIONS ###################################################################################################
########################################################################################################################


def plot_jupiter_and_galilean_orbits(ax, plot_orbits=True, title_addition=''):

    # draw jupiter
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = jupiter_radius * np.cos(u) * np.sin(v)
    y = jupiter_radius * np.sin(u) * np.sin(v)
    z = jupiter_radius * np.cos(v)
    ax.plot_wireframe(x, y, z, color="saddlebrown")

    # label axes and figure
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Jupiter aerocapture trajectory\n' + title_addition)

    # draw galilean moons' orbits
    if plot_orbits:
        for moon in galilean_moons_data.keys():
            moon_sma = galilean_moons_data[moon]['SMA']
            theta_angle = np.linspace(0, 2 * np.pi, 200)
            x_m = moon_sma * np.cos(theta_angle)
            y_m = moon_sma * np.sin(theta_angle)
            z_m = np.zeros(len(theta_angle))
            ax.plot3D(x_m, y_m, z_m, 'b')

    return ax

def set_axis_limits(ax):
    # set proper plot axis limits
    xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
    XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim * 0.75)
    ax.set_aspect('auto')
    return ax

def plot_galilean_moon(ax, galilean_moon, epoch):
    if galilean_moon not in ['Io', 'Europa', 'Ganymede', 'Callisto']:
        raise Exception('wrong name for galilea moon')

    flyby_moon_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name=galilean_moon,
        observer_body_name="Jupiter",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=epoch)
    moon_radius = galilean_moons_data[galilean_moon]['Radius']

    # draw moon
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x_0 = flyby_moon_state[0]
    y_0 = flyby_moon_state[1]
    z_0 = flyby_moon_state[2]
    x = x_0 + moon_radius * np.cos(u) * np.sin(v)
    y = y_0 + moon_radius * np.sin(u) * np.sin(v)
    z = z_0 + moon_radius * np.cos(v)
    ax.plot_wireframe(x, y, z, color="b")
    return ax

def plot_grid(ax, tick_size):
    # Show the major grid lines with dark grey lines
    ax.grid(visible=True, which='major', color='#666666', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    return ax