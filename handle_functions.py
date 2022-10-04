import numpy as np
import math
import numpy.linalg as LA
from matplotlib import pyplot as plt
import warnings

# TUDAT imports
from tudatpy.kernel.astro import two_body_dynamics
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.astro import element_conversion

# Load spice kernels
spice_interface.load_standard_kernels()

# Ref system axes
x_axis = np.array([1., 0., 0.])
y_axis = np.array([0., 1., 0.])
z_axis = np.array([0., 0., 1.])

# Global values of the problem
jupiter_mass = 1898.6e24  # kg
jupiter_radius = 69911e3  # m
jupiter_SOI_radius = 48.2e9  # m
central_body_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Jupiter')
global_frame_orientation = 'ECLIPJ2000'

# orbital and physical data of galilean moons in SI units
# (entries: 'SMA' 'Mass' 'Radius' 'Orbital_period' 'SOI_Radius' 'mu' 'g_0')
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

# for moon in galilean_moons_data.keys():
#     g_0 = 'g_0'
#     print(f'g_0 of {moon}: {galilean_moons_data[moon][g_0]:.3f} m/s^2')


# Problem parameters
flight_path_angle_at_atmosphere_entry = -3.  # degrees
interplanetary_arrival_velocity_in_jupiter_frame = 5600 # m/s
delta_angle_from_hohmann_trajectory = 0.94  # degrees
# second_arc_time_of_flight = 0.3  # days
choose_flyby_moon = 'Europa'
number_of_epochs_to_plot = 1000



# Parameters reprocessing
flight_path_angle_at_atmosphere_entry = flight_path_angle_at_atmosphere_entry * np.pi / 180  # rad
delta_angle_from_hohmann_trajectory = delta_angle_from_hohmann_trajectory * np.pi / 180  # rad


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def ccw_rotation_z(vector, angle):
    """ Returns the input vector rotated counterclockwise w.r.t. the z axis by the input signed angle"""
    if len(vector) != 3:
        raise Exception('Vector for rotation must be 3-D!')
    counterclockwise_rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    return (counterclockwise_rotation_matrix @ vector.reshape(3,1)).reshape(3)


def ccw_rotation_x(vector, angle):
    """ Returns the input vector rotated counterclockwise w.r.t. the x axis by the input signed angle"""

    vector_shape = np.shape(vector)
    counterclockwise_rotation_matrix = np.array(
        [[1, 0., 0.], [0., np.cos(angle), -np.sin(angle)], [0., np.sin(angle), np.cos(angle)]])

    if len(vector_shape) == 1:
        if len(vector) != 3:
            raise Exception('Vector for rotation must have three components!')
        return (counterclockwise_rotation_matrix @ vector.reshape(3, 1)).reshape(3)

    elif len(vector_shape) == 2:
        vector = vector.T  # makes the vector with shape (3,n)
        if len(vector[:,0]) != 3:
            raise Exception('Vector for rotation must have three components!')

        number_of_vectors = len(vector[0,:])
        rotated_vectors = np.zeros((3,number_of_vectors))

        for i in range(number_of_vectors):
            rotated_vectors[:,i] = counterclockwise_rotation_matrix @ vector[:,i]
        rotated_vectors = rotated_vectors.T  # shape of rotated_vectors is now (n,3)
        return rotated_vectors

    else:
        raise Exception('Wrong input vector dimensions. '
                        'Vector must be either a 1-D array with 3 entries, '
                        'or a 2-D array of vectors with shape (n,3) where "n" is the number of vectors')


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


def rotate_vectors_by_given_matrix(rot_matrix, vector):
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


def true_anomaly_from_radius(radius,eccentricity,sma):
    """ WARNING: the solutions of ths function are 2! +theta and -theta"""
    theta = np.arccos(np.clip(1/eccentricity * (sma*(1-eccentricity**2)/radius - 1), -1, 1))
    return theta


def radius_from_true_anomaly(true_anomaly, eccentricity, sma, planet_SoI = jupiter_SOI_radius):
    e = eccentricity
    theta = true_anomaly
    if e>1:
        radius = np.where(theta < np.arccos(-1/e), sma * (1 - e**2) / (1 + e*np.cos(theta)), planet_SoI)
    else:
        radius = sma * (1 - e**2) / (1 + e*np.cos(theta))
    return radius


def compute_lambert_targeter_state_history(
        initial_state,
        final_state,
        departure_epoch,
        arrival_epoch,
        central_body_grav_parameter=central_body_gravitational_parameter,
        number_of_epochs=number_of_epochs_to_plot):
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

# TO BE FIXED
def plot_trajectory_arc(lambert_arc_history,
                        central_body='Jupiter',
                        figure_title='',
                        ):
    cartesian_elements_lambert = np.vstack(list(lambert_arc_history.values()))
    trajectory_epochs = lambert_arc_history.keys()
    # t_list = time_plot

    # arrival_transverse_velocity = lambertTargeter.get_transverse_arrival_velocity()
    # arrival_radial_velocity = lambertTargeter.get_radial_arrival_velocity()
    #
    # arrival_flight_path_angle = np.arctan(arrival_radial_velocity / arrival_transverse_velocity)
    #
    # print(
    #     f'Initial velocity: {LA.norm(cartesian_elements_lambert[0, 3:])} \n Final velocity: {LA.norm(cartesian_elements_lambert[-1, 3:])}')
    # print(f'Final velocity vector: {cartesian_elements_lambert[-1, 3:]}')
    # print(f'Final f.p.a. : {arrival_flight_path_angle} rad    {arrival_flight_path_angle * 180 / np.pi} degrees')
    # print(
    #     f'For moon phase angle boundaries you should shift time by: ({delta_t_to_low_boundary / constants.JULIAN_DAY:.5f}, {delta_t_to_high_boundary / constants.JULIAN_DAY:.5f})')

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xline = cartesian_elements_lambert[:, 0]
    yline = cartesian_elements_lambert[:, 1]
    zline = cartesian_elements_lambert[:, 2]
    ax.plot3D(xline, yline, zline, 'gray')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title(figure_title)

    xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
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
    ax.plot_wireframe(x, y, z, color="r")

    # TO BE ADDED
    # # draw moon
    # u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    # x_0 = moon_cartesian_state_at_arrival[0]
    # y_0 = moon_cartesian_state_at_arrival[1]
    # z_0 = moon_cartesian_state_at_arrival[2]
    # x = x_0 + moon_radius * np.cos(u) * np.sin(v)
    # y = y_0 + moon_radius * np.sin(u) * np.sin(v)
    # z = z_0 + moon_radius * np.cos(v)
    # ax.plot_wireframe(x, y, z, color="b")
    plt.show()
