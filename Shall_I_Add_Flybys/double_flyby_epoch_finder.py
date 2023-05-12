import numpy as np

# # Tudatpy imports
# from tudatpy.io import save2txt
# from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
# from tudatpy.kernel.numerical_simulation import environment_setup
# from tudatpy.kernel.numerical_simulation import propagation_setup
# from tudatpy.kernel.numerical_simulation import environment
# from tudatpy.kernel import numerical_simulation
# from tudatpy.kernel.math import interpolators

# Problem-specific imports
from JupiterTrajectory_GlobalParameters import *
import CapsuleEntryUtilities as Util
from handle_functions import *
import class_AerocaptureSemianalyticalModel as ae_analytical
import class_FlybysAnalytical as fb_analytical

# Load spice kernels
spice_interface.load_standard_kernels()

MJD200_date = 62503.9325  # 01/01/2030 62502
J2000_date = MJD200_date - 51544

first_flyby_initial_epoch = J2000_date * constants.JULIAN_DAY
timespan = ...  # it will be used for interval range search

# The two moons must be different
choose_first_moon = 'Europa'
choose_second_moon = 'Io'

flight_path_angle_at_atmospheric_entry = np.deg2rad(-3)  # input value in: deg
interplanetary_arrival_velocity = 5600  # m/s
delta_angle_from_hohmann_trajectory = np.deg2rad(0.98)  # input value in: deg


first_moon_first_flyby_state = spice_interface.get_body_cartesian_state_at_epoch(
    target_body_name=choose_first_moon,
    observer_body_name="Jupiter",
    reference_frame_name=global_frame_orientation,
    aberration_corrections="NONE",
    ephemeris_time=first_flyby_initial_epoch)

second_moon_first_flyby_state = spice_interface.get_body_cartesian_state_at_epoch(
    target_body_name=choose_second_moon,
    observer_body_name="Jupiter",
    reference_frame_name=global_frame_orientation,
    aberration_corrections="NONE",
    ephemeris_time=first_flyby_initial_epoch)

sun_state = spice_interface.get_body_cartesian_state_at_epoch(
    target_body_name='Sun',
    observer_body_name="Jupiter",
    reference_frame_name=global_frame_orientation,
    aberration_corrections="NONE",
    ephemeris_time=first_flyby_initial_epoch)
sunline = unit_vector(sun_state[0:3])
sunline[2] = 0.


# first_moon_first_flyby_state__equatorial = first_moon_first_flyby_state
# first_moon_first_flyby_state__equatorial[2] = 0.
# first_moon_first_flyby_state__equatorial[5] = 0.
#
# second_moon_first_flyby_state__equatorial = second_moon_first_flyby_state
# second_moon_first_flyby_state__equatorial[2] = 0.
# second_moon_first_flyby_state__equatorial[5] = 0.

first_moon_longitude__first_flyby_state = longitude_from_cartesian(first_moon_first_flyby_state[0:3])
first_moon_sma = LA.norm(first_moon_first_flyby_state[0:3])
second_moon_longitude__first_flyby_state = longitude_from_cartesian(second_moon_first_flyby_state[0:3])
second_moon_sma = LA.norm(second_moon_first_flyby_state[0:3])

sc_angular_momentum__initial = jupiter_SOI_radius * interplanetary_arrival_velocity * np.sin(delta_angle_from_hohmann_trajectory)
sc_orbital_energy__initial = orbital_energy(jupiter_SOI_radius, interplanetary_arrival_velocity, jupiter_gravitational_parameter)

sc_semilatus_rectum = sc_angular_momentum__initial ** 2 / jupiter_gravitational_parameter
# They depend on problem parameters
sc_sma__initial =  - jupiter_gravitational_parameter / (2 * sc_orbital_energy__initial)
sc_eccentricity__initial = np.sqrt(1 - sc_semilatus_rectum/ sc_sma__initial)

# arrival_pericenter_altitude =
print(f'Arrival orbit pericenter altitude: {(sc_sma__initial*(1-sc_eccentricity__initial)-jupiter_radius)/1e3} km')
print(f'Moon of choice: {choose_first_moon}    Moon Altitude: {(first_moon_sma-jupiter_radius)/1e3} km')
print(f'Orbit pericenter - Moon SMA: {(sc_sma__initial*(1-sc_eccentricity__initial)-first_moon_sma)/1e3} km')

initial_true_anomaly = - true_anomaly_from_radius(jupiter_SOI_radius, sc_eccentricity__initial, sc_sma__initial, True)
first_flyby_true_anomaly = - true_anomaly_from_radius(first_moon_sma, sc_eccentricity__initial, sc_sma__initial, True)

# flyby occurs
orbital_parameters = [interplanetary_arrival_velocity, flight_path_angle_at_atmospheric_entry, delta_angle_from_hohmann_trajectory]
flyby_problem = fb_analytical.SingleFlybyApproach([0.,0.],first_flyby_initial_epoch,choose_first_moon,verbose=True)
flyby_problem.fitness(orbital_parameters)
flyby_parameters = flyby_problem.flyby_approach_parameters_function()

sc_eccentricity__post_flyby = flyby_parameters[1]
sc_sma__post_flyby = flyby_parameters[0]
atmospheric_entry_fpa = flyby_parameters[5]


print(f'Atmospheric entry fpa: {np.rad2deg(atmospheric_entry_fpa):.3f} deg')
print(f'Atm entry fpa initially set: {np.rad2deg(flight_path_angle_at_atmospheric_entry):.3f} deg')

atmospheric_entry_true_anomaly = - true_anomaly_from_radius(atmospheric_entry_altitude, sc_eccentricity__post_flyby,
                                                            sc_sma__post_flyby, True)

true_anomaly_range = np.array([first_flyby_true_anomaly, atmospheric_entry_true_anomaly])
second_arc_delta_t = delta_t_from_delta_true_anomaly(true_anomaly_range,sc_eccentricity__post_flyby,sc_sma__post_flyby, jupiter_gravitational_parameter)
second_arc_delta_true_anomaly = atmospheric_entry_true_anomaly - first_flyby_true_anomaly


# aerocapture
aerocapture_analytical_problem = ae_analytical.AerocaptureSemianalyticalModel([0., 0.], orbit_datapoints=100,
                                                                              equations_order=2)
aerocapture_analytical_problem.fitness([interplanetary_arrival_velocity, flight_path_angle_at_atmospheric_entry])
aerocapture_problem_parameters = aerocapture_analytical_problem.aerocapture_parameters_function()

# Atmosphere exit fpa
atmospheric_exit_fpa = aerocapture_problem_parameters[0]
# Atmosphere exit velocity
atmospheric_exit_velocity_norm = aerocapture_problem_parameters[1]
# Minimum altitude
minimum_altitude = aerocapture_problem_parameters[3]
# Travelled distance (assumed at surface)
final_distance_travelled = aerocapture_problem_parameters[2]
# Final radius after aerocapture
atmospheric_exit_radius = aerocapture_problem_parameters[4]
# Aerocapture phase angle
atmospheric_entry_phase_angle = aerocapture_problem_parameters[5]

# atmospheric_exit_radius = jupiter_radius + atmospheric_entry_altitude

aerocapture_delta_t = 0  # considered instantaneous
aerocapture_delta_true_anomaly = atmospheric_entry_phase_angle

sc_orbital_energy__exit = orbital_energy(atmospheric_exit_radius, atmospheric_exit_velocity_norm, jupiter_gravitational_parameter)
sc_angular_momentum__exit = angular_momentum(atmospheric_exit_radius, atmospheric_exit_velocity_norm, atmospheric_exit_fpa)
sc_semilatus_rectum__exit = sc_angular_momentum__exit ** 2 / jupiter_gravitational_parameter

sc_sma__post_aerocapture = - jupiter_gravitational_parameter / (2 * sc_orbital_energy__exit)
sc_eccentricity__post_aerocapture = np.sqrt(1 - sc_semilatus_rectum__exit/sc_sma__post_aerocapture)

atmospheric_exit_true_anomaly = true_anomaly_from_radius(atmospheric_entry_altitude, sc_eccentricity__post_aerocapture,
                                                         sc_sma__post_aerocapture, True)
second_flyby_true_anomaly = true_anomaly_from_radius(second_moon_sma, sc_eccentricity__post_aerocapture,
                                                     sc_sma__post_aerocapture, True)

true_anomaly_range = np.array([atmospheric_exit_true_anomaly, second_flyby_true_anomaly])
third_arc_delta_t = delta_t_from_delta_true_anomaly(true_anomaly_range,sc_eccentricity__post_aerocapture,sc_sma__post_aerocapture, jupiter_gravitational_parameter)
third_arc_delta_true_anomaly = second_flyby_true_anomaly - atmospheric_exit_true_anomaly

delta_t_between_flybys = second_arc_delta_t + aerocapture_delta_t + third_arc_delta_t
second_flyby_epoch = first_flyby_initial_epoch + delta_t_between_flybys
second_moon_second_flyby_state = spice_interface.get_body_cartesian_state_at_epoch(
    target_body_name=choose_second_moon,
    observer_body_name="Jupiter",
    reference_frame_name=global_frame_orientation,
    aberration_corrections="NONE",
    ephemeris_time=second_flyby_epoch)

second_moon_mean_motion = 2*np.pi/galilean_moons_data[choose_second_moon]['Orbital_Period']
second_moon_revolutions = int((second_moon_mean_motion * delta_t_between_flybys)/(2*np.pi))

second_moon_longitude__second_flyby_state = longitude_from_cartesian(second_moon_second_flyby_state[0:3])

# first moon as is, gets subtracted
# second moon as is if positive, otherwise absolute value + pi
if second_moon_longitude__second_flyby_state <0:
    second_moon_longitude__second_flyby_state = np.pi - second_moon_longitude__second_flyby_state

moons_delta_longitude_between_flybys = second_moon_longitude__second_flyby_state - first_moon_longitude__first_flyby_state + 2*np.pi * second_moon_revolutions
sc_delta_longitude_between_flybys = second_arc_delta_true_anomaly + aerocapture_delta_true_anomaly + third_arc_delta_true_anomaly

longitude_error = moons_delta_longitude_between_flybys - sc_delta_longitude_between_flybys

print(longitude_error)