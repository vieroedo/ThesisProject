import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt
from handle_functions import *
import warnings

from tudatpy.kernel.astro import two_body_dynamics
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.astro import element_conversion


# Additional problem data
global_frame_orientation = 'ECLIPJ2000'


# Problem parameters to set  (not working w these)
departure_epoch_days = 11201.  # days
time_of_flight = 83  # days (for first arc it is usually around 80-90 days, mostly 85)
choose_arrival_moon = "Europa"
# arrival_pericenter_altitude = 400e3  # m (maybe put SOI edge of the moon)
# departure_pericenter_altitude = 100e3  # m
# moon_pericenter_phase_angle = np.pi * 0.8 # rad  (beta angle)
number_of_epochs_to_plot = 1000  # /
deviation_angle_from_hohmann = 0.  # max value: np.pi/18 (=10 degrees)
arrival_range_angle_wrt_moon_velocity = -np.pi/4




# radii and epochs calculation
moon_radius = galilean_moons_data[choose_arrival_moon]['Radius']  # moons_radii[choose_arrival_moon]  # m
# arrival_pericenter_radius = moon_radius + arrival_pericenter_altitude  # m
# departure_moon_SOI_radius = moon_radius + departure_pericenter_altitude  # m
departure_epoch = departure_epoch_days * constants.JULIAN_DAY   # s
arrival_epoch = departure_epoch + time_of_flight * constants.JULIAN_DAY  # s

# preliminary calculations for initial state
sun_distance_vector = spice_interface.get_body_cartesian_state_at_epoch(
    target_body_name='Sun',
    observer_body_name="Jupiter",
    reference_frame_name=global_frame_orientation,
    aberration_corrections="NONE",
    ephemeris_time=departure_epoch)
sun_line = sun_distance_vector[0:3] / LA.norm(sun_distance_vector[0:3])
counterclockwise_rotation_matrix = np.array([[np.cos(-np.pi/2+deviation_angle_from_hohmann), -np.sin(-np.pi/2+deviation_angle_from_hohmann), 0.],
                                             [np.sin(-np.pi/2+deviation_angle_from_hohmann), np.cos(-np.pi/2+deviation_angle_from_hohmann), 0.], [0., 0., 1.]])
hohmann_arrival_direction = counterclockwise_rotation_matrix @ sun_line.reshape(3,1)

# initial state
initial_state = np.concatenate((hohmann_arrival_direction * jupiter_SOI_radius, np.array([0.,0.,0.]).reshape(3,1)), axis=0).reshape(6)

# FINAL STATE ######################################################################################

# Define moon phase angle based on epoch
moon_cartesian_state_t0 = spice_interface.get_body_cartesian_state_at_epoch(
    target_body_name=choose_arrival_moon,
    observer_body_name="Jupiter",
    reference_frame_name=global_frame_orientation,
    aberration_corrections="NONE",
    ephemeris_time=arrival_epoch) # the t_0 is set at the arrival epoch

plane_normal = np.array([0., 0., 1.])
moon_phase_angle_at_t0 = np.arccos(np.dot(unit_vector(hohmann_arrival_direction.reshape(3)),
                                          unit_vector(moon_cartesian_state_t0[0:3])))
vectors_cross_product = np.cross(hohmann_arrival_direction.reshape(3), moon_cartesian_state_t0[0:3])
if np.dot(plane_normal,vectors_cross_product) < 0:
    moon_phase_angle_at_t0 = - moon_phase_angle_at_t0

usable_moon_phase_angles = (0., np.pi/2)
moon_orbital_period = galilean_moons_data[choose_arrival_moon]['Orbital_Period']
delta_phase_angle_low_boundary = usable_moon_phase_angles[0] - moon_phase_angle_at_t0
delta_phase_angle_high_boundary = usable_moon_phase_angles[1] - moon_phase_angle_at_t0

delta_t_to_low_boundary = delta_phase_angle_low_boundary * (moon_orbital_period/(2*np.pi))
delta_t_to_high_boundary = delta_phase_angle_high_boundary * (moon_orbital_period/(2*np.pi))
if moon_phase_angle_at_t0 > usable_moon_phase_angles[1] or moon_phase_angle_at_t0 < usable_moon_phase_angles[0]:
    warnings.warn('UNFEASIBLE TRAJECTORY ARC\n ')


# preliminary calculations for final state
moon_cartesian_state_at_arrival = spice_interface.get_body_cartesian_state_at_epoch(
    target_body_name=choose_arrival_moon,
    observer_body_name="Jupiter",
    reference_frame_name=global_frame_orientation,
    aberration_corrections="NONE",
    ephemeris_time=arrival_epoch)
moon_velocity = moon_cartesian_state_at_arrival[3:6]
delta_angle_flyby = np.pi + arrival_range_angle_wrt_moon_velocity
counterclockwise_rotation_matrix = np.array([[np.cos(delta_angle_flyby), -np.sin(delta_angle_flyby), 0.],
                                             [np.sin(delta_angle_flyby), np.cos(delta_angle_flyby), 0.], [0., 0., 1.]])
clockwise_rotation_matrix = np.array([[np.cos(delta_angle_flyby), np.sin(delta_angle_flyby), 0.], [-np.sin(delta_angle_flyby), np.cos(delta_angle_flyby), 0.], [0., 0., 1.]])

arrival_range_vector = ( counterclockwise_rotation_matrix @ ((-moon_velocity).reshape(3,1)) / LA.norm(moon_velocity)).reshape(3)

arrival_range_vector = arrival_range_vector * galilean_moons_data[choose_arrival_moon]['SOI_Radius']


# final state
# final_state = moon_cartesian_state_at_arrival[0:3]  # TO BE UPDATED

final_state = np.concatenate((moon_cartesian_state_at_arrival[0:3] + arrival_range_vector, np.array([0.,0.,0.])), axis=0)


# Lambert targeter initialisation
lambertTargeter = two_body_dynamics.LambertTargeterIzzo(
    initial_state[:3], final_state[:3], arrival_epoch - departure_epoch, jupiter_gravitational_parameter)

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
epoch_list = np.linspace(departure_epoch, arrival_epoch, number_of_epochs_to_plot)

# Building lambert arc history dictionary
lambert_arc_history = dict()
for state in epoch_list:
    lambert_arc_history[state] = lambert_arc_ephemeris.cartesian_state(state)

# time = lambert_arc_history.keys()
cartesian_elements_lambert = np.vstack(list(lambert_arc_history.values()))
# time_plot = list(time)
# t_list = time_plot

arrival_transverse_velocity = lambertTargeter.get_transverse_arrival_velocity()
arrival_radial_velocity = lambertTargeter.get_radial_arrival_velocity()

arrival_flight_path_angle = np.arctan(arrival_radial_velocity/arrival_transverse_velocity)

print(f'Initial velocity: {LA.norm(cartesian_elements_lambert[0, 3:])} \n Final velocity: {LA.norm(cartesian_elements_lambert[-1, 3:])}')
print(f'Final velocity vector: {cartesian_elements_lambert[-1, 3:]}')
print(f'Final f.p.a. : {arrival_flight_path_angle} rad    {arrival_flight_path_angle*180/np.pi} degrees')
print(f'For moon phase angle boundaries you should shift time by: ({delta_t_to_low_boundary/constants.JULIAN_DAY:.5f}, {delta_t_to_high_boundary/constants.JULIAN_DAY:.5f})')


fig = plt.figure()
ax = plt.axes(projection='3d')
xline = cartesian_elements_lambert[:,0]
yline = cartesian_elements_lambert[:,1]
zline = cartesian_elements_lambert[:,2]
ax.plot3D(xline, yline, zline, 'gray')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title(f'Lambert Trajectory from Jupiter SOI edge to {choose_arrival_moon}')

xyzlim = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()]).T
XYZlim = np.asarray([min(xyzlim[0]),max(xyzlim[1])])
ax.set_xlim3d(XYZlim)
ax.set_ylim3d(XYZlim)
ax.set_zlim3d(XYZlim * 0.75)
ax.set_aspect('auto')


# draw jupiter
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = jupiter_radius * np.cos(u)*np.sin(v)
y = jupiter_radius * np.sin(u)*np.sin(v)
z = jupiter_radius * np.cos(v)
ax.plot_wireframe(x, y, z, color="r")

# draw moon
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_0 = moon_cartesian_state_at_arrival[0]
y_0 = moon_cartesian_state_at_arrival[1]
z_0 = moon_cartesian_state_at_arrival[2]
x = x_0 + moon_radius * np.cos(u)*np.sin(v)
y = y_0 + moon_radius * np.sin(u)*np.sin(v)
z = z_0 + moon_radius * np.cos(v)
ax.plot_wireframe(x, y, z, color="b")
plt.show()


