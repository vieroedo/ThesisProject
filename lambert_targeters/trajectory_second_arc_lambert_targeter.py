import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt
from handle_functions import *
import warnings

# TUDAT imports
from tudatpy.kernel.astro import two_body_dynamics
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.astro import element_conversion



# Additional problem data
global_frame_orientation = 'ECLIPJ2000'


# Control booleans
arrival_from_SOI_edge_case = False
arrival_from_ganymede_case = False


# Problem parameters to set
departure_epoch_days = 11200.  # days
time_of_flight = 0.3  # days
choose_moon = "Europa"
arrival_pericenter_altitude = 2000e3  # m
# departure_pericenter_altitude = 100e3  # m
moon_pericenter_phase_angle = np.pi * 0.75 # rad
number_of_epochs_to_plot = 1000  # /
first_arc_arrival_range_angle_wrt_moon_velocity = -np.pi/4

if arrival_from_SOI_edge_case:
    time_of_flight = 85  # days
    moon_pericenter_phase_angle = np.pi * 0.8  # rad
    number_of_epochs_to_plot = 10000  # /
if arrival_from_ganymede_case:
    time_of_flight = 0.6  # days
    choose_moon = "Ganymede"
    moon_pericenter_phase_angle = np.pi * 0.8  # rad
    number_of_epochs_to_plot = 1000  # /

# radii and epochs calculation
moon_radius = galilean_moons_data[choose_moon]['Radius']  # m
arrival_pericenter_radius = jupiter_radius + arrival_pericenter_altitude  # m
departure_moon_SOI_radius = galilean_moons_data[choose_moon]['SOI_Radius']  # moon_radius + departure_pericenter_altitude  # m
departure_epoch = departure_epoch_days * constants.JULIAN_DAY   # s
arrival_epoch = departure_epoch + time_of_flight * constants.JULIAN_DAY  # s


# Initial state vector
initial_state_moon_center = spice_interface.get_body_cartesian_state_at_epoch(
    target_body_name=choose_moon,
    observer_body_name="Jupiter",
    reference_frame_name=global_frame_orientation,
    aberration_corrections="NONE",
    ephemeris_time=departure_epoch)
# moon_departure_range_vector = np.array([-initial_state_moon_center[1],initial_state_moon_center[0],0., 0.,0.,0.])
# angle_to_rotate_departure_moon_vector = np.pi/6

# FLYBY ###############################################################################################################
# Velocity vectors
moon_velocity = initial_state_moon_center[3:6]
first_arc_arrival_velocity = np.array([-19965.64522041,   1036.27090357,    -42.70934684])

# Flyby angles
phi_2_flyby = np.arccos(np.dot(unit_vector(-moon_velocity), unit_vector(first_arc_arrival_velocity)))
delta_angle_flyby = np.pi + first_arc_arrival_range_angle_wrt_moon_velocity
impact_parameter_B_flyby = departure_moon_SOI_radius * np.sin(phi_2_flyby - delta_angle_flyby)

# Flyby parameters
mu_moon_flyby = galilean_moons_data[choose_moon]['mu']
safety_altitude_flyby = 100e3  # m

# Flyby outcomes
recurring_term = (impact_parameter_B_flyby**2 * LA.norm(first_arc_arrival_velocity)**4 / mu_moon_flyby**2)
pericenter_radius_flyby = mu_moon_flyby / LA.norm(first_arc_arrival_velocity)**2 * (np.sqrt(1 + recurring_term) + 1)
if pericenter_radius_flyby < galilean_moons_data[choose_moon]['Radius'] + safety_altitude_flyby:
    warnings.warn('MOON IMPACT - FLYBY FAILED')
alpha_angle_flyby = 2 * np.arcsin(1 / np.sqrt(1 + recurring_term))
beta_angle_flyby = phi_2_flyby + alpha_angle_flyby/2 - np.pi/2
#######################################################################################################################



# DONE: moon departure range vector is R_SOI long and directed 2pi-delta wrt -moon_vel

moon_departure_range_vector = ccw_rotation_z(unit_vector(-moon_velocity),
                                             2 * np.pi - delta_angle_flyby) * departure_moon_SOI_radius
# moon_departure_range_vector = np.array([initial_state_moon_center[0], initial_state_moon_center[1], 0., 0., 0., 0.])
# moon_departure_range_vector = moon_departure_range_vector / LA.norm(moon_departure_range_vector) * departure_moon_SOI_radius

initial_state = initial_state_moon_center + np.concatenate((moon_departure_range_vector, np.array([0,0,0])), axis=0)

if arrival_from_SOI_edge_case:
    initial_state = np.array([jupiter_SOI_radius, 0.,0., 0.,0.,0.])

# moon departure velocity vector is first arc arrival velocity rotated counterclockwise by alpha
# departure velocity is set. do algorithm that finds best ToF or rot angle

# Find ToF and moon pericenter phase angle


# Final state vector
phi = moon_pericenter_phase_angle
# clockwise_rotation_matrix = np.array([[np.cos(phi), np.sin(phi), 0.], [-np.sin(phi), np.cos(phi), 0.], [0., 0., 1.]])
# counterclockwise_rotation_matrix = np.array([[np.cos(phi), -np.sin(phi), 0.], [np.sin(phi), np.cos(phi), 0.], [0., 0., 1.]])
# rotation_matrix = counterclockwise_rotation_matrix
# final_state = rotation_matrix @ ((initial_state_moon_center[0:3] / LA.norm(initial_state_moon_center[0:3])).reshape((3,1))) * arrival_pericenter_radius
final_state = ccw_rotation_z(initial_state_moon_center[0:3] / LA.norm(initial_state_moon_center[0:3]),
                             phi) * arrival_pericenter_radius

# Lambert targeter initialisation
lambertTargeter = two_body_dynamics.LambertTargeterIzzo(
    initial_state[:3], final_state[:3], arrival_epoch - departure_epoch, central_body_gravitational_parameter)

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
print(f'Final f.p.a. : {arrival_flight_path_angle} rad    {arrival_flight_path_angle*180/np.pi} degrees')
print(f'Flyby altitude: {(pericenter_radius_flyby-moon_radius)/1e3} km')

fig = plt.figure()
ax = plt.axes(projection='3d')
xline = cartesian_elements_lambert[:,0]
yline = cartesian_elements_lambert[:,1]
zline = cartesian_elements_lambert[:,2]
ax.plot3D(xline, yline, zline, 'gray')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title(f'Lambert Trajectory from {choose_moon} to Jupiter')

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
x_0 = initial_state_moon_center[0]
y_0 = initial_state_moon_center[1]
z_0 = initial_state_moon_center[2]
x = x_0 + moon_radius * np.cos(u)*np.sin(v)
y = y_0 + moon_radius * np.sin(u)*np.sin(v)
z = z_0 + moon_radius * np.cos(v)
ax.plot_wireframe(x, y, z, color="b")
plt.show()

