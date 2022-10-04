import matplotlib.pyplot as plt
import numpy as np

from handle_functions import *

# known issues:
# trajectory goes away from moon
# some oriented angles are not calculated with their true sign

# Set flyby epoch -> edit this to be initial epoch!!
first_arc_arrival_epoch_days = 11293.  # days
flyby_altitude = 100e3  # m

# Retrieve moon state at flyby epoch
moon_epoch_of_flyby = first_arc_arrival_epoch_days * constants.JULIAN_DAY  # s
moon_flyby_state = spice_interface.get_body_cartesian_state_at_epoch(
    target_body_name=choose_flyby_moon,
    observer_body_name="Jupiter",
    reference_frame_name=global_frame_orientation,
    aberration_corrections="NONE",
    ephemeris_time=moon_epoch_of_flyby)

# ASSUMPTION: Trajectory lies on the moon orbital plane

# Calculate first arc orbital plane based on the one of the moon
first_arc_angular_momentum_cap = np.cross(moon_flyby_state[0:3], moon_flyby_state[3:6])
first_arc_line_of_nodes_cap = np.cross(z_axis, first_arc_angular_momentum_cap)

# First arc initial known conditions
first_arc_departure_radius = jupiter_SOI_radius
first_arc_departure_velocity = interplanetary_arrival_velocity_in_jupiter_frame
radius_velocity_vectors_angle = np.pi - delta_angle_from_hohmann_trajectory

# First arc final known conditions
first_arc_arrival_radius = LA.norm(moon_flyby_state[0:3])

# Calculate first arc orbital energy
first_arc_orbital_energy = first_arc_departure_velocity **2 / 2 - central_body_gravitational_parameter / first_arc_departure_radius

# Calculate max allowed delta_hoh based on chosen initial velocity
frac_numerator = np.sqrt(2*first_arc_arrival_radius*(first_arc_orbital_energy*first_arc_arrival_radius + central_body_gravitational_parameter))
max_allowed_delta_hoh = np.arcsin(frac_numerator / (first_arc_departure_radius*first_arc_departure_velocity))

# End script if the angle exceeds the max value
if delta_angle_from_hohmann_trajectory > max_allowed_delta_hoh:
    raise Exception(fr'The chosen delta_angle_from_hohmann_trajectory is too big, max value allowed is of {max_allowed_delta_hoh*180/np.pi} degrees with an initial velocity of {first_arc_departure_velocity/1e3} km/s')

# Calculate angular momentum magnitude and build the vector
first_arc_angular_momentum = first_arc_departure_radius * first_arc_departure_velocity * np.sin(radius_velocity_vectors_angle)
first_arc_angular_momentum_vector = first_arc_angular_momentum * first_arc_angular_momentum_cap

# Calculate p, sma, and e of the first arc
first_arc_semilatus_rectum = first_arc_angular_momentum ** 2 / central_body_gravitational_parameter
first_arc_semimajor_axis = - central_body_gravitational_parameter / (2 * first_arc_orbital_energy)
first_arc_eccentricity = np.sqrt(1 - first_arc_semilatus_rectum/ first_arc_semimajor_axis)


# LEGACY: Gets determined later on
# flyby_pericenter = moon_radius + flyby_altitude


# ASSUMPTION: Arrival position is set to the moon's center, from that arrival velocity is determined
first_arc_arrival_position = moon_flyby_state[0:3]  #( moon_flyby_state[0:3] / LA.norm(moon_flyby_state[0:3]) ) * first_arc_arrival_radius
first_arc_arrival_velocity = np.sqrt(2*(first_arc_orbital_energy + central_body_gravitational_parameter/first_arc_arrival_radius))

# Calculate arrival fpa of the first arc. (For a counterclockwise trajectory it is negative since the flyby happens before arriving at the trajectory pericentre)
first_arc_arrival_fpa = -np.arccos(np.clip(first_arc_angular_momentum/(first_arc_arrival_radius * first_arc_arrival_velocity), -1, 1))
# Swipe fpa sign if the trajectory is clockwise
if delta_angle_from_hohmann_trajectory < 0:
    first_arc_arrival_fpa = - first_arc_arrival_fpa

# Find the first_arc_arrival_velocity_vector
first_arc_arrival_velocity_vector = rotation_matrix(first_arc_angular_momentum_vector, np.pi / 2 - first_arc_arrival_fpa) @ \
                                    unit_vector(first_arc_arrival_position).reshape(3,1)
first_arc_arrival_velocity_vector = first_arc_arrival_velocity_vector.reshape(3) * first_arc_arrival_velocity

# LEGACY: find first_arc_arrival_velocity_vector
# # find axis to rotate r to find velocity rotation axis
# arrival_position_rotation_axis = np.cross(first_arc_arrival_position,z_axis)
# arrival_position_rotation_axis = unit_vector(arrival_position_rotation_axis)
#
# # rotate r by that axis to find velocity rotation axis
# arrival_velocity_rotation_axis = rotation_matrix(arrival_position_rotation_axis,np.pi/2) @ unit_vector(first_arc_arrival_position).reshape(3,1)
# arrival_velocity_rotation_axis = arrival_velocity_rotation_axis.reshape(3)
#
# # rotate r_cap by velocity rotation axis to find direction of velocity
# first_arc_arrival_velocity_vector = rotation_matrix(arrival_velocity_rotation_axis,np.pi / 2 - first_arc_arrival_fpa) @ \
#                                     unit_vector(first_arc_arrival_position).reshape(3,1)
# first_arc_arrival_velocity_vector = first_arc_arrival_velocity_vector.reshape(3) * first_arc_arrival_velocity


# shall i really do it?
# here you can derive initial position wrt to the moon and the final position is at the moon itself -> DONE BELOW


# FLYBY ###################################################################################

# Retrieve flyby moon data: Radius, mu, velocity vector
moon_radius = galilean_moons_data[choose_flyby_moon]['Radius']
mu_moon = galilean_moons_data[choose_flyby_moon]['mu']
moon_velocity = moon_flyby_state[3:6]

# Calculate the pericenter of the flyby traj. using the set altitude
flyby_pericenter = moon_radius + flyby_altitude

# Calculate flyby incoming v_infinity
flyby_initial_velocity_vector = first_arc_arrival_velocity_vector - moon_velocity
flyby_initial_velocity = LA.norm(flyby_initial_velocity_vector)

# Calculate the axis normal to the flyby plane using v_infinity and v_moon
flyby_orbital_plane_normal_axis = unit_vector(np.cross(flyby_initial_velocity_vector,moon_velocity))

# Calculate the resulting bending angle on the velocity vector
flyby_alpha_angle = 2 * np.arcsin(1/(1+flyby_pericenter*flyby_initial_velocity**2/mu_moon))

# Rotate the incoming v_infinity by alpha to obtain the departing v_infinity
flyby_final_velocity_vector = rotation_matrix(flyby_orbital_plane_normal_axis, flyby_alpha_angle) @ flyby_initial_velocity_vector.reshape(3,1)
flyby_final_velocity_vector = flyby_final_velocity_vector.reshape(3)

###########################################################################################

# ASSUMPTION: flyby treated as point in space -> first arc final and second arc initial positions have to match

# Calculate the initial position and radius of the second arc
second_arc_departure_position = first_arc_arrival_position
second_arc_departure_radius = LA.norm(second_arc_departure_position)

# Calculate the initial velocity vector and magnitude as a result of the flyby
# second_arc_departure_velocity_vector = flyby_final_velocity_vector + moon_velocity
# second_arc_departure_velocity = LA.norm(second_arc_departure_velocity_vector)

# Calculate the orbital energy of the second arc
second_arc_orbital_energy = second_arc_departure_velocity**2/2 - central_body_gravitational_parameter / second_arc_departure_radius



# Set the arrival radius to be at the edge of Jupiter's atmosphere
arrival_pericenter_altitude = 2000e3  # m
second_arc_arrival_radius = jupiter_radius + arrival_pericenter_altitude

second_arc_angular_momentum_2 = LA.norm(np.cross(second_arc_departure_position, second_arc_departure_velocity_vector))
second_arc_arrival_velocity_2 = second_arc_angular_momentum_2/(second_arc_arrival_radius*np.cos(flight_path_angle_at_atmosphere_entry))

# Calculate the arrival velocity magnitude using the arrival radius and the orbital energy
second_arc_arrival_velocity = np.sqrt(2*(second_arc_orbital_energy + central_body_gravitational_parameter/second_arc_arrival_radius))

#OVERDETERMINED!!!!!!!!!!!!!!
second_arc_angular_momentum = second_arc_arrival_radius * second_arc_arrival_velocity * np.cos(flight_path_angle_at_atmosphere_entry)
second_arc_angular_momentum_2 = LA.norm(np.cross(second_arc_departure_position, second_arc_departure_velocity_vector))

second_arc_semilatus_rectum = second_arc_angular_momentum**2/central_body_gravitational_parameter
second_arc_semimajor_axis = - central_body_gravitational_parameter / (2*second_arc_orbital_energy)
second_arc_eccentricity = np.sqrt(1-second_arc_semilatus_rectum/second_arc_semimajor_axis)

second_arc_departure_fpa = -np.arccos(np.clip(second_arc_angular_momentum/(second_arc_departure_radius * second_arc_departure_velocity), -1, 1))

# ASSUMPTION: flyby doesn't change orbital plane
second_arc_departure_velocity_vector = rotation_matrix(first_arc_angular_momentum_vector, np.pi / 2 - second_arc_departure_fpa) @ unit_vector(second_arc_departure_position).reshape(3,1)
second_arc_departure_velocity_vector = second_arc_departure_velocity_vector.reshape(3) * second_arc_departure_velocity

# FLYBY GEOMETRY CHECK #################################################################################################

# ASSUMPTION: moon velocity vector lays on flyby plane

# Retrieve moon radius, gravitatonal parameter and velocity vector (DUPLICATED)
moon_radius = galilean_moons_data[choose_flyby_moon]['Radius']
mu_moon = galilean_moons_data[choose_flyby_moon]['mu']
moon_velocity = moon_flyby_state[3:6]

# Calculate flyby initial geometry
flyby_initial_velocity_vector = first_arc_arrival_velocity_vector - moon_velocity
flyby_initial_velocity = LA.norm(flyby_initial_velocity_vector)

flyby_final_velocity_vector = second_arc_departure_velocity_vector - moon_velocity

# Calculate the flyby angle needed to rotate the velocity vector of the first arc to match that one of the second arc
flyby_alpha_angle = np.arccos(np.dot(unit_vector(flyby_initial_velocity_vector),unit_vector(flyby_final_velocity_vector)))

# Calculate the required flyby pericenter to achieve the difference in velocity vectors between first and second arc
flyby_required_pericenter = mu_moon/flyby_initial_velocity**2 *(1/np.sin(flyby_alpha_angle/2) - 1)

# Check the flyby pericenter is above the moon radius, with the possibility of adding a minimum safety altitude
safety_altitude_flyby = 0.
print(f'Flyby altitude: {(flyby_required_pericenter-moon_radius)/1e3} km')
if flyby_required_pericenter < moon_radius + safety_altitude_flyby:
    warnings.warn(f'\nMOON IMPACT - FLYBY FAILED')

########################################################################################################################

second_arc_arrival_true_anomaly = true_anomaly_from_radius(second_arc_arrival_radius, second_arc_eccentricity, second_arc_semimajor_axis)
second_arc_departure_true_anomaly = true_anomaly_from_radius(second_arc_departure_radius, second_arc_eccentricity, second_arc_semimajor_axis)

second_arc_angular_momentum_vector = np.cross(second_arc_departure_position, second_arc_departure_velocity_vector)
if np.dot(second_arc_angular_momentum_vector, np.array([0, 0, 1])) > 0:
    second_arc_arrival_true_anomaly = - second_arc_arrival_true_anomaly
    second_arc_departure_true_anomaly = - second_arc_departure_true_anomaly

second_arc_phase_angle = second_arc_arrival_true_anomaly - second_arc_departure_true_anomaly

first_arc_arrival_true_anomaly = - true_anomaly_from_radius(first_arc_arrival_radius, first_arc_eccentricity, first_arc_semimajor_axis)
first_arc_departure_true_anomaly = - true_anomaly_from_radius(first_arc_departure_radius, first_arc_eccentricity, first_arc_semimajor_axis)

if...:
    ...

first_arc_phase_angle = first_arc_arrival_true_anomaly - first_arc_departure_true_anomaly

moon_angle_wrt_x_axis = np.arccos(np.dot(unit_vector(moon_flyby_state[0:3]),np.array([1,0,0])))
if np.dot(np.cross(np.array([1,0,0]), moon_flyby_state[0:3]),np.array([0,0,1])) < 0:
    moon_angle_wrt_x_axis = - moon_angle_wrt_x_axis + 2 * np.pi

# plot first arc
first_arc_number_of_points = 200
true_anomaly_vector_arc_1 = np.linspace(first_arc_departure_true_anomaly, first_arc_arrival_true_anomaly, first_arc_number_of_points)
radius_vector_arc_1 = radius_from_true_anomaly(true_anomaly_vector_arc_1, first_arc_eccentricity, first_arc_semimajor_axis)

true_anomaly_plot_arc_1 = np.linspace(moon_angle_wrt_x_axis-first_arc_phase_angle,moon_angle_wrt_x_axis, first_arc_number_of_points)

x_arc1, y_arc1 = cartesian_2d_from_polar(radius_vector_arc_1, true_anomaly_plot_arc_1)
z_arc1 = np.zeros(len(x_arc1))


second_arc_number_of_points = 200
true_anomaly_vector_arc_2 = np.linspace(second_arc_departure_true_anomaly, second_arc_arrival_true_anomaly, second_arc_number_of_points)
radius_vector_arc_2 = radius_from_true_anomaly(true_anomaly_vector_arc_2, second_arc_eccentricity, second_arc_semimajor_axis)



# moon_angle_wrt_x_axis = np.arctan2(LA.norm(np.cross(np.array([1,0,0]),moon_flyby_state[0:3])),np.dot(moon_flyby_state[0:3],np.array([1,0,0])))
true_anomaly_plot_arc_2 = np.linspace(moon_angle_wrt_x_axis,moon_angle_wrt_x_axis + second_arc_phase_angle, second_arc_number_of_points)

x_arc2, y_arc2 = cartesian_2d_from_polar(radius_vector_arc_2, true_anomaly_plot_arc_2)
z_arc2 = np.zeros(len(x_arc2))

first_arc_states = np.vstack((x_arc1, y_arc1, z_arc1)).T
first_arc_rotation_axis = unit_vector(np.cross(first_arc_angular_momentum_vector,z_axis))
first_arc_inclination = np.arccos(first_arc_angular_momentum_vector[2]/LA.norm(first_arc_angular_momentum_vector))
first_arc_rotation_matrix = rotation_matrix(first_arc_rotation_axis, -first_arc_inclination)

first_arc_rotated_states = rotate_vectors_by_given_matrix(first_arc_rotation_matrix,first_arc_states)
x_arc1 = first_arc_rotated_states[:,0]
y_arc1 = first_arc_rotated_states[:,1]
z_arc1 = first_arc_rotated_states[:,2]


second_arc_states = np.vstack((x_arc2, y_arc2, z_arc2)).T
second_arc_rotation_axis = unit_vector(np.cross(second_arc_angular_momentum_vector,z_axis))
second_arc_inclination = np.arccos(second_arc_angular_momentum_vector[2]/LA.norm(second_arc_angular_momentum_vector))
second_arc_rotation_matrix = rotation_matrix(second_arc_rotation_axis, -second_arc_inclination)

second_arc_rotated_states = rotate_vectors_by_given_matrix(second_arc_rotation_matrix,second_arc_states)
x_arc2 = second_arc_rotated_states[:,0]
y_arc2 = second_arc_rotated_states[:,1]
z_arc2 = second_arc_rotated_states[:,2]


fig = plt.figure()
ax = plt.axes(projection='3d')
# xline = cartesian_elements_lambert[:, 0]
# yline = cartesian_elements_lambert[:, 1]
# zline = cartesian_elements_lambert[:, 2]
ax.plot3D(x_arc1, y_arc1, z_arc1, 'gray')
ax.plot3D(x_arc2, y_arc2, z_arc2, 'gray')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('trajectory up to jupiter atmosphere')

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


# draw moon
u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
x_0 = moon_flyby_state[0]
y_0 = moon_flyby_state[1]
z_0 = moon_flyby_state[2]
x = x_0 + moon_radius * np.cos(u) * np.sin(v)
y = y_0 + moon_radius * np.sin(u) * np.sin(v)
z = z_0 + moon_radius * np.cos(v)
ax.plot_wireframe(x, y, z, color="b")


# origin = np.array([[0, 0, 0],[0, 0, 0]])
# V = np.concatenate(())
#     np.array([[1,1], [-2,2], [4,-7]])
#
# ax.quiver(*origin,V[:,0], V[:,1], color=['r','b','g'] )

# vectors for debugging
ax.plot3D([0., second_arc_departure_position[0]], [0., second_arc_departure_position[1]], [0., second_arc_departure_position[2]], 'red')
ax.plot3D([second_arc_departure_position[0], second_arc_departure_velocity_vector[0]*1e6+second_arc_departure_position[0]],
          [second_arc_departure_position[1], second_arc_departure_velocity_vector[1]*1e6+second_arc_departure_position[1]],
          [second_arc_departure_position[2], second_arc_departure_velocity_vector[2]*1e6+second_arc_departure_position[2]], 'blue')
ax.plot3D([first_arc_arrival_position[0], first_arc_arrival_velocity_vector[0]*1e6+first_arc_arrival_position[0]],
          [first_arc_arrival_position[1], first_arc_arrival_velocity_vector[1]*1e6+first_arc_arrival_position[1]],
          [first_arc_arrival_position[2], first_arc_arrival_velocity_vector[2]*1e6+first_arc_arrival_position[2]], 'green')
plt.show()


