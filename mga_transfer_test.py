import numpy as np
import matplotlib.pyplot as plt
from tudatpy.kernel.trajectory_design import transfer_trajectory
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.util import result2array
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface

#  J2000 = JD - 2451545


# Load spice kernels
spice_interface.load_standard_kernels()
# Define settings for celestial bodies
bodies_to_create = ['Jupiter', 'Io', 'Callisto', 'Europa', 'Ganymede']

# Define coordinate system
global_frame_origin = 'Jupiter'
global_frame_orientation = 'ECLIPJ2000' # check if you can use a better orientation
# Create body settings
body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                            global_frame_origin,
                                                            global_frame_orientation)


# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)
central_body = 'Jupiter'
transfer_body_order = ['Jupiter', 'Europa', 'Jupiter', 'Ganymede', 'Jupiter']
departure_semi_major_axis = np.inf
departure_eccentricity = 0.
callisto_pericenter = 1869000  # km
Jupiter_radius = 69911  # km
final_orbit_pericenter_altitude = 2000  # km
altitude_addition = 3000  # km
arrival_semi_major_axis = ((final_orbit_pericenter_altitude + Jupiter_radius * 2 + callisto_pericenter + altitude_addition) * 0.5) / 0.02 *1e3
arrival_eccentricity = 0.98
leg_type = transfer_trajectory.unpowered_unperturbed_leg_type
# leg_type = transfer_trajectory.dsm_velocity_based_leg_type

transfer_leg_settings, transfer_node_settings = transfer_trajectory.mga_transfer_settings(
    transfer_body_order,
    leg_type,
    minimum_pericenters = {'Jupiter': 69911e3,'Io': 2000e3, 'Europa': 1900e3, 'Ganymede': 2900e3, 'Callisto': 2750e3},
    departure_orbit = ( departure_semi_major_axis, departure_eccentricity ),
    arrival_orbit = ( arrival_semi_major_axis, arrival_eccentricity) )

transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
    bodies,
    transfer_leg_settings,
    transfer_node_settings,
    transfer_body_order,
    central_body )

julian_day = constants.JULIAN_DAY

default_trajectory_parameters = [5.6,# * Entry 0: Arrival velocity in km/s
                                 11206.,# * Entry 1: Initial epoch in Julian days since J2000
                                 50.,# * Entry 2: Period of Target Orbit in Julian days
                                 83.,# * Entry 3: Time-of-flight from the arrival at Jupiter to the flyby at the first moon in Julian days
                                 0.3,# * Entry 4: Time-of-flight from the flyby at the first moon to the atmospheric entry at Jupiter in Julian days
                                 5.0,# * Entry 5: Time-of-flight from the atmospheric entry at Jupiter to the pericenter-raise flyby in Julian days
                                 2,# * Entry 6: ordered sequences of flybys of vector [EI, GI, CI, GE, CE, CG]
                                 1]# * Entry 7: Indexes of [I, E, G, C]
node_times = list()
node_times.append( ( default_trajectory_parameters[1] ) * julian_day )  # departure epoch
node_times.append( node_times[ 0 ] + default_trajectory_parameters[3] * julian_day ) # ToF first arc - from beginning to first moon
node_times.append( node_times[ 1 ] + default_trajectory_parameters[4] * julian_day ) # ToF second arc - from first moon to "atmospheric entry" (no atmosphere included here)
node_times.append( node_times[ 2 ] + default_trajectory_parameters[5] * julian_day ) # ToF third arc - from atmospheric entry to second moon flyby (the aim here is to raise the pericenter of the final orbit around jupiter)
node_times.append( node_times[ 3 ] + default_trajectory_parameters[2] * julian_day )  # target orbit period

# Code for the DSM scenario (not suitable for the problem tho)
# leg_free_parameters = list( )
# leg_free_parameters.append( np.array( [ 0.0 ] ) )
# leg_free_parameters.append( np.array( [ 0.0 ] ) )
# leg_free_parameters.append( np.array( [ 0.0 ] ) )
# leg_free_parameters.append( np.array( [ 0.0 ] ) )
# leg_free_parameters.append( np.array( [ 0.0 ] ) )
#
#
# node_free_parameters = list( )
# node_free_parameters.append( np.array( [ 5600.0, 1 * 2.0 * 3.14159265358979, 0.0 ] ) )  # np.arccos( 2.0 * 0.498004040298 - 1.0 ) - 3.14159265358979 / 2.0
# node_free_parameters.append( np.array( [ 2000e3, 1.35077257078, 0.0 ] ) )
# node_free_parameters.append( np.array( [ 20000 + 6e8, 1.6, 2200.0 ] ) )
# node_free_parameters.append( np.array( [ 2900e3, 1.34317576594, 0.0 ] ) )
# node_free_parameters.append( np.array( [ ] ) )

leg_free_parameters = list( )
for i in transfer_leg_settings:
    leg_free_parameters.append( np.zeros(0))

node_free_parameters = list( )
for i in transfer_node_settings:
    node_free_parameters.append( np.zeros(0))

# transfer_trajectory.print_parameter_definitions(transfer_leg_settings, transfer_node_settings)

transfer_trajectory_object.evaluate( node_times, leg_free_parameters, node_free_parameters )

delta_v = transfer_trajectory_object.delta_v                   # Total Delta V [m/s]
time_of_flight = transfer_trajectory_object.time_of_flight     # Total time of flight [s]
delta_v_per_leg = transfer_trajectory_object.delta_v_per_leg   # List of Delta V's in each leg (here list of zeroes) [m/s]
delta_v_per_node = transfer_trajectory_object.delta_v_per_node # List of Delta V's at each node [m/s]

# Print the total DeltaV and time of Flight required for the MGA
print('Total Delta V of %.3f m/s and total Time of flight of %.3f days\n' % \
    (transfer_trajectory_object.delta_v, transfer_trajectory_object.time_of_flight / julian_day))

# Print the DeltaV required during each leg
print('Delta V per leg: ')
for i in range(len(transfer_body_order)-1):
    print(" - between %s and %s: %.3f m/s" % \
        (transfer_body_order[i], transfer_body_order[i+1], transfer_trajectory_object.delta_v_per_leg[i]))
print()

# Print the DeltaV required at each node
print('Delta V per node : ')
for i in range(len(transfer_body_order)):
    print(" - at %s: %.3f m/s" % \
        (transfer_body_order[i], transfer_trajectory_object.delta_v_per_node[i]))
print()

# Print transfer parameter definitions
print("Transfer parameter definitions:")
transfer_trajectory.print_parameter_definitions(transfer_leg_settings, transfer_node_settings)


# transfer_trajectory.print_parameter_definitions( transfer_leg_settings, transfer_node_settings )

# state_history = result2array(transfer_trajectory_object.states_along_trajectory(500))

# Extract the state history
state_history = transfer_trajectory_object.states_along_trajectory(500)
fly_by_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
state_history = result2array(state_history)
au = 1.5e11

# Plot the transfer
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111, projection='3d')
# Plot the trajectory from the state history
ax.plot(state_history[:, 1] / au, state_history[:, 2] / au, state_history[:, 3] / au)
# Plot the position of the nodes
ax.scatter(fly_by_states[0, 0] / au, fly_by_states[0, 1] / au, fly_by_states[0, 2] / au, color='blue', label='Jupiter SOI dep')
ax.scatter(fly_by_states[1, 0] / au, fly_by_states[1, 1] / au, fly_by_states[1, 2] / au, color='brown', label='1st moon fly-by')
ax.scatter(fly_by_states[2, 0] / au, fly_by_states[2, 1] / au, fly_by_states[2, 2] / au, color='brown', label='Jupiter fly-by')
ax.scatter(fly_by_states[3, 0] / au, fly_by_states[3, 1] / au, fly_by_states[3, 2] / au, color='green', label='2nd moon fly-by')
ax.scatter(fly_by_states[4, 0] / au, fly_by_states[4, 1] / au, fly_by_states[4, 2] / au, color='peru', label='Jupiter insertion')
# ax.scatter(fly_by_states[5, 0] / au, fly_by_states[5, 1] / au, fly_by_states[5, 2] / au, color='red', label='Saturn arrival')
# Plot the position of the Sun
# ax.scatter([0], [0], [0], color='orange', label='Sun')
# Add axis labels and limits
ax.set_xlabel('x wrt Sun [AU]')
ax.set_ylabel('y wrt Sun [AU]')
ax.set_zlabel('z wrt Sun [AU]')
ax.set_xlim([-4, -3])
ax.set_ylim([-4.5, -3.5])
ax.set_zlim([-2, 2])
# Put legend on the right
ax.legend(bbox_to_anchor=[1.15, 1])
plt.tight_layout()
plt.show()

# fig_3dtraj = plt.figure()
# ax_3dtraj = plt.axes(projection='3d')
# xline = state_history[:, 1]
# yline = state_history[:, 2]
# zline = state_history[:, 3]
# ax_3dtraj.plot3D(xline, yline, zline, 'orange', linewidth=3)
# # lowlim = -7e11
# # highlim = -4e11
# # ax_3dtraj.set_xlim([lowlim,highlim])
# # ax_3dtraj.set_ylim([lowlim,highlim])
# # ax_3dtraj.set_zlim([1e10,3.1e11])
# plt.show()

# # Extract the state history
# state_history = transfer_trajectory_object.states_along_trajectory(500)
# fly_by_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
# state_history = result2array(state_history)
# au = 1.5e11
#
# # Plot the state history
# fig = plt.figure(figsize=(8,5))
# ax = fig.add_subplot(111)
# ax.plot(state_history[:, 1] / au, state_history[:, 2] / au)
# ax.scatter(fly_by_states[0, 0] / au, fly_by_states[0, 1] / au, color='blue', label='Earth departure')
# ax.scatter(fly_by_states[1, 0] / au, fly_by_states[1, 1] / au, color='green', label='Earth fly-by')
# ax.scatter(fly_by_states[2, 0] / au, fly_by_states[2, 1] / au, color='brown', label='Venus fly-by')
# ax.scatter(fly_by_states[3, 0] / au, fly_by_states[3, 1] / au, color='brown')
# ax.scatter(fly_by_states[4, 0] / au, fly_by_states[4, 1] / au, color='grey', label='Mercury arrival')
# ax.scatter([0], [0], color='orange', label='Sun')
# ax.set_xlabel('x wrt Sun [AU]')
# ax.set_ylabel('y wrt Sun [AU]')
# ax.set_aspect('equal')
# ax.legend(bbox_to_anchor=[1, 1])
# plt.show()