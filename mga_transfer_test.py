import numpy as np
from tudatpy.kernel.trajectory_design import transfer_trajectory
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.util import result2array
from tudatpy.kernel import constants


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
transfer_body_order = ['Jupiter', 'Europa', 'Jupiter', 'Callisto']
departure_semi_major_axis = np.inf
departure_eccentricity = 0.
callisto_pericenter = 1869000  # km
Jupiter_radius = 69911  # km
final_orbit_pericenter_altitude = 2000  # km
altitude_addition = 3000  # km
arrival_semi_major_axis = (final_orbit_pericenter_altitude + altitude_addition + Jupiter_radius) / 0.02 *1e3
arrival_eccentricity = 0.98
leg_type = transfer_trajectory.unpowered_unperturbed_leg_type

transfer_leg_settings, transfer_node_settings = transfer_trajectory.mga_transfer_settings(
    transfer_body_order,
    leg_type,
    departure_orbit = ( departure_semi_major_axis, departure_eccentricity ),
    arrival_orbit = ( arrival_semi_major_axis, arrival_eccentricity) )

transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
    bodies,
    transfer_leg_settings,
    transfer_node_settings,
    transfer_body_order,
    central_body )

julian_day = constants.JULIAN_DAY

node_times = list( )
node_times.append( ( -789.8117 - 0.5 ) * julian_day )
node_times.append( node_times[ 0 ] + 158.302027105278 * julian_day )
node_times.append( node_times[ 1 ] + 449.385873819743 * julian_day )
node_times.append( node_times[ 2 ] + 54.7489684339665 * julian_day )
node_times.append( node_times[ 3 ] + 1024.36205846918 * julian_day )
node_times.append( node_times[ 4 ] + 4552.30796805542 * julian_day )

leg_free_parameters = list( )
for i in transfer_leg_settings:
    leg_free_parameters.append( np.zeros(0))

node_free_parameters = list( )
for i in transfer_node_settings:
    node_free_parameters.append( np.zeros(0))



transfer_trajectory_object.evaluate( node_times, leg_free_parameters, node_free_parameters )

delta_v = transfer_trajectory_object.delta_v                   # Total Delta V [m/s]
time_of_flight = transfer_trajectory_object.time_of_flight     # Total time of flight [s]
delta_v_per_leg = transfer_trajectory_object.delta_v_per_leg   # List of Delta V's in each leg (here list of zeroes) [m/s]
delta_v_per_node = transfer_trajectory_object.delta_v_per_node # List of Delta V's at each node [m/s]

transfer_trajectory.print_parameter_definitions( transfer_leg_settings, transfer_node_settings )

state_history = result2array(transfer_trajectory_object.states_along_trajectory(500))