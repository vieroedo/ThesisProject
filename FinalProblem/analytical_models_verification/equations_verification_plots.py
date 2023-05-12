# General imports
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from time import process_time as pt

from JupiterTrajectory_GlobalParameters import *
from handle_functions import *

show_orbits = True
fontsize = 13
# Make everything big
plt.rc('font', size=fontsize)

start_from_item_no = 4


current_dir = os.path.dirname(__file__)


integer_variable_range = [[1],
                          [4]]

decision_variable_names = ['InterplanetaryVelocity', 'EntryFpa', 'FlybyEpoch', 'ImpactParameter']
integer_variable_names = ['FlybyMoon']
# The entries of the vector 'trajectory_parameters' contains the following:
# * Entry 0: Arrival velocity in m/s
# * Entry 1: Flight path angle at atmospheric entry in degrees
                                                                # * Entry 2: Moon of flyby: 1, 2, 3, 4  ([I, E, G, C])
# * Entry 3: Flyby epoch in Julian days since J2000 (MJD2000)
# * Entry 4: Impact parameter of the flyby in meters

cmap = plt.get_cmap('viridis')

number_of_runs = 50
# decision_variable_investigated =
simulation_directory = current_dir + '/VerificationOutput'

for variable_no, decision_variable_investigated in enumerate(decision_variable_names):

    for current_moon_no in range(integer_variable_range[0][0],integer_variable_range[1][0]+1):
        current_moon = moons_optimization_parameter_dict[current_moon_no]

        MJD200_date = 66154  # 01/01/2040
        J2000_date = MJD200_date - 51544
        first_january_2040_epoch = J2000_date * constants.JULIAN_DAY

        B_param_boundary = galilean_moons_data[current_moon]['SOI_Radius']
        flyby_epoch_boundary = galilean_moons_data[current_moon]['Orbital_Period']
        decision_variable_range = [[5100., -4., first_january_2040_epoch, -B_param_boundary],
                                   [6100., -0.1, first_january_2040_epoch + flyby_epoch_boundary, B_param_boundary]]

        variable_linspace = np.linspace(decision_variable_range[0][variable_no],
                                        decision_variable_range[1][variable_no], number_of_runs)

        subdirectory = '/' + decision_variable_investigated + '/' + current_moon + '/'

        # NOMINAL Decision variables
        interplanetary_arrival_velocity = 5600  # m/s

        flight_path_angle_at_atmosphere_entry = -3  # degrees

        simulation_flyby_epoch = first_january_2040_epoch

        flyby_B_parameter = 1 / 2 * (galilean_moons_data[moons_optimization_parameter_dict[current_moon_no]]['Radius'] +
                                     galilean_moons_data[moons_optimization_parameter_dict[current_moon_no]]['SOI_Radius'])

        fig, ax = plt.subplots()

        fig_vel, ax_vel = plt.subplots()

        epoch_of_minimum_altitude_range = np.array([0.,0.])

        current_color = 0

        for run in range(number_of_runs):
            print(f'\nRun: {run}        Moon: {current_moon}')
            print(f'Decision variable: {decision_variable_investigated}     Value: {variable_linspace[run]}')

            if decision_variable_investigated == 'InterplanetaryVelocity':
                interplanetary_arrival_velocity = variable_linspace[run]
            elif decision_variable_investigated == 'EntryFpa':
                flight_path_angle_at_atmosphere_entry = variable_linspace[run]
            elif decision_variable_investigated == 'FlybyEpoch':
                simulation_flyby_epoch = variable_linspace[run]
            elif decision_variable_investigated == 'ImpactParameter':
                flyby_B_parameter = variable_linspace[run]
            else:
                raise Exception('wrong variable name or nonexisting variable')

            output_path = simulation_directory + subdirectory

            try:
                state_difference = np.loadtxt(output_path + 'numerical_analytical_state_difference_' + str(run) + '.dat')
            except:
                continue

            cartesian_state_difference = state_difference[:,1:]
            epochs = (state_difference[:,0] - state_difference[0,0]) / constants.JULIAN_DAY

            y_sam = LA.norm(cartesian_state_difference[start_from_item_no:,0:3], axis=1)/1e3
            ax.plot(epochs[start_from_item_no:],y_sam, c=cmap(current_color))
            ax_vel.plot(epochs[start_from_item_no:], LA.norm(cartesian_state_difference[start_from_item_no:,3:6], axis=1)/1e3, color=cmap(current_color))
            current_color = current_color+5

            try:
                numerical_orbit_state = np.loadtxt(output_path + 'simulation_state_history_' + str(run) + '.dat')
                analytical_orbit_state = np.loadtxt(output_path + 'analytical_state_history_' + str(run) + '.dat')
            except:
                continue

            numerical_cartesian_orbit_state = numerical_orbit_state[:, 1:]
            x_pos_num, y_pos_num, z_pos_num = numerical_cartesian_orbit_state[:, 0], numerical_cartesian_orbit_state[:, 1], numerical_cartesian_orbit_state[:, 2]

            analytical_cartesian_orbit_state = analytical_orbit_state[:, 1:]
            x_pos_anal, y_pos_anal, z_pos_anal = analytical_cartesian_orbit_state[:, 0], analytical_cartesian_orbit_state[:,1], analytical_cartesian_orbit_state[:, 2]

            # identify minimum altitude and epoch of occurrence
            numerical_sim_distance_vector = LA.norm(numerical_cartesian_orbit_state[:, 0:3], axis=1)
            minimum_altitude_cell = list(np.where(numerical_sim_distance_vector == min(numerical_sim_distance_vector))[0])
            epoch_of_minimum_altitude = epochs[minimum_altitude_cell]

            # assign new bundaries if epoch_of_minimum_altitude is outside the given range
            if not epoch_of_minimum_altitude_range[0] < epoch_of_minimum_altitude < epoch_of_minimum_altitude_range[1]:
                # the first run assigns the first value to both boundaries
                if epoch_of_minimum_altitude_range[0] == 0 and epoch_of_minimum_altitude_range[1] == 0:
                    epoch_of_minimum_altitude_range[1] = epoch_of_minimum_altitude
                    epoch_of_minimum_altitude_range[0] = epoch_of_minimum_altitude
                # updates boundaries according to where the value exceeds
                if epoch_of_minimum_altitude > epoch_of_minimum_altitude_range[1]:
                    epoch_of_minimum_altitude_range[1] = epoch_of_minimum_altitude
                if epoch_of_minimum_altitude < epoch_of_minimum_altitude_range[0]:
                    epoch_of_minimum_altitude_range[0] = epoch_of_minimum_altitude

            if show_orbits: # and run%5 == 0:
                # Plot 3-D Trajectory
                fig_orbit = plt.figure()
                ax_orbit = plt.axes(projection='3d')

                ax_orbit = plot_jupiter_and_galilean_orbits(ax_orbit, plot_orbits=False, title_addition=f'Decision variable: {decision_variable_investigated}     Value: {variable_linspace[run]}         Moon: {current_moon}')
                ax_orbit = plot_galilean_moon(ax_orbit, current_moon, simulation_flyby_epoch)
                ax_orbit = set_axis_limits(ax_orbit)
                ax_orbit.plot3D(x_pos_num, y_pos_num, z_pos_num, color='gray')
                ax_orbit.plot3D(x_pos_anal, y_pos_anal, z_pos_anal, color ='green', linestyle='--')

                fig_orbit.show()

        # plot vertical axis on aggregated plots that defines minimum altitude
        ax.axvline(x=epoch_of_minimum_altitude_range[0], linestyle='--', color='grey')
        ax.axvline(x=epoch_of_minimum_altitude_range[1], linestyle='--', color='grey')

        ax_vel.axvline(x=epoch_of_minimum_altitude_range[0], linestyle='--', color='grey')
        ax_vel.axvline(x=epoch_of_minimum_altitude_range[1], linestyle='--', color='grey')


        ax = plot_grid(ax, fontsize)
        ax_vel = plot_grid(ax_vel, fontsize)

        ax.set_yscale('log')
        # ax.set_title('position error')
        ax.set_xlabel('Elapsed time [days]')
        ax.set_ylabel('Position error [km]')

        ax_vel.set_yscale('log')
        # ax_vel.set_title('velocity error')
        ax_vel.set_xlabel('Elapsed time [days]')
        ax_vel.set_ylabel('Velocity error [km/s]')
        plt.tight_layout()
        plt.show()
