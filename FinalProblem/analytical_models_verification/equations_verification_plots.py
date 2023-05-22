# General imports
import os
import shutil
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import linalg as LA
from time import process_time as pt

from JupiterTrajectory_GlobalParameters import *
from handle_functions import *
from class_GalileanMoon import GalileanMoon

show_orbits = False
skip_orbits_to_show = 10

fontsize = 13
# Make everything big
plt.rc('font', size=fontsize)

exclude_samples = 2


current_dir = os.path.dirname(__file__)
simulation_directory = current_dir + '/VerificationOutput/'

integer_variable_range = [[0],
                          [4]]

decision_variable_range = np.loadtxt(simulation_directory + 'decision_variable_range.dat')
number_of_runs = int(np.loadtxt(simulation_directory + 'ancillary_information.dat')[1])

decision_variable_plotnames = [r'$V_{J_{\infty}}$ [m/s]', r'$\gamma_E$ [deg]', r'$t_{flyby}$ [s]', r'$B$ [m]']
decision_variable_names = ['InterplanetaryVelocity', 'EntryFpa', 'FlybyEpoch', 'ImpactParameter']
decision_variable_names_selected = ['InterplanetaryVelocity']
integer_variable_names = ['FlybyMoon']
# The entries of the vector 'trajectory_parameters' contains the following:
# * Entry 0: Arrival velocity in m/s
# * Entry 1: Flight path angle at atmospheric entry in degrees
    # * Entry 2: Moon of flyby: 1, 2, 3, 4  ([I, E, G, C])
# * Entry 3: Flyby epoch in Julian days since J2000 (MJD2000)
# * Entry 4: Impact parameter of the flyby in meters

for decision_variable_investigated in decision_variable_names_selected:
    variable_no = decision_variable_names.index(decision_variable_investigated)
    decision_variable_plotname = decision_variable_plotnames[variable_no]

    for current_moon_no in range(integer_variable_range[0][0],integer_variable_range[1][0]+1):
        current_moon_name = moons_optimization_parameter_dict[current_moon_no]
        if current_moon_name == 'NoMoon' and decision_variable_investigated in ['FlybyEpoch', 'ImpactParameter']:
            warnings.warn(f'Cannot show {decision_variable_investigated} with no moon in the environment. Data unavailable')
            continue
        current_moon = GalileanMoon(current_moon_name)


        B_param_boundary = current_moon.SOI_radius  # galilean_moons_data[current_moon_name]['SOI_Radius']
        flyby_epoch_boundary = current_moon.orbital_period  # galilean_moons_data[current_moon_name]['Orbital_Period']

        # decision_variable_range = [[5100., -4., first_january_2040_epoch, -B_param_boundary],
        #                            [6100., -0.1, first_january_2040_epoch + flyby_epoch_boundary, B_param_boundary]]
        decision_variable_range[1,2] = decision_variable_range[0,2] + flyby_epoch_boundary
        decision_variable_range[:,3] = np.array([-B_param_boundary, B_param_boundary])

        variable_linspace = np.linspace(decision_variable_range[0][variable_no],
                                        decision_variable_range[1][variable_no], number_of_runs)

        subdirectory = '/' + decision_variable_investigated + '/' + current_moon_name + '/'

        # NOMINAL Decision variables
        interplanetary_arrival_velocity = 5600  # m/s

        flight_path_angle_at_atmosphere_entry = -3  # degrees

        simulation_flyby_epoch = first_january_2040_epoch

        flyby_B_parameter = 1 / 2 * (galilean_moons_data[moons_optimization_parameter_dict[current_moon_no]]['Radius'] +
                                     galilean_moons_data[moons_optimization_parameter_dict[current_moon_no]]['SOI_Radius'])

        fig, ax = plt.subplots(constrained_layout=True)
        fig_vel, ax_vel = plt.subplots(constrained_layout=True)

        fig_rsw, ax_rsw = plt.subplots(3,2, figsize=(8,8), constrained_layout=True)

        fig_depvar, ax_depvar = plt.subplots(6, figsize=(6,10), constrained_layout=True)
        fig_depvar_1, ax_depvar_1 = plt.subplots(3, figsize=(7,6), constrained_layout=True)
        fig_depvar_2, ax_depvar_2 = plt.subplots(3, figsize=(7,6), constrained_layout=True)

        fig_ecc, ax_ecc = plt.subplots(2, figsize=(5,6), constrained_layout=True, sharex='col')


        epoch_of_minimum_altitude_range = np.array([0.,0.])
        success_runs = 0
        for run in range(number_of_runs):
            output_path = simulation_directory + subdirectory
            try:
                state_difference = np.loadtxt(output_path + 'numerical_analytical_state_difference_' + str(run) + '.dat')
            except:
                continue
            success_runs = success_runs + 1
        success_runs = 1 if success_runs == 0 else success_runs

        final_numerical_eccentricity = np.zeros(success_runs)
        final_analytical_eccentricity = np.zeros(success_runs)
        eccentricity_difference = np.zeros(success_runs)
        variable_samples = np.zeros(success_runs)

        cmap = cm.get_cmap('viridis', success_runs)
        color_step = 1 / success_runs
        current_color = 0
        for run in range(number_of_runs):
            print(f'\nRun: {run}        Moon: {current_moon_name}')
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
            verification_epochs = state_difference[:,0]
            plotting_epochs = (verification_epochs - verification_epochs[0]) / constants.JULIAN_DAY

            position_norm_difference_km = LA.norm(cartesian_state_difference[exclude_samples:-exclude_samples-1, 0:3], axis=1) / 1e3
            velocity_norm_difference_km = LA.norm(cartesian_state_difference[exclude_samples:-exclude_samples-1, 3:6], axis=1) / 1e3
            ax.plot(plotting_epochs[exclude_samples:-exclude_samples-1], position_norm_difference_km, c=cmap(current_color))
            ax_vel.plot(plotting_epochs[exclude_samples:-exclude_samples-1], velocity_norm_difference_km, color=cmap(current_color))


            try:
                numerical_orbit_state = np.loadtxt(output_path + 'simulation_state_history_' + str(run) + '.dat')
                analytical_orbit_state = np.loadtxt(output_path + 'analytical_state_history_' + str(run) + '.dat')
            except:
                continue

            numerical_cartesian_orbit_state = numerical_orbit_state[:, 1:]
            numerical_epochs = numerical_orbit_state[:,0]
            x_pos_num, y_pos_num, z_pos_num = numerical_cartesian_orbit_state[:, 0], numerical_cartesian_orbit_state[:, 1], numerical_cartesian_orbit_state[:, 2]

            analytical_cartesian_orbit_state = analytical_orbit_state[:, 1:]
            x_pos_anal, y_pos_anal, z_pos_anal = analytical_cartesian_orbit_state[:, 0], analytical_cartesian_orbit_state[:,1], analytical_cartesian_orbit_state[:, 2]

            # Eccentricity
            final_numerical_eccentricity[current_color] =  LA.norm(eccentricity_vector_from_cartesian_state(numerical_cartesian_orbit_state[-1,:]))
            final_analytical_eccentricity[current_color] = LA.norm(eccentricity_vector_from_cartesian_state(analytical_cartesian_orbit_state[-1,:]))
            eccentricity_difference[current_color] = final_numerical_eccentricity[current_color] - final_analytical_eccentricity[current_color]


            # Error in RSW directions
            matching_cells = np.isin(numerical_epochs, verification_epochs).nonzero()[0]
            numerical_cartesian_orbit_state_for_comparison = numerical_cartesian_orbit_state[matching_cells,:]
            rsw_state_difference_km = rsw_state_from_cartesian(numerical_cartesian_orbit_state_for_comparison, cartesian_state_difference) / 1e3

            for row in range(3):
                for column in range(2):
                    ax_rsw[row,column].plot(plotting_epochs[exclude_samples:-exclude_samples-1], rsw_state_difference_km[exclude_samples:-exclude_samples-1, row + 3 * column], c=cmap(current_color))

            # Dependent variable error
            try:
                aerocapture_dependent_variable_difference_history = np.loadtxt(output_path + 'aerocapture_dependent_variable_difference_' + str(run) + '.dat')
            except:
                continue

            aerocapture_epochs = aerocapture_dependent_variable_difference_history[:,0]
            aerocapture_epochs_plot = aerocapture_epochs - aerocapture_epochs[0]
            aerocapture_dependent_variable_difference = aerocapture_dependent_variable_difference_history[:,1:]
            number_of_dependent_variables = len(aerocapture_dependent_variable_difference[0,:])

            altitude_difference = aerocapture_dependent_variable_difference[:,0]
            fpa_difference = aerocapture_dependent_variable_difference[:,1]
            airspeed_difference = aerocapture_dependent_variable_difference[:,2]
            density_difference = aerocapture_dependent_variable_difference[:,3]
            drag_difference = aerocapture_dependent_variable_difference[:,4]
            lift_difference = aerocapture_dependent_variable_difference[:,5]

            for index in range(number_of_dependent_variables):
                ax_depvar[index].plot(aerocapture_epochs_plot, aerocapture_dependent_variable_difference[:,index])

            ax_depvar_1[0].plot(aerocapture_epochs_plot, altitude_difference, c=cmap(current_color))
            ax_depvar_1[1].plot(aerocapture_epochs_plot, fpa_difference, c=cmap(current_color))
            ax_depvar_1[2].plot(aerocapture_epochs_plot, airspeed_difference, c=cmap(current_color))

            ax_depvar_2[0].plot(aerocapture_epochs_plot, density_difference, c=cmap(current_color))
            ax_depvar_2[1].plot(aerocapture_epochs_plot, drag_difference, c=cmap(current_color))
            ax_depvar_2[2].plot(aerocapture_epochs_plot, lift_difference, c=cmap(current_color))

            # identify minimum altitude and epoch of occurrence
            numerical_sim_distance_vector = LA.norm(numerical_cartesian_orbit_state[:, 0:3], axis=1)
            minimum_altitude_cell = list(np.where(numerical_sim_distance_vector == min(numerical_sim_distance_vector))[0])
            epoch_of_minimum_altitude = plotting_epochs[minimum_altitude_cell]

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

            if show_orbits and run%int(skip_orbits_to_show) == 0:
                # Plot 3-D Trajectory
                fig_orbit = plt.figure()
                ax_orbit = plt.axes(projection='3d')

                ax_orbit = plot_jupiter_and_galilean_orbits(ax_orbit, plot_orbits=False, title_addition=f'Decision variable: {decision_variable_investigated}     Value: {variable_linspace[run]}         Moon: {current_moon_name}')
                ax_orbit = plot_galilean_moon(ax_orbit, current_moon_name, simulation_flyby_epoch)
                ax_orbit = set_axis_limits(ax_orbit)
                ax_orbit.plot3D(x_pos_num, y_pos_num, z_pos_num, color='gray')
                ax_orbit.plot3D(x_pos_anal, y_pos_anal, z_pos_anal, color ='green', linestyle='--')

                plt.show()

            # if current_color == 0:
            #     v_min_colorbar, v_max_colorbar = variable_linspace[run], variable_linspace[run]
            #
            # if variable_linspace[run] < v_min_colorbar:
            #     v_min_colorbar = variable_linspace[run]
            # if variable_linspace[run] > v_max_colorbar:
            #     v_max_colorbar = variable_linspace[run]

            variable_samples[current_color] = variable_linspace[run]

            # current_color = current_color + color_step
            current_color = current_color + 1

        v_min_colorbar, v_max_colorbar = min(variable_samples), max(variable_samples)

        ax_ecc[0].plot(variable_samples, final_numerical_eccentricity, label='Numerical')
        ax_ecc[0].plot(variable_samples, final_analytical_eccentricity, label='Semi-analytical')
        ax_ecc[0].legend()
        ax_ecc[0].set(ylabel='Final eccentricity')
        ax_ecc[0].axhline(y=1, linestyle='--', color='gray', label=r'$e_{par}$')
        ax_ecc[0] = plot_grid(ax_ecc[0], fontsize)

        ax_ecc[1].plot(variable_samples, eccentricity_difference)
        ax_ecc[1].set(ylabel='Eccentricity difference')
        ax_ecc[1] = plot_grid(ax_ecc[1], fontsize)
        fig_ecc.supxlabel(decision_variable_plotname)
        fig_ecc.suptitle(f'Moon: {current_moon_name}')

        # plot vertical axis on aggregated plots that defines minimum altitude
        # ax.axvline(x=epoch_of_minimum_altitude_range[0], linestyle='--', color='grey')
        # ax.axvline(x=epoch_of_minimum_altitude_range[1], linestyle='--', color='grey')
        # fig.suptitle(f'Moon: {current_moon_name}')

        # ax_vel.axvline(x=epoch_of_minimum_altitude_range[0], linestyle='--', color='grey')
        # ax_vel.axvline(x=epoch_of_minimum_altitude_range[1], linestyle='--', color='grey')
        # fig_vel.suptitle(f'Moon: {current_moon_name}')


        rsw_ylabels = np.asarray([['R [km]', 'S [km]', 'W [km]'], ['V$_R$ [km/s]', 'V$_S$ [km/s]', 'V$_W$ [km/s]']]).T
        for row in range(3):
            for column in range(2):
                ax_rsw[row,column] = plot_grid(ax_rsw[row,column], fontsize)
                ax_rsw[row, column].set(xlabel='Elapsed time [days]', ylabel=r'' + rsw_ylabels[row, column])

        depvar_labels = ['Altitude [m]', r'$\gamma$ [deg]', 'Airspeed [m/s]', r'$\rho$ [kg/m$^3$]', r'$a_{D}$ [m/s$^2$]', r'$a_{L}$ [m/s$^2$]']
        for index in range(len(depvar_labels)):
            ax_depvar[index].set(xlabel='Elapsed time [s]', ylabel=depvar_labels[index])
            ax_depvar[index] = plot_grid(ax_depvar[index], fontsize)

        for index in range(3):
            ax_depvar_1[index].set(ylabel=depvar_labels[index], xlabel='Elapsed time [s]')
            ax_depvar_1[index] = plot_grid(ax_depvar_1[index], fontsize)
            ax_depvar_2[index].set(ylabel=depvar_labels[index + 3], xlabel='Elapsed time [s]')
            ax_depvar_2[index] = plot_grid(ax_depvar_2[index], fontsize)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=v_min_colorbar, vmax=v_max_colorbar))
        plots_to_colorbar = [(fig_depvar_1, ax_depvar_1), (fig_depvar_2, ax_depvar_2), (fig_rsw, ax_rsw), (fig_vel, ax_vel), (fig, ax)]
        for figure, axes in plots_to_colorbar:
            cbar = figure.colorbar(sm, ax=axes)
            cbar.set_label(decision_variable_plotname)
            figure.suptitle(f'Moon: {current_moon_name}')


        ax = plot_grid(ax, fontsize)
        ax.set_yscale('log')
        # ax.set_title('position error')
        ax.set_xlabel('Elapsed time [days]')
        ax.set_ylabel('Position error [km]')

        ax_vel = plot_grid(ax_vel, fontsize)
        ax_vel.set_yscale('log')
        # ax_vel.set_title('velocity error')
        ax_vel.set_xlabel('Elapsed time [days]')
        ax_vel.set_ylabel('Velocity error [km/s]')
        # plt.tight_layout()
        plt.show()
