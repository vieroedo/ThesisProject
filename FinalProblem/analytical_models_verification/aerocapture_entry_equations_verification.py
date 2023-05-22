# General imports
import os
import shutil
import warnings

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from time import process_time as pt


# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators

# Problem-specific imports
import handle_functions
from JupiterTrajectory_GlobalParameters import *
import CapsuleEntryUtilities as Util
import class_AerocaptureNumericalProblem as ae_model
import class_InitialStateTargeting as ist

check_folder_existence = True
write_results_to_file = True
choose_model = 0 # zero is the default model, the final one. 1 is the most raw one, higher numbers are better ones.
integrator_settings_index = -4

current_dir = os.path.dirname(__file__)
# Load spice kernels
spice_interface.load_standard_kernels()


###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

start_at_entry_interface = False
number_of_runs = 100

# MJD200_date = 66154  # 01/01/2040
# J2000_date = MJD200_date - 51544
# first_january_2040_epoch = J2000_date * constants.JULIAN_DAY


# B_parameter + flyby_epoch are connected actually

# moon_of_flyby = moons_optimization_parameter_dict[moon_no]
# flyby_B_parameter_boundaries = [galilean_moons_data[moon_of_flyby]['Radius'], galilean_moons_data[moon_of_flyby]['SOI_Radius']]
B_param_boundary = 0.
flyby_epoch_boundary = 0.
for i in range(1,5):
    i_moon = moons_optimization_parameter_dict[i]
    soi_rad = galilean_moons_data[i_moon]['SOI_Radius']
    period = galilean_moons_data[i_moon]['Orbital_Period']

    B_param_boundary = soi_rad if soi_rad > B_param_boundary else B_param_boundary
    flyby_epoch_boundary = period if period > flyby_epoch_boundary else flyby_epoch_boundary


# ###########################################################################
# # CREATE AEOCAPTURE PROBLEM ###############################################
# ###########################################################################

decision_variable_range = [[5100., -4., first_january_2040_epoch,                     -B_param_boundary],
                           [6100.,   -0.1, first_january_2040_epoch+flyby_epoch_boundary, B_param_boundary]]

integer_variable_range = [[0],
                          [4]]

decision_variable_names = ['InterplanetaryVelocity', 'EntryFpa', 'FlybyEpoch', 'ImpactParameter']
integer_variable_names = ['FlybyMoon']

# The entries of the vector 'trajectory_parameters' contains the following:
# * Entry 0: Arrival velocity in m/s
# * Entry 1: Flight path angle at atmospheric entry in degrees
                                                                # * Entry 2: Moon of flyby: 1, 2, 3, 4  ([I, E, G, C])
# * Entry 3: Flyby epoch in Julian days since J2000 (MJD2000)
# * Entry 4: Impact parameter of the flyby in meters


simulation_directory = current_dir + '/VerificationOutput/'

if check_folder_existence:
    does_folder_exists = os.path.exists(simulation_directory)
    shutil.rmtree(simulation_directory) if does_folder_exists else None
    check_folder_existence = False

if start_at_entry_interface:
    arc_to_compute = 12
else:
    arc_to_compute = -1


if write_results_to_file:
    NUMBER_OF_RUNS_KEY = 0
    ancillary_information = dict()
    ancillary_information[NUMBER_OF_RUNS_KEY] = np.array([number_of_runs])

    save2txt(ancillary_information, 'ancillary_information.dat',simulation_directory)
    # save2txt(, 'ancillary_information.dat',simulation_directory)
    np.savetxt(simulation_directory + 'decision_variable_range.dat', np.asarray(decision_variable_range))


for variable_no, decision_variable_investigated in enumerate(decision_variable_names):

    for current_moon_no in range(integer_variable_range[0][0],integer_variable_range[1][0]+1):
        current_moon = moons_optimization_parameter_dict[current_moon_no]
        if current_moon == 'NoMoon' and decision_variable_investigated in ['FlybyEpoch', 'ImpactParameter']:
            warnings.warn(f'Cannot evaluate {decision_variable_investigated} with no moon in the environment.')
            continue

        B_param_boundary = galilean_moons_data[current_moon]['SOI_Radius']
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
                if flyby_B_parameter == 0.:
                    continue
            else:
                raise Exception('wrong variable name or nonexisting variable')

            ###########################################################################
            # RUN SIMULATION ##########################################################
            ###########################################################################


            # are_dependent_variables_to_save = True

            # RUN ANALYTICAL THINGY
            initial_state_targeting_problem = ist.InitialStateTargeting(atmosphere_entry_fpa=flight_path_angle_at_atmosphere_entry,
                                                                        atmosphere_entry_altitude=atmospheric_entry_altitude,
                                                                        B_parameter=flyby_B_parameter, flyby_moon=current_moon,
                                                                        flyby_epoch=simulation_flyby_epoch,
                                                                        jupiter_arrival_v_inf=interplanetary_arrival_velocity,
                                                                        start_at_entry_interface=start_at_entry_interface
                                                                        )
            # DO NOT WRITE ANYTHING INBETWEEN THESE LINES
            if not initial_state_targeting_problem.trajectory_is_feasible:
                print(f'trajectory unfeasible with {decision_variable_investigated} and moon {current_moon} (NOTE: Ganymede is used if no moon is inserted). Decision variable value: {variable_linspace[run]}')
                continue
            elif initial_state_targeting_problem.escape_trajectory:
                print(f'Escape trajectory. currently set to skip it')
                continue

            aerocapture_initial_epoch = initial_state_targeting_problem.arcs_time_information[1, 0]
            simulation_start_epoch = initial_state_targeting_problem.get_simulation_start_epoch()


            # RUN NUMERICAL SIMULATION
            do_flyby = True if current_moon in ['Io','Europa', 'Ganymede', 'Callisto'] else False

            aerocapture_problem = ae_model.AerocaptureNumericalProblem(decision_variable_range, choose_model,
                                                                       integrator_settings_index, do_flyby=do_flyby,
                                                                       arc_to_compute=arc_to_compute)
            if do_flyby:
                aerocapture_problem.fitness([interplanetary_arrival_velocity,
                                             flight_path_angle_at_atmosphere_entry,
                                             current_moon_no,
                                             simulation_flyby_epoch,
                                             flyby_B_parameter])
            else:
                aerocapture_problem.fitness([interplanetary_arrival_velocity,
                                             flight_path_angle_at_atmosphere_entry,
                                             simulation_start_epoch])
            dynamics_simulator = aerocapture_problem.get_last_run_dynamics_simulator()
            # numerical_simulation_start_epoch = aerocapture_problem.get_simulation_start_epoch()
            # numerical_initial_state = aerocapture_problem.get_initial_state()


            ### OUTPUT OF THE SIMULATION ###
            # Retrieve propagated state and dependent variables
            numerical_state_history = dynamics_simulator.state_history
            # analytical_state_history = initial_state_targeting_problem.get_trajectory_state_history()
            analytical_state_history = initial_state_targeting_problem.get_trajectory_state_history_from_epochs(np.array(list(numerical_state_history.keys())))
            # unprocessed_state_history = dynamics_simulator.unprocessed_state_history
            numerical_dependent_variable_history = dynamics_simulator.dependent_variable_history
            aerocapture_dependent_variable_history = initial_state_targeting_problem.get_aerocapture_dependent_variable_history(aerocapture_initial_epoch)

            if write_results_to_file:
                save2txt(numerical_state_history, 'simulation_state_history_' + str(run) + '.dat', simulation_directory+subdirectory)
                save2txt(numerical_dependent_variable_history, 'simulation_dependent_variable_history_' + str(run) + '.dat', simulation_directory + subdirectory)

                save2txt(analytical_state_history, 'analytical_state_history_' + str(run) + '.dat', simulation_directory+subdirectory)
                save2txt(aerocapture_dependent_variable_history, 'aerocapture_dependent_variable_history_' + str(run) + '.dat', simulation_directory + subdirectory)


            # # DO COMPARISONS IDK
            # numerical_epochs = np.array(list(numerical_state_history.keys()))
            # analytical_epochs = np.array(list(analytical_state_history.keys()))
            #
            # exclude_cells = 1
            # # Get limit times at which both histories can be validly interpolated
            # interpolation_lower_limit = max(numerical_epochs[exclude_cells], analytical_epochs[exclude_cells])
            # interpolation_upper_limit = min(numerical_epochs[-exclude_cells-1], analytical_epochs[-exclude_cells-1])
            #
            # # Create vector of verification_epochs to be compared (boundaries are referred to the first case)
            # unfiltered_interpolation_epochs = numerical_epochs
            # unfiltered_interpolation_epochs = [n for n in unfiltered_interpolation_epochs if
            #                                    n <= interpolation_upper_limit]
            # interpolation_epochs = [n for n in unfiltered_interpolation_epochs if n >= interpolation_lower_limit]
            # interpolation_epochs = np.array(interpolation_epochs)

            exclude_cells = 1
            interpolation_epochs = handle_functions.get_interpolation_epochs(numerical_state_history, analytical_state_history, exclude_cells)

            output_path = simulation_directory+subdirectory

            Util.compare_models(numerical_state_history,
                                analytical_state_history,
                                interpolation_epochs,
                                output_path,
                                'numerical_analytical_state_difference_' + str(run) + '.dat')



            exclude_cells = 1
            # (ae_fpas, ae_velocities, ae_radii, ae_densities, ae_drag, ae_lift, ae_wall_hfx, ae_range_angles)
            analytical_ae_epochs = np.array(list(aerocapture_dependent_variable_history.keys()))
            aerocapture_quantities = np.vstack(list(aerocapture_dependent_variable_history.values()))
            ae_fpas = aerocapture_quantities[:, 0]
            ae_velocities = aerocapture_quantities[:, 1]
            ae_radii = aerocapture_quantities[:,2]
            ae_altitudes = ae_radii - jupiter_radius
            ae_densities = aerocapture_quantities[:,3]
            ae_drag = aerocapture_quantities[:,4]
            ae_lift = aerocapture_quantities[:,5]
            ae_wall_hfx = aerocapture_quantities[:,6]
            ae_range_angles = aerocapture_quantities[:,7]

            anal_depvar_values = np.vstack((ae_altitudes, ae_fpas, ae_velocities, ae_densities, ae_drag, ae_lift)).T
            analytical_dependent_variable_dictionary = dict(zip(analytical_ae_epochs, anal_depvar_values))


            numerical_cartesian_states = np.vstack(list(numerical_state_history.values()))
            dependent_variables = np.vstack(list(numerical_dependent_variable_history.values()))
            numerical_epochs = np.array(list(numerical_dependent_variable_history.keys()))
            ae_cells = np.where(numerical_epochs<analytical_ae_epochs[-1])[0]
            numerical_ae_epochs = numerical_epochs[ae_cells]
            # Slice to separate various dep variables and quantities
            aero_acc = dependent_variables[:, 0:3]
            grav_acc = dependent_variables[:, 3]
            altitude = dependent_variables[:, 4]
            flight_path_angle = dependent_variables[:, 5]
            airspeed = dependent_variables[:, 6]
            mach_number = dependent_variables[:, 7]
            atmospheric_density = dependent_variables[:, 8]
            inertial_fpa = dependent_variables[:, 9]
            inertial_velocity = LA.norm(numerical_cartesian_states[:,3:6], axis=1)
            drag_acc, lift_acc = Util.drag_lift_accelerations_from_aerodynamic(aero_acc, numerical_cartesian_states)
            drag_acc_mag, lift_acc_mag = LA.norm(drag_acc, axis=1), LA.norm(lift_acc, axis=1)
            # wall_heat_flux = Util.calculate_trajectory_heat_fluxes(atmospheric_density, airspeed, nose_radius=vehicle_nose_radius)


            num_depvar_values = np.vstack((altitude, inertial_fpa, inertial_velocity, atmospheric_density, drag_acc_mag, lift_acc_mag)).T
            numerical_dependent_variable_dictionary = dict(zip(numerical_ae_epochs, num_depvar_values[ae_cells,:]))

            interpolation_depvar_epochs = handle_functions.get_interpolation_epochs(numerical_dependent_variable_history, aerocapture_dependent_variable_history, exclude_cells)
            Util.compare_models(numerical_dependent_variable_dictionary,
                                analytical_dependent_variable_dictionary,
                                interpolation_depvar_epochs,
                                output_path,
                                'aerocapture_dependent_variable_difference_' + str(run) + '.dat')

