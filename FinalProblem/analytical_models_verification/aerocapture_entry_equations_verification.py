# General imports
import os
import shutil

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
from JupiterTrajectory_GlobalParameters import *
import CapsuleEntryUtilities as Util
import class_AerocaptureNumericalProblem as ae_model
import class_InitialStateTargeting as ist


write_results_to_file = True
choose_model = 0 # zero is the default model, the final one. 1 is the most raw one, higher numbers are better ones.
integrator_settings_index = -4

current_dir = os.path.dirname(__file__)
# Load spice kernels
spice_interface.load_standard_kernels()


###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# # Set simulation flyby epoch
# simulation_start_epoch = 11293 * constants.JULIAN_DAY  # s
# simulation_flyby_epoch = 11293 * constants.JULIAN_DAY  # s
# # Atmospheric entry conditions
# flight_path_angle_at_atmosphere_entry = -3.1  # degrees
# interplanetary_arrival_velocity = 5600  # m/s
# moon_no = 3
# flyby_B_parameter = ...

MJD200_date = 66154  # 01/01/2040
J2000_date = MJD200_date - 51544
first_january_2040_epoch = J2000_date * constants.JULIAN_DAY


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


number_of_runs = 10

# decision_variable_investigated =
simulation_directory = current_dir + '/VerificationOutput'

aerocapture_problem = ae_model.AerocaptureNumericalProblem(decision_variable_range, choose_model,
                                                                       integrator_settings_index, do_flyby=True,
                                                                       arc_to_compute=-1)

for variable_no, decision_variable_investigated in enumerate(decision_variable_names):

    for current_moon_no in range(integer_variable_range[0][0],integer_variable_range[1][0]+1):
        current_moon = moons_optimization_parameter_dict[current_moon_no]

        B_param_boundary = galilean_moons_data[current_moon]['SOI_Radius']
        decision_variable_range = [[5100., -4., first_january_2040_epoch, -B_param_boundary],
                                   [6100., -0.1, first_january_2040_epoch + flyby_epoch_boundary, B_param_boundary]]

        variable_linspace = np.linspace(decision_variable_range[0][variable_no],
                                        decision_variable_range[1][variable_no], number_of_runs)

        subdirectory = '/' + decision_variable_investigated + '/' + current_moon + '/'

        # NOMINAL Decision variables
        interplanetary_arrival_velocity = 5600  # m/s

        flight_path_angle_at_atmosphere_entry = -3  # degrees

        MJD200_date = 66154  # 01/01/2040
        J2000_date = MJD200_date - 51544
        first_january_2040_epoch = J2000_date * constants.JULIAN_DAY
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
                                                                        jupiter_arrival_v_inf=interplanetary_arrival_velocity
                                                                        )
            # analytical_simulation_start_epoch = initial_state_targeting_problem.get_simulation_start_epoch()
            # analytical_initial_state = initial_state_targeting_problem.get_initial_state()

            # analytical_cartesian_states = initial_state_targeting_problem.get_trajectory_cartesian_states()

            if not initial_state_targeting_problem.trajectory_is_feasible:
                print(f'trajectory unfeasible with {decision_variable_investigated} and moon {current_moon}. Decision variable value: {variable_linspace[run]}')
                continue
            elif initial_state_targeting_problem.escape_trajectory:
                print(f'Escape trajectory. currently set to skip it')
                continue


            # RUN NUMERICAL SIMULATION
            aerocapture_problem.fitness([interplanetary_arrival_velocity,
                                         flight_path_angle_at_atmosphere_entry,
                                         current_moon_no,
                                         simulation_flyby_epoch,
                                         flyby_B_parameter])
            dynamics_simulator = aerocapture_problem.get_last_run_dynamics_simulator()
            # numerical_simulation_start_epoch = aerocapture_problem.get_simulation_start_epoch()
            # numerical_initial_state = aerocapture_problem.get_initial_state()


            ### OUTPUT OF THE SIMULATION ###
            # Retrieve propagated state and dependent variables
            numerical_state_history = dynamics_simulator.state_history
            analytical_state_history = initial_state_targeting_problem.get_trajectory_state_history()
            # unprocessed_state_history = dynamics_simulator.unprocessed_state_history
            dependent_variable_history = dynamics_simulator.dependent_variable_history

            if write_results_to_file:
                save2txt(numerical_state_history, 'simulation_state_history_' + str(run) + '.dat', simulation_directory+subdirectory)
                save2txt(dependent_variable_history, 'simulation_dependent_variable_history_' + str(run) + '.dat', simulation_directory+subdirectory)

                save2txt(analytical_state_history, 'analytical_state_history_' + str(run) + '.dat', simulation_directory+subdirectory)


            # DO COMPARISONS IDK

            interpolation_epochs = np.array(list(numerical_state_history.keys()))
            output_path = simulation_directory+subdirectory

            Util.compare_models(numerical_state_history,
                                analytical_state_history,
                                interpolation_epochs,
                                output_path,
                                'numerical_analytical_state_difference_' + str(run) + '.dat')
