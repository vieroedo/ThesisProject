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


write_results_to_file = True  # when in doubt leave true (idk anymore what setting it to false does hehe)

fly_galileo = False
choose_model = 0 # zero is the default model, the final one. 1 is the most raw one, higher numbers are better ones.
integrator_settings_index = -4

use_benchmark = False
# if use_benchmark is True ####################################################
generate_benchmarks = True

silence_benchmark_related_plots = True

# Set dedicated step sizes for case 3
dedicated_step_sizes = [8e3, 4, 8e3]

plot_fit_to_benchmark_errors = False
choose_ref_value_cell = 11
global_truncation_error_power = 8  # 7
#####################################################################################


current_dir = os.path.dirname(__file__)

if fly_galileo:
    current_dir = current_dir + '/GalileoMission'

# Load spice kernels
spice_interface.load_standard_kernels()


# Atmospheric entry conditions
atmospheric_entry_interface_altitude = Util.atmospheric_entry_altitude  # m (DO NOT CHANGE - consider changing only with valid and sound reasons)
flight_path_angle_at_atmosphere_entry = -3.05  # degrees
interplanetary_arrival_velocity = 5630  # m/s

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Set simulation start epoch
simulation_start_epoch = 11293 * constants.JULIAN_DAY  # s


# ###########################################################################
# # CREATE AEOCAPTURE PROBLEM ###############################################
# ###########################################################################

decision_variable_range = [[0.],
                           [0.]]

inertial_state_deviation = np.array([0.00000,0.00000,0.00000,-8.51131,-2.95697,4.91416])
# inertial_state_deviation = np.zeros(6)

aerocapture_problem = ae_model.AerocaptureNumericalProblem(simulation_start_epoch, decision_variable_range,
                                                           choose_model, integrator_settings_index,
                                                           fly_galileo=fly_galileo, arc_to_compute=0,
                                                           initial_state_perturbation=inertial_state_deviation)
are_dependent_variables_to_save = True

###########################################################################
# RUN SIMULATION BENCHMARK ################################################
###########################################################################

if use_benchmark:
    # Define benchmark interpolator settings to make a comparison between the two benchmarks
    benchmark_interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation = interpolators.extrapolate_at_boundary)

    benchmark_output_path = current_dir + '/SimulationOutput/benchmarks/full_traj/' if write_results_to_file else None

    if generate_benchmarks:
        does_folder_exists = os.path.exists(benchmark_output_path)
        shutil.rmtree(benchmark_output_path) if does_folder_exists else None
        check_folder_existence = False

    # because of how
    benchmark_initial_state = np.zeros(6)

    benchmark_info = dict()

    benchmark_step_size = dedicated_step_sizes

    if fly_galileo:
        benchmark_step_size = 0.01

    t0 = pt()
    if generate_benchmarks:

        bodies = aerocapture_problem.get_bodies()
        propagator_settings = aerocapture_problem.create_propagator_settings(flight_path_angle_at_atmosphere_entry,
                                                                             interplanetary_arrival_velocity)
        benchmark_list, final_epoch, final_state = Util.generate_benchmarks(benchmark_step_size,
                                                                            simulation_start_epoch,
                                                                            bodies,
                                                                            propagator_settings,
                                                                            are_dependent_variables_to_save,
                                                                            benchmark_output_path,
                                                                            galileo_mission=fly_galileo)
        benchmark_info['step_sizes'] = benchmark_step_size

    else:
        first_bench = np.loadtxt(benchmark_output_path + 'benchmark_1_states.dat')
        first_bench_dep_var = np.loadtxt(benchmark_output_path + 'benchmark_1_dependent_variables.dat')
        second_bench = np.loadtxt(benchmark_output_path + 'benchmark_2_states.dat')
        second_bench_dep_var = np.loadtxt(benchmark_output_path + 'benchmark_2_dependent_variables.dat')
        benchmark_list = [dict(zip(first_bench[:,0], first_bench[:,1:])),
                          dict(zip(second_bench[:,0], second_bench[:,1:])),
                          dict(zip(first_bench_dep_var[:, 0], first_bench_dep_var[:, 1:])),
                          dict(zip(second_bench_dep_var[:, 0], second_bench_dep_var[:, 1:]))
                          ]


    benchmark_cpu_time = pt() - t0
    # Extract benchmark states
    first_benchmark_state_history = benchmark_list[0]
    second_benchmark_state_history = benchmark_list[1]
    # Create state interpolator for first benchmark
    benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(first_benchmark_state_history,
                                                                                            benchmark_interpolator_settings)


    # Compare benchmark states, returning interpolator of the first benchmark
    benchmark_state_difference = Util.compare_benchmarks(first_benchmark_state_history,
                                                         second_benchmark_state_history,
                                                         benchmark_output_path,
                                                         'benchmarks_state_difference.dat')

    # Extract benchmark dependent variables, if present
    if are_dependent_variables_to_save:
        first_benchmark_dependent_variable_history = benchmark_list[2]
        second_benchmark_dependent_variable_history = benchmark_list[3]
        # Create dependent variable interpolator for first benchmark
        benchmark_dependent_variable_interpolator = interpolators.create_one_dimensional_vector_interpolator(
            first_benchmark_dependent_variable_history,
            benchmark_interpolator_settings)

        # Compare benchmark dependent variables, returning interpolator of the first benchmark, if present
        benchmark_dependent_difference = Util.compare_benchmarks(first_benchmark_dependent_variable_history,
                                                                 second_benchmark_dependent_variable_history,
                                                                 benchmark_output_path,
                                                                 'benchmarks_dependent_variable_difference.dat')

    if not silence_benchmark_related_plots:
        fig1, ax1 = Util.plot_base_trajectory(first_benchmark_state_history)
        fig2, ax2 = Util.plot_time_step(first_benchmark_state_history)

        fig3, ax3 = Util.plot_base_trajectory(second_benchmark_state_history)
        fig4, ax4 = Util.plot_time_step(second_benchmark_state_history)


    benchmark_error = np.vstack(list(benchmark_state_difference.values()))
    bench_diff_epochs = np.array(list(benchmark_state_difference.keys()))
    bench_diff_epochs_plot = (bench_diff_epochs - bench_diff_epochs[0]) / constants.JULIAN_DAY
    benchmark_error = benchmark_error[2:-2,:]
    bench_diff_epochs_plot = bench_diff_epochs_plot[2:-2]
    bench_diff_epochs = bench_diff_epochs[2:-2]  # useless for now
    position_error = LA.norm(benchmark_error[:, 0:3], axis=1)
    # max_position_error = np.amax(position_error)

    if not silence_benchmark_related_plots:
        fig_bm, ax_bm = plt.subplots(figsize=(6, 5))
        ax_bm.plot(bench_diff_epochs_plot, position_error)
        ax_bm.set_yscale('log')

    if not silence_benchmark_related_plots:
        plt.show()

    if write_results_to_file and generate_benchmarks:
        save2txt(benchmark_info, 'ancillary_benchmark_info.txt', benchmark_output_path)

###########################################################################
# RUN SIMULATION ##########################################################
###########################################################################

aerocapture_problem.fitness([interplanetary_arrival_velocity, flight_path_angle_at_atmosphere_entry])
dynamics_simulator = aerocapture_problem.get_last_run_dynamics_simulator()


### OUTPUT OF THE SIMULATION ###
# Retrieve propagated state and dependent variables
state_history = dynamics_simulator.state_history
unprocessed_state_history = dynamics_simulator.unprocessed_state_history
dependent_variable_history = dynamics_simulator.dependent_variable_history

if write_results_to_file:
    save2txt(state_history, 'simulation_state_history.dat', current_dir + '/SimulationOutput')
    save2txt(dependent_variable_history, 'simulation_dependent_variable_history.dat', current_dir + '/SimulationOutput')
