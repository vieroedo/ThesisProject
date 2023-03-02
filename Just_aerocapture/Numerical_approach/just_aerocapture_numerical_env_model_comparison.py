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
import CapsuleEntryUtilities as Util


write_results_to_file = True  # when in doubt leave true (idk anymore what setting it to false does hehe)

fly_galileo = True


use_benchmark = True
# if use_benchmark is True ####################################################
generate_benchmarks = False

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

# shape_parameters = [8.148730872315355,
#                     2.720324489288032,
#                     0.2270385167794302,
#                     -0.4037530896422072,
#                     0.2781438040896319,
#                     0.4559143679738996]

# Atmospheric entry conditions
atmospheric_entry_interface_altitude = 450e3  # m (DO NOT CHANGE - consider changing only with valid and sound reasons)
flight_path_angle_at_atmosphere_entry = -2.7  # degrees


###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Set simulation start epoch
simulation_start_epoch = 11293 * constants.JULIAN_DAY  # s
# Set termination conditions
maximum_duration = 85 * constants.JULIAN_DAY  # s
# termination_altitude = 270.0E3  # m
# Set vehicle properties
# capsule_density = 250.0  # kg m-3


###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Set number of models
number_of_models = 3

# if use_def_model:
#     number_of_models = 22
#     first_model = 21

# Initialize dictionary to store the results of the simulation
simulation_results = dict()

# Set the interpolation step at which different runs are compared
output_interpolation_step = constants.JULIAN_DAY  # s

for model_test in range(number_of_models):

    # Define settings for celestial bodies
    bodies_to_create = ['Jupiter']
    # Define coordinate system
    global_frame_origin = 'Jupiter'
    global_frame_orientation = 'ECLIPJ2000'

    # Create body settings
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation)

    # Add Jupiter exponential atmosphere
    jupiter_scale_height = 27e3  # m      https://web.archive.org/web/20111013042045/http://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html
    jupiter_1bar_density = 0.16  # kg/m^3
    density_scale_height = jupiter_scale_height
    density_at_zero_altitude = jupiter_1bar_density
    body_settings.get('Jupiter').atmosphere_settings = environment_setup.atmosphere.exponential(
            density_scale_height, density_at_zero_altitude)

    # Maybe add it, yes, but later, cs now jupiter's already rotating
    # target_frame = 'IAU_Jupiter_Simplified'
    # target_frame_spice = "IAU_Jupiter"
    # body_settings.get('Jupiter').rotation_model_settings = environment_setup.rotation_model.simple_from_spice(global_frame_orientation, target_frame, target_frame_spice,simulation_start_epoch)

    # Create bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Create and add capsule to body system
    # NOTE TO STUDENTS: When making any modifications to the capsule vehicle, do NOT make them in this code, but in the
    # add_capsule_to_body_system function
    # Util.add_capsule_to_body_system(bodies,
    #                                 shape_parameters,
    #                                 capsule_density)


    # Create vehicle object
    bodies.create_empty_body('Capsule')

    # Set mass of vehicle
    if fly_galileo:
        bodies.get_body('Capsule').mass = Util.galileo_mass  # kg
    else:
        bodies.get_body('Capsule').mass = Util.vehicle_mass  # kg

    # Create aerodynamic coefficients interface (drag and lift only)
    if fly_galileo:
        reference_area = Util.galileo_ref_area  # m^2
        drag_coefficient = Util.galileo_cd
        lift_coefficient = Util.galileo_cl
    else:
        reference_area = Util.vehicle_reference_area  # m^2
        drag_coefficient = Util.vehicle_cd
        lift_coefficient = Util.vehicle_cl
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
            reference_area, [drag_coefficient, 0.0, lift_coefficient])  # [Drag, Side-force, Lift]
    environment_setup.add_aerodynamic_coefficient_interface(
                    bodies, 'Capsule', aero_coefficient_settings )


    ###########################################################################
    # CREATE (CONSTANT) PROPAGATION SETTINGS ##################################
    ###########################################################################

    # Retrieve termination settings
    termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                         maximum_duration,
                                                         galileo_termination_settings=fly_galileo
                                                         )
    # Retrieve dependent variables to save
    dependent_variables_to_save = Util.get_dependent_variable_save_settings()
    # Check whether there is any
    are_dependent_variables_to_save = False if not dependent_variables_to_save else True


    ###########################################################################
    # RUN SIMULATION #####################################
    ###########################################################################

    # Get current propagator, and define propagation settings
    current_propagator = propagation_setup.propagator.unified_state_model_quaternions


    settings_index = -4
    # Create integrator settings
    current_integrator_settings = Util.get_integrator_settings(settings_index,
                                                               simulation_start_epoch,
                                                               galileo_integration_settings=fly_galileo,
                                                               galileo_step_size=0.1)



    if use_benchmark:
        # Define benchmark interpolator settings to make a comparison between the two benchmarks
        benchmark_interpolator_settings = interpolators.lagrange_interpolation(
            8,boundary_interpolation = interpolators.extrapolate_at_boundary)

        benchmark_output_path = current_dir + '/ModelsOutput/benchmarks/full_traj/' if write_results_to_file else None

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
        propagator_settings = Util.get_propagator_settings(flight_path_angle_at_atmosphere_entry,
                                                           atmospheric_entry_interface_altitude,
                                                           bodies,
                                                           Util.get_termination_settings(simulation_start_epoch, galileo_termination_settings=fly_galileo),
                                                           dependent_variables_to_save,
                                                           current_propagator,
                                                           galileo_propagator_settings=fly_galileo,
                                                           model_choice=selected_model)

        if generate_benchmarks:


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


    current_propagator_settings = Util.get_propagator_settings(flight_path_angle_at_atmosphere_entry,
                                                               atmospheric_entry_interface_altitude,
                                                               bodies,
                                                               Util.get_termination_settings(simulation_start_epoch, galileo_termination_settings=fly_galileo),
                                                               dependent_variables_to_save,
                                                               current_propagator,
                                                               galileo_propagator_settings=fly_galileo)

    # Create Shape Optimization Problem object
    dynamics_simulator = numerical_simulation.SingleArcSimulator(
        bodies, current_integrator_settings, current_propagator_settings, print_dependent_variable_data=False )


    ### OUTPUT OF THE SIMULATION ###
    # Retrieve propagated state and dependent variables
    state_history = dynamics_simulator.state_history
    unprocessed_state_history = dynamics_simulator.unprocessed_state_history
    dependent_variable_history = dynamics_simulator.dependent_variable_history

    if write_results_to_file:
        save2txt(state_history, 'simulation_state_history.dat', current_dir + '/ModelsOutput')
        save2txt(dependent_variable_history, 'simulation_dependent_variable_history.dat', current_dir + '/ModelsOutput')
