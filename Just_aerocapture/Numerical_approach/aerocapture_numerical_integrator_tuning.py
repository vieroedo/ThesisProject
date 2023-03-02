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

use_benchmark = True
use_rkf_benchmark = False

# # if use_benchmark is True ####################################################
# generate_benchmarks = False
#
# playwith_benchmark = False
# silence_benchmark_related_plots = True
#
# plot_error_wrt_benchmark = True
#
# benchmark_portion_to_evaluate = 3  # 0, 1, 2, 3 (3 means all three together)
#
# # If you play with benchmarks
# lower_limit = 10e3
# upper_limit = 160e3
# no_of_entries = 31
#
# # If you set a single step_size
# choose_step_size = 40e3
#
# # If you set dedicated step sizes for case 3
# set_dedicated_step_sizes = True
# dedicated_step_sizes = [8e3, 4, 8e3]
#
# # For both cases
# reduce_step_size = 1
#
# change_coefficient_set = False
# choose_coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_45
#
# plot_fit_to_benchmark_errors = False
# choose_ref_value_cell = 11
# global_truncation_error_power = 8  # 7
# #####################################################################################


current_dir = os.path.dirname(__file__)

# Load spice kernels
spice_interface.load_standard_kernels()

# shape_parameters = [8.148730872315355,
#                     2.720324489288032,
#                     0.2270385167794302,
#                     -0.4037530896422072,
#                     0.2781438040896319,
#                     0.4559143679738996]

# Atmospheric entry conditions
atmospheric_entry_interface_altitude = 400e3  # m (DO NOT CHANGE - consider changing only with valid and sound reasons)
flight_path_angle_at_atmosphere_entry = -2.1  # degrees


###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Set simulation start epoch
simulation_start_epoch = 11293 * constants.JULIAN_DAY  # s

# Set vehicle properties
# capsule_density = 250.0  # kg m-3


###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

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
bodies.get_body('Capsule').mass = 2000  # kg

# Create aerodynamic coefficients interface (drag and lift only)
reference_area = 5.  # m^2
drag_coefficient = 1.2
lift_coefficient = 0.6
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area, [drag_coefficient, 0.0, lift_coefficient])  # [Drag, Side-force, Lift]
environment_setup.add_aerodynamic_coefficient_interface(
                bodies, 'Capsule', aero_coefficient_settings )


###########################################################################
# CREATE (CONSTANT) PROPAGATION SETTINGS ##################################
###########################################################################

# Retrieve termination settings
termination_settings = Util.get_termination_settings(simulation_start_epoch
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
current_propagator_settings = Util.get_propagator_settings(flight_path_angle_at_atmosphere_entry,
                                                           atmospheric_entry_interface_altitude,
                                                           bodies,
                                                           termination_settings,
                                                           dependent_variables_to_save,
                                                           current_propagator)

# from -5 to +2
number_of_settings = range(-5, 2)

output_folder = current_dir + '/SimulationOutput/IntegratorTuning/'
benchmark_output_path = current_dir + '/SimulationOutput/benchmarks/full_traj/' if use_benchmark else None
benchmark_output_path = current_dir + '/SimulationOutput/' if use_rkf_benchmark else benchmark_output_path

if use_benchmark:
    if use_rkf_benchmark:
        benchmark_history_matrix = np.loadtxt(benchmark_output_path + 'simulation_state_history_rkf.dat')
        benchmark_dependent_variable_history_matrix = np.loadtxt(
            benchmark_output_path + 'simulation_dependent_variable_history_rkf.dat')
    else:
        benchmark_history_matrix = np.loadtxt(benchmark_output_path + 'benchmark_1_states.dat')
        benchmark_dependent_variable_history_matrix = np.loadtxt(benchmark_output_path + 'benchmark_1_dependent_variables.dat')

    first_benchmark_state_history = dict(zip(benchmark_history_matrix[:,0], benchmark_history_matrix[:,1:]))
    first_benchmark_dependent_variable_history = dict(zip(benchmark_dependent_variable_history_matrix[:,0], benchmark_dependent_variable_history_matrix[:,1:]))

    benchmark_interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation=interpolators.extrapolate_at_boundary)

    benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        first_benchmark_state_history,
        benchmark_interpolator_settings)
    benchmark_dependent_variable_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        first_benchmark_dependent_variable_history,
        benchmark_interpolator_settings)

for settings_index in number_of_settings:

    # Create integrator settings
    current_integrator_settings = Util.get_integrator_settings(settings_index,
                                                               simulation_start_epoch)

    output_path = output_folder + '/tol_1e' + str(-10 + settings_index) + '/'

    t0 = pt()
    dynamics_simulator = numerical_simulation.SingleArcSimulator(
        bodies, current_integrator_settings, current_propagator_settings, print_dependent_variable_data=False)
    simulation_cpu_time = pt() - t0


    ### OUTPUT OF THE SIMULATION ###

    # Retrieve propagated state and dependent variables
    state_history = dynamics_simulator.state_history
    unprocessed_state_history = dynamics_simulator.unprocessed_state_history
    dependent_variable_history = dynamics_simulator.dependent_variable_history

    # Get the number of function evaluations (for comparison of different integrators)
    function_evaluation_dict = dynamics_simulator.cumulative_number_of_function_evaluations
    number_of_function_evaluations = list(function_evaluation_dict.values())[-1]
    # Add it to a dictionary
    dict_to_write = {'Number of function evaluations (ignore the line above)': number_of_function_evaluations}
    # Check if the propagation was run successfully
    propagation_outcome = dynamics_simulator.integration_completed_successfully
    dict_to_write['Propagation run successfully'] = propagation_outcome
    # Note the propagation time
    dict_to_write['Process time'] = simulation_cpu_time

    # Print the ancillary information
    print('\n### ANCILLARY SIMULATION INFORMATION ###')
    for (elem, (info, result)) in enumerate(dict_to_write.items()):
        if elem > 1:
            print(info + ': ' + str(result))

    # Save results to a file
    if write_results_to_file:
        save2txt(state_history, 'state_history.dat', output_path)
        save2txt(unprocessed_state_history, 'unprocessed_state_history.dat', output_path)
        save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)
        save2txt(dict_to_write, 'ancillary_simulation_info.txt', output_path)

    if use_benchmark:

        state_difference = dict()

        # NOTE: it extrapolates at borders
        for epoch in state_history.keys():
            state_difference[epoch] = state_history[epoch] - benchmark_state_interpolator.interpolate(epoch)

        if write_results_to_file:
            save2txt(state_difference, 'state_difference_wrt_benchmark.dat', output_path)

        if are_dependent_variables_to_save:
            # Initialize containers
            dependent_difference = dict()
            # Loop over the propagated dependent variables and use the benchmark interpolators
            for epoch in dependent_variable_history.keys():
                dependent_difference[epoch] = dependent_variable_history[
                                                  epoch] - benchmark_dependent_variable_interpolator.interpolate(epoch)
            # Write differences with respect to the benchmarks to files
            if write_results_to_file:
                save2txt(dependent_difference, 'dependent_variable_difference_wrt_benchmark.dat', output_path)
