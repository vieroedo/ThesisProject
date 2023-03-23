# General imports
import os
import shutil
import random

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
from tudatpy.kernel.astro import frame_conversion

# Problem-specific imports
from JupiterTrajectory_GlobalParameters import *
import CapsuleEntryUtilities as Util
import class_AerocaptureNumericalProblem as ae_model

from UncertaintyStudy_GlobalVariables import *


write_results_to_file = True  # when in doubt leave true (idk anymore what setting it to false does hehe)


# arc_to_compute = 1  # 0: from beginning to aerocapture.  1: from atm entry to exit.  12: from entry to end.  -1: everything
fly_galileo = False
choose_model = 0  # zero is the default model, the final one. 1 is the most raw one, higher numbers are better ones.
integrator_settings_index = -4

current_dir = os.path.dirname(__file__)

if fly_galileo:
    current_dir = current_dir + '/GalileoMission'

# Load spice kernels
spice_interface.load_standard_kernels()

# Atmospheric entry conditions
atmospheric_entry_interface_altitude = Util.atmospheric_entry_altitude  # m (DO NOT CHANGE - consider changing only with valid and sound reasons)
flight_path_angle_at_atmosphere_entry = -3.1  # degrees
interplanetary_arrival_velocity = 5600  # m/s

trajectory_parameters = [interplanetary_arrival_velocity,
                         flight_path_angle_at_atmosphere_entry]

decision_variable_range = [[5000, -5],
                           [6200, 0]]
# The entries of the vector 'trajectory_parameters' contains the following:
# * Entry 0:
# * Entry 1:
# * Entry ...:


# Save trajectory parameters
traj_paramteres_dict = {
    0: np.array(trajectory_parameters)
}
save2txt(traj_paramteres_dict, 'trajectory_parameters.dat', current_dir + '/UncertaintyAnalysisData/')

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Set simulation start epoch
simulation_start_epoch = 11293 * constants.JULIAN_DAY  # s

###########################################################################
###########################################################################
###########################################################################

# Set number of runs per uncertainty
number_of_runs_per_uncertainty = 100

# Set the interpolation step at which different runs are compared
output_interpolation_step = constants.JULIAN_DAY  # s
# output_interpolation_step =

random.seed(50)


uncertainties = list(uncertainties_dictionary.keys())  # list of uncertainty names
arcs_to_compute = list(uncertainties_dictionary.values())  # list of corresponding arcs


just_evaluate_some_uncertainties = True
uncertainties_to_evaluate = [*range(4,8)]
# uncertainties_to_evaluate = [9]
if just_evaluate_some_uncertainties == True:
    uncertainties_to_run = [uncertainties[i] for i in uncertainties_to_evaluate]
    arcs_to_compute_actually = [arcs_to_compute[i] for i in uncertainties_to_evaluate]
else:
    uncertainties_to_run = uncertainties
    arcs_to_compute_actually = arcs_to_compute


all_results = dict()
for uncertainty_nr, uncertainty in enumerate(uncertainties_to_run):
    actual_arc_to_compute = arcs_to_compute_actually[uncertainty_nr]

    # Create aerocapture problem with model and integration settings of choice.
    aerocapture_problem = ae_model.AerocaptureNumericalProblem(simulation_start_epoch,decision_variable_range,
                                                               choose_model, integrator_settings_index,
                                                               fly_galileo, arc_to_compute=actual_arc_to_compute)

    # Initialize dictionary to store the results of the simulation
    simulation_results = dict()
    perturbations = dict()

    for run in range(number_of_runs_per_uncertainty):
        print(f'\nRun: {run}  ' + uncertainty)
        perturbation = 0.0
        initial_state_deviation_rsw = np.zeros(6)


        # Initial position deviation in R, S, and W
        if run > 0 and (uncertainty in (uncertainties[0], uncertainties[8], uncertainties[18])):
            for i in range(3):
                initial_state_deviation_rsw[i] = random.gauss(0, multiple_initial_position_deviation_uncertainty)
            perturbation = initial_state_deviation_rsw[0:3]
        # Initial position R deviation
        if run > 0 and (uncertainty in (uncertainties[1], uncertainties[9], uncertainties[19])):
            if uncertainty_nr == 9:
                initial_position_deviation_uncertainty = 7000  # m
            initial_state_deviation_rsw[0] = random.gauss(0, initial_position_deviation_uncertainty)
            perturbation = initial_state_deviation_rsw[0]
        # Initial position S deviation
        if run > 0 and (uncertainty in (uncertainties[2], uncertainties[10], uncertainties[20])):
            initial_state_deviation_rsw[1] = random.gauss(0, initial_position_deviation_uncertainty)
            perturbation = initial_state_deviation_rsw[1]
        # Initial position W deviation
        if run > 0 and (uncertainty in (uncertainties[3], uncertainties[11], uncertainties[21])):
            initial_state_deviation_rsw[2] = random.gauss(0, initial_position_deviation_uncertainty)
            perturbation = initial_state_deviation_rsw[2]

        # Initial velocity deviation in R, S, and W
        if run > 0 and (uncertainty in (uncertainties[4], uncertainties[12], uncertainties[22])):
            for i in range(3):
                initial_state_deviation_rsw[i+3] = random.gauss(0, multiple_initial_velocity_deviation_uncertainty)
            perturbation = initial_state_deviation_rsw[3:6]
        # Initial velocity R deviation
        if run > 0 and (uncertainty in (uncertainties[5], uncertainties[13], uncertainties[23])):
            initial_state_deviation_rsw[3] = random.gauss(0, initial_velocity_deviation_uncertainty)
            perturbation = initial_state_deviation_rsw[3]
        # Initial velocity S deviation
        if run > 0 and (uncertainty in (uncertainties[6], uncertainties[14], uncertainties[24])):
            initial_state_deviation_rsw[4] = random.gauss(0, initial_velocity_deviation_uncertainty)
            perturbation = initial_state_deviation_rsw[4]
        # Initial velocity W deviation
        if run > 0 and (uncertainty in (uncertainties[7], uncertainties[15], uncertainties[25])):
            initial_state_deviation_rsw[5] = random.gauss(0, initial_velocity_deviation_uncertainty)
            perturbation = initial_state_deviation_rsw[5]

        body_settings = aerocapture_problem.create_body_settings_environment()

        # If you want to add environment uncertainties
        if run > 0 and False:
            # define variable for scaling factor
            scaling_constant = random.gauss(0, 33)  # 3sigma = 100
            # # define variables containing the existing ephemeris settings
            # unscaled_ephemeris_settings = body_settings.get("Earth").ephemeris_settings
            # # make new ephemeris settings
            # body_settings.get("Earth").ephemeris_settings = environment_setup.ephemeris.scaled_by_constant(
            #     unscaled_ephemeris_settings, scaling_constant, is_scaling_absolute=True)
            perturbation = scaling_constant

            # Define the scaling vector
            scaling_vector = [scaling_constant, scaling_constant, scaling_constant, 0, 0, 0]
            # Extract the unscaled ephemeris settings from Jupiter
            unscaled_ephemeris_settings = body_settings.get("Jupiter").ephemeris_settings
            # Create the scaled ephemeris settings and apply to the body "Jupiter"
            body_settings.get("Jupiter").ephemeris_settings = environment_setup.ephemeris.scaled_by_vector(
                unscaled_ephemeris_settings,
                scaling_vector, is_scaling_absolute=True)

        # Create bodies
        bodies = aerocapture_problem.create_bodies_environment(body_settings)

        # Aerodynamic coefficients C_D and C_L uncertainty
        if run > 0 and False:
            aerodynamic_coefficients_uncertainties = [0.1, 0.1]  # C_D, C_L
            perturbation = aerodynamic_coefficients_uncertainties
            aerodynamic_coefficients_variation = np.array([random.gauss(1, aerodynamic_coefficients_uncertainties[0]),
                      random.gauss(1, aerodynamic_coefficients_uncertainties[1])])
        # Drag coefficient C_D uncertainty
        elif run > 0 and False:
            aerodynamic_coefficients_uncertainties = [0.1, 0.]  # C_D, C_L
            perturbation = aerodynamic_coefficients_uncertainties
            aerodynamic_coefficients_variation = np.array([random.gauss(1, aerodynamic_coefficients_uncertainties[0]),
                                                           random.gauss(1, aerodynamic_coefficients_uncertainties[1])])
        # Lift coefficient C_L uncertainty
        elif run > 0 and False:
            aerodynamic_coefficients_uncertainties = [0., 0.1]  # C_D, C_L
            perturbation = aerodynamic_coefficients_uncertainties
            aerodynamic_coefficients_variation = np.array([random.gauss(1, aerodynamic_coefficients_uncertainties[0]),
                                                           random.gauss(1, aerodynamic_coefficients_uncertainties[1])])
        else:
            aerodynamic_coefficients_variation = None
        bodies = aerocapture_problem.add_aerodynamic_interface(bodies, aerodynamic_coefficients_variation)
        aerocapture_problem.set_bodies(bodies)

        initial_state_deviation_inertial = np.zeros(6)
        trajectory_initial_state = Util.get_initial_state(trajectory_parameters[1], atmospheric_entry_altitude,
                                                          trajectory_parameters[0])
        rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(trajectory_initial_state)
        if run > 0 and (uncertainty in [ uncertainties[i] for i in [0,1,2,3, 8,9,10,11, 18,19,20,21]]):
            initial_state_deviation_inertial[0:3] = rotation_matrix @ initial_state_deviation_rsw[0:3]
        elif run > 0 and (uncertainty in [ uncertainties[i] for i in [4,5,6,7, 12,13,14,15, 22,23,24,25]]):
            initial_state_deviation_inertial[3:6] = rotation_matrix @ initial_state_deviation_rsw[3:6]

        # set initial state perturbation for the aerocapture problem
        aerocapture_problem.set_initial_state_perturbation(initial_state_deviation_inertial)

        entry_parameters_perturbation = np.zeros(2)
        if run > 0 and (uncertainty in (uncertainties[16], uncertainties[26])):
            fpa_deviation = random.gauss(0, entry_fpa_deviation_uncertainty)  # degrees
            entry_parameters_perturbation = np.array([0., fpa_deviation])
            perturbation = fpa_deviation
        if run > 0 and (uncertainty in (uncertainties[17], uncertainties[27])):
            entry_velocity_deviation = random.gauss(0, entry_velocity_magnitude_deviation_uncertainty)
            entry_parameters_perturbation = np.array([entry_velocity_deviation, 0.])
            perturbation = entry_velocity_deviation
        aerocapture_problem.set_entry_parameters_perturbation(entry_parameters_perturbation)

        ###########################################################################
        # CREATE PROPAGATOR SETTINGS ##############################################
        ###########################################################################

        aerocapture_problem.fitness([interplanetary_arrival_velocity, flight_path_angle_at_atmosphere_entry])
        dynamics_simulator = aerocapture_problem.get_last_run_dynamics_simulator()

        ### OUTPUT OF THE SIMULATION ###
        # Retrieve propagated state and dependent variables
        state_history = dynamics_simulator.state_history
        unprocessed_state_history = dynamics_simulator.unprocessed_state_history
        dependent_variable_history = dynamics_simulator.dependent_variable_history
        heat_fluxes_history = aerocapture_problem.get_entry_heat_fluxes(return_history_dictionary=True)

        # Save results to a dictionary
        simulation_results[run] = [state_history, dependent_variable_history, heat_fluxes_history]
        if run > 0:
            perturbations[run] = perturbation

        # Get output path
        subdirectory = '/UncertaintyAnalysisData/' + uncertainty + '/'

        # Decide if output writing is required
        if write_results_to_file:
            output_path = current_dir + subdirectory
        else:
            output_path = None

        # If desired, write output to a file
        if write_results_to_file:
            save2txt(state_history, 'state_history_' + str(run) + '.dat', output_path)
            save2txt(dependent_variable_history, 'dependent_variable_history_' + str(run) + '.dat', output_path)
            save2txt(heat_fluxes_history, 'heat_fluxes_history_' + str(run) + '.dat', output_path)

    all_results[uncertainty] = simulation_results
    if write_results_to_file:
        subdirectory = '/UncertaintyAnalysisData/'
        output_path = current_dir + subdirectory
        save2txt(perturbations, 'simulation_results_' + uncertainty + '.dat',output_path)

"""
The first index of the dictionary simulation_results refers to the model case, while the second index can be 0 (states)
or 1 (dependent variables).
You can use this dictionary to make all the cross-comparison that you deem necessary. The code below currently compares
every case with respect to the "nominal" one.
"""

# uncertainties  are --> [] -- not really :/

print('\n\nAll runs have been done, now comparing the models\n')

for uncertainty_nr, uncertainty in enumerate(uncertainties_to_run):
    simulation_results = all_results[uncertainty]
    actual_arc_to_compute = arcs_to_compute_actually[uncertainty_nr]

    if actual_arc_to_compute == 1:
        selected_output_interpolation_step = 1  # s
    elif actual_arc_to_compute == 12:
        selected_output_interpolation_step = output_interpolation_step/4
    else:
        selected_output_interpolation_step = output_interpolation_step

    # Compare all the model settings with the nominal case
    for run in range(1, number_of_runs_per_uncertainty):
        print(f'\nDifference evaluation for run: {run}  -  ' + uncertainty)
        # Get output path
        output_path = current_dir + '/UncertaintyAnalysisData/' + uncertainty + '/'

        # Set time limits to avoid numerical issues at the boundaries due to the interpolation
        nominal_state_history = simulation_results[0][0]
        nominal_dependent_variable_history = simulation_results[0][1]
        nominal_heat_fluxes_history = simulation_results[0][2]
        nominal_times = list(nominal_state_history.keys())

        # Retrieve current state and dependent variable history
        current_state_history = simulation_results[run][0]
        current_dependent_variable_history = simulation_results[run][1]
        current_heat_fluxes_history = simulation_results[run][2]
        current_times = list(current_state_history.keys())

        # Get limit times at which both histories can be validly interpolated
        limit_value = 4
        interpolation_lower_limit = max(nominal_times[limit_value],current_times[limit_value])
        interpolation_upper_limit = min(nominal_times[-limit_value],current_times[-limit_value])

        # Create vector of epochs to be compared (boundaries are referred to the first case)
        if actual_arc_to_compute == 1 or actual_arc_to_compute == 12:
            unfiltered_interpolation_epochs = np.arange(current_times[0], current_times[-1], selected_output_interpolation_step)
        else:
            unfiltered_interpolation_epochs = np.geomspace(current_times[0], current_times[-1], int(abs(current_times[0]-current_times[-1])/selected_output_interpolation_step))
        # unfiltered_interpolation_epochs = np.array(current_times)
        unfiltered_interpolation_epochs = [n for n in unfiltered_interpolation_epochs if n <= interpolation_upper_limit]
        interpolation_epochs = [n for n in unfiltered_interpolation_epochs if n >= interpolation_lower_limit]

        #interpolation_epochs = unfiltered_interpolation_epochs
        # Compare state history
        state_difference_wrt_nominal = Util.compare_models(current_state_history,
                                                           nominal_state_history,
                                                           interpolation_epochs,
                                                           output_path,
                                                           'state_difference_wrt_nominal_case_' + str(run) + '.dat')
        # Compare dependent variable history
        dependent_variable_difference_wrt_nominal = Util.compare_models(current_dependent_variable_history,
                                                                        nominal_dependent_variable_history,
                                                                        interpolation_epochs,
                                                                        output_path,
                                                                        'dependent_variable_difference_wrt_nominal_case_' + str(run) + '.dat')

        # Compare heat_fluxes history
        heat_fluxes_difference_wrt_nominal = Util.compare_models(current_heat_fluxes_history,
                                                                 nominal_heat_fluxes_history,
                                                                 interpolation_epochs,
                                                                 output_path,
                                                                 'heat_fluxes_difference_wrt_nominal_case_' + str(run) + '.dat')

        # Compute the atmospheric entry interface difference here:
        # ...
