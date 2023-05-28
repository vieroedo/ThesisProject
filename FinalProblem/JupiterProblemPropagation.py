# import os
# import shutil
#
# import numpy as np
# import matplotlib.pyplot as plt
# from numpy import linalg as LA
# from time import process_time as pt

from tudatpy.io import save2txt

# Problem-specific imports
# from JupiterTrajectory_GlobalParameters import *
# import CapsuleEntryUtilities as Util
import class_AerocaptureNumericalProblem as ae_model
import optimization_functions
from handle_functions import *

do_refinement_of_solutions = True

write_results_to_file = True
choose_model = 0  # zero is the default model, the final one. 1 is the most raw one, higher numbers are better ones.
integrator_settings_index = -4
simulation_start_epoch = first_january_2040_epoch

grid_search_v_inf_samples = 100
grid_search_entry_fpa_samples = 100


current_dir = os.path.dirname(__file__)

# Load spice kernels
spice_interface.load_standard_kernels()


# decision_variable_names = ['InterplanetaryVelocity', 'EntryFpa']
# decision_variable_to_evaluate = ['InterplanetaryVelocity', 'EntryFpa']

if do_refinement_of_solutions:
    decision_variable_range = [[5800., -3.1],
                               [6100., -2.9]]

    interplanetary_velocity_boundaries = [5800., 6100.]
    entry_fpa_boundaries_deg = [-3.1, -2.9]
else:
    decision_variable_range = [[5100., -5],
                               [6100., -1.5]]

    interplanetary_velocity_boundaries = [5100., 6100.]
    entry_fpa_boundaries_deg = [-5, -1.5]

aerocapture_problem = ae_model.AerocaptureNumericalProblem(decision_variable_range,
                                                           choose_model,
                                                           integrator_settings_index)
are_dependent_variables_to_save = True

interplanetary_velocity_range = np.linspace(interplanetary_velocity_boundaries[0], interplanetary_velocity_boundaries[1], grid_search_v_inf_samples)
entry_fpa_range_deg = np.linspace(entry_fpa_boundaries_deg[0], entry_fpa_boundaries_deg[1], grid_search_entry_fpa_samples)


decision_variable_list = ['InterplanetaryVelocity', 'EntryFpa']
objectives_list = ['payload_mass_fraction', 'total_radiation_dose', 'benefit_over_insertion_burn']
constraints_list = ['maximum_aerodynamic_load', 'peak_heat_flux', 'minimum_jupiter_distance', 'maximum_jupiter_distance', 'final_eccentricity']

total_columns = decision_variable_list + objectives_list + constraints_list

grid_search_rows = grid_search_v_inf_samples * grid_search_entry_fpa_samples
grid_search_columns = len(total_columns)
grid_search_values = np.zeros((grid_search_rows, grid_search_columns))
for current_interplanetary_velocity_no, current_interplanetary_velocity in enumerate(interplanetary_velocity_range):

    for current_entry_fpa_no, current_entry_fpa_deg in enumerate(entry_fpa_range_deg):
        # Set current row number
        current_row_number = current_entry_fpa_no + grid_search_entry_fpa_samples * current_interplanetary_velocity_no

        if do_refinement_of_solutions:
            print('REFINED SOLUTIONS')

        print(f'Simulation No: {current_row_number}     V_inf = {current_interplanetary_velocity/1e3} km/s      Entry fpa: {current_entry_fpa_deg} deg')

        fitness_values = aerocapture_problem.fitness([current_interplanetary_velocity,
                                                      current_entry_fpa_deg,
                                                      simulation_start_epoch])

        decision_variable_values = np.array([current_interplanetary_velocity, current_entry_fpa_deg])
        objective_values = fitness_values[0]
        constraint_values = fitness_values[1]

        # payload_mass_fraction, total_radiation_dose, benefit_over_insertion_burn = objective_values[0], objective_values[1], objective_values[2]
        # have_constraints_been_violated, constraint_number = optimization_functions.have_constraints_been_violated(constraint_values)

        grid_search_values[current_row_number,:] = np.concatenate((decision_variable_values, objective_values, constraint_values))



        # dynamics_simulator = aerocapture_problem.get_last_run_dynamics_simulator()
        #
        # # Retrieve propagated state and dependent variables
        # state_history = dynamics_simulator.state_history
        # unprocessed_state_history = dynamics_simulator.unprocessed_state_history
        # dependent_variable_history = dynamics_simulator.dependent_variable_history
        #
        # if write_results_to_file:
        #     save2txt(state_history, 'simulation_state_history.dat', current_dir + '/SimulationOutput')
        #     save2txt(dependent_variable_history, 'simulation_dependent_variable_history.dat', current_dir + '/SimulationOutput')

if write_results_to_file:
    savedir = current_dir + '/GridSearch/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    filename = 'grid_search_results.dat'
    if do_refinement_of_solutions:
        filename = 'grid_search_results_refined.dat'
    np.savetxt(savedir + filename, grid_search_values)
