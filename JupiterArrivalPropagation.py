"""
Copyright (c) 2010-2022, Delft University of Technology
All rights reserved
This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import random

import numpy as np
import os
from scipy.stats.qmc import Sobol

# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.math import interpolators

# Problem-specific imports
import JupiterArrivalUtilities as Util
from JupiterArrivalProblem import JupiterArrivalProblem

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

mission_case = 1  # 1 or 2

# Load spice kernels
spice_interface.load_standard_kernels()

# INPUT YOUR PARAMETER SET HERE

if mission_case == 1:
    default_trajectory_parameters = [5.6,
                                     62650,
                                     50,
                                     30,
                                     0.3,
                                     0.3,
                                     2,
                                     1]
else:
    default_trajectory_parameters = [5.6,
                                     62650,
                                     50,
                                     30,
                                     0.3,
                                     0.3,
                                     0.3,  # added parameter wrt case 1
                                     2,
                                     1]


# Choose whether output of the propagation is written to files
write_results_to_file = False
# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Vehicle settings
vehicle_mass =   # kg
vehicle_reference_area =   # m^2
specific_impulse =   # s
# Fixed parameters - Mars SoI
minimum_mars_distance =   # m

# initial_propagation_time = Util.get_trajectory_initial_time(trajectory_parameters,
#                                                             time_buffer)
number_of_simulations = 1

################################
### DESIGN SPACE EXPLORATION ###
################################

use_sobol_sampling = False
use_one_seed_only = True

# List of minimum and maximum values for each design parameter (trajectory parameter)
if mission_case == 1:
    decision_variable_range = \
        [[5.1, 10958,  6, 10, 0.1, 0.1, 1, 1],
         [6.1, 11291, 80, 80,  10,  10, 4, 4]]
    # The entries of the vector 'trajectory_parameters' contains the following:
    # * Entry 0: Arrival velocity in km/s
    # * Entry 1: Initial epoch in Julian days since J2000 (MJD2000)
    # * Entry 2: Period of Target Orbit in Julian days
    # * Entry 3: Time-of-flight from the arrival at Jupiter to the flyby at the first moon in Julian days
    # * Entry 4: Time-of-flight from the flyby at the first moon to the atmospheric entry at Jupiter in Julian days
    # * Entry 5: Time-of-flight from the atmospheric entry at Jupiter to the pericenter-raise flyby in Julian days
    # * Entry 6: ordered sequences of flybys of vector [EI, GI, CI, GE, CE, CG]
    # * Entry 7: Indexes of [I, E, G, C]

else:
    decision_variable_range = \
        [[5.1, 10958, 6, 10, 0.1, 0.1, 0.1, 1, 1],
         [6.1, 11291, 80, 80, 10,  10,  10, 4, 4]]
    # The entries of the vector 'trajectory_parameters' contains the following:
    # * Entry 0: Arrival velocity in km/s
    # * Entry 1: Initial epoch in Julian days since J2000 (MJD2000)
    # * Entry 2: Period of Target Orbit in Julian days
    # * Entry 3: Time-of-flight from the arrival at Jupiter to the flyby at the first moon in Julian days
    # * Entry 4: Time-of-flight from the flyby at the first moon to the flyby at the second moon in Julian days
    # * Entry 5: Time-of-flight from the flyby at the second moon to the atmospheric entry at Jupiter in Julian days
    # * Entry 6: Time-of-flight from the atmospheric entry at Jupiter to the pericenter-raise flyby in Julian days
    # * Entry 7: ordered sequences of flybys of vector [EI, GI, CI, GE, CE, CG]
    # * Entry 8: Indexes of [I, E, G, C]


# INPUT WHAT DESIGN SPACE EXPLORATION METHOD YOU USE
design_space_method = 'simulate' # 'monte_carlo' 'factorial_design' 'monte_carlo_one_at_a_time'

number_of_parameters = len(decision_variable_range[0])

# if use_one_seed_only or use_sobol_sampling or design_space_method == 'factorial_design' or design_space_method == 'monte_carlo_one_at_a_time':
#     seeds_to_try = [42]
# else:
#     seeds_to_try = [42, 420]

if use_one_seed_only == False:
    seeds_to_try = [42, 420]
else:
    seeds_to_try = [42]


for random_seed in seeds_to_try:
    # The number of Monte Carlo simulations is defined, as well as the seed which
    # is passed to the MT19937 BitGenerator
    if design_space_method == 'monte_carlo_one_at_a_time':
        number_of_simulations_per_parameter = 50
        number_of_simulations = number_of_parameters * number_of_simulations_per_parameter
        nominal_parameters = [4335.5, 288, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # [3000.0, 500.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] [4335.5, 288, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # random_seed = 42  # ;)
        np.random.seed(random_seed)  # Slightly outdated way of doing this, but works

        # Generating Sobol distribution
        # parameters_data_sobol = np.zeros((number_of_simulations_per_parameter, number_of_parameters))
        parameters_data_sobol = np.zeros(number_of_simulations)
        for i in range(number_of_parameters):
            sobol_data = Util.sobol(decision_variable_range[0][i],
                                    decision_variable_range[1][i], number_of_simulations_per_parameter)
            for j in range(number_of_simulations_per_parameter):
                index = j + i * number_of_simulations_per_parameter
                parameters_data_sobol[index] = sobol_data[j]

        # print('\n Random Seed :', random_seed, '\n')
        changed_parameters = np.zeros(
            (number_of_simulations_per_parameter, number_of_parameters))  # only for monte-carlo one at a time

    elif design_space_method == 'monte_carlo':
        number_of_simulations = 50000
        random_seed = 42  # ;)
        np.random.seed(random_seed)  # Slightly outdated way of doing this, but works

        if use_sobol_sampling:
            parameters_data_sobol = np.zeros((number_of_simulations, number_of_parameters))
            for i in range(number_of_parameters):
                sobol_data = Util.sobol(decision_variable_range[0][i],
                                        decision_variable_range[1][i], number_of_simulations)
                random.shuffle(sobol_data)

                for j in range(number_of_simulations):
                    parameters_data_sobol[j, i] = sobol_data[j]

        print('\n Random Seed :', random_seed, '\n')

    elif design_space_method == 'factorial_design':
        # no_of_factors equals the number of parameters, all interactions are
        # included somewhere in the factorial design
        no_of_factors = number_of_parameters
        no_of_levels = 2
        # Function that creates the yates_array
        yates_array = Util.yates_array_v2(no_of_levels, no_of_factors)
        number_of_simulations = len(yates_array)

        # Evenly distributed set of values between—and including—the minimum and maximum value
        # defined earlier
        design_variable_arr = np.zeros((no_of_levels, no_of_factors))
        for par in range(no_of_factors):
            design_variable_arr[:, par] = np.linspace(decision_variable_range[0][par], decision_variable_range[1][par],
                                                      no_of_levels, endpoint=True)

    elif design_space_method == 'simulate':
        number_of_simulations = 1

    parameters = dict()

    objectives_and_constraints = dict()

    trajectory_parameters = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for simulation_index in range(number_of_simulations):
        print('\n Simulation number :', simulation_index, '\n')

        # These three methods are overall very messy and there are also some mistakes to fix
        # The factorial design runs through each row of Yates array and translates
        # the value at each index to a corresponding parameter value in
        # design_variable_arr
        if design_space_method == 'factorial_design':
            level_combination = yates_array[simulation_index, :]
            # Enumerate simplifies the code because the entries in yates_array can
            # directly be fed as indexes to the design parameters
            for it, j in enumerate(level_combination):
                j2 = int((j + 1) / 2)
                trajectory_parameters[it] = design_variable_arr[j2, it]
        elif design_space_method == 'monte_carlo':
            # For Monte Carlo and FFD, a separate loop exists
            for parameter_index in range(number_of_parameters):
                if use_sobol_sampling:
                    trajectory_parameters[parameter_index] = parameters_data_sobol[simulation_index, parameter_index]
                else:
                    trajectory_parameters[parameter_index] = np.random.uniform(
                        decision_variable_range[0][parameter_index],
                        decision_variable_range[1][parameter_index])
        elif design_space_method == 'monte_carlo_one_at_a_time':
            # If Monte Carlo, a random value is chosen with a uniform distribution (NOTE: You can change the distribution)
            parameter_labels = ['Departure Time', 'Time of Flight', 'Revolutions', 'rad1', 'rad2', 'norm1', 'norm2',
                                'ax1',
                                'ax2']
            trajectory_parameters = nominal_parameters.copy()
            current_parameter = int(simulation_index / number_of_simulations_per_parameter)
            # trajectory_parameters[current_parameter] = np.random.uniform(decision_variable_range[0][current_parameter],
            #                                                              decision_variable_range[1][current_parameter])

            trajectory_parameters[current_parameter] = parameters_data_sobol[simulation_index]

            changed_parameters[simulation_index - current_parameter * number_of_simulations_per_parameter][
                current_parameter] \
                = trajectory_parameters[current_parameter]

        elif design_space_method == 'simulate':
            trajectory_parameters = default_trajectory_parameters


        parameters[simulation_index] = trajectory_parameters.copy()

        ###########################################################################
        # CREATE ENVIRONMENT ######################################################
        ###########################################################################

        # Define settings for celestial bodies
        bodies_to_create = ['Sun',
                            'Mercury',
                            'Venus',
                            'Earth', 'Moon',
                            'Mars',
                            'Jupiter', 'Io', 'Callisto', 'Europa', 'Ganymede',
                            'Saturn',
                            'Uranus',
                            'Neptune']

        # Define coordinate system
        global_frame_origin = 'Jupiter'
        global_frame_orientation = 'ECLIPJ2000' # check if you can use a better orientation
        # Create body settings
        body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                                    global_frame_origin,
                                                                    global_frame_orientation)

        # OUTDATED: to remove
        # time_buffer = trajectory_parameters[1] / 10 * constants.JULIAN_DAY
        # if time_buffer < 15 * constants.JULIAN_DAY:
        #     time_buffer = 15 * constants.JULIAN_DAY
        time_buffer = 0

        initial_propagation_time = Util.get_trajectory_initial_time(trajectory_parameters)

        # print('For loop: ', epoch)
        # print('Distance from Earth', position_difference_from_earth)
        # CHANGE EPHEMERIDES
        # planets = ['Mercury', 'Saturn', 'Uranus', 'Neptune']
        # for planet in planets:
        #     if planet == 'Uranus':
        #         planet_bar = 'Uranus_BARYCENTER'
        #     elif planet == 'Neptune':
        #         planet_bar = 'Neptune_BARYCENTER'
        #     else:
        #         planet_bar = planet
        #     effective_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Sun') + \
        #                                         spice_interface.get_body_gravitational_parameter(planet)
        #     body_settings.get(planet).ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
        #         planet_bar, initial_propagation_time, effective_gravitational_parameter, 'Sun',
        #         global_frame_orientation)
        #
        # moons = ['Io', 'Ganymede', 'Callisto', 'Europa']
        # for moon in moons:
        #     effective_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Jupiter') + \
        #                                         spice_interface.get_body_gravitational_parameter(moon)
        #     body_settings.get(moon).ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
        #         moon, initial_propagation_time, effective_gravitational_parameter, 'Jupiter',
        #         global_frame_orientation
        #     )

        # Create bodies
        bodies = environment_setup.create_system_of_bodies(body_settings)

        # Create vehicle object and add it to the existing system of bodies
        bodies.create_empty_body('Vehicle')
        bodies.get_body('Vehicle').mass = vehicle_mass

        # Create solar radiation pressure coefficients interface
        reference_area_radiation = vehicle_reference_area
        radiation_pressure_coefficient =  # 1.2
        occulting_bodies = ['Jupiter']
        radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
            'Sun', reference_area_radiation, radiation_pressure_coefficient, occulting_bodies)
        environment_setup.add_radiation_pressure_interface(
            bodies, 'Vehicle', radiation_pressure_settings)

        # Create aerodynamic coefficients interface
        # Create aerodynamic coefficients interface (drag-only; zero side force and lift)
        reference_area_aerodynamic =   # m^2
        drag_coefficient =  # 1.2
        aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
            reference_area_aerodynamic, [drag_coefficient, 0.0, 0.0])  # I suppose it is [Drag, Side-force, Lift]
        environment_setup.add_aerodynamic_coefficient_interface(
            bodies, 'Vehicle', aero_coefficient_settings)

        ##############################################
        ### RETRIEVING DEPENDENT VARIABLES TO SAVE ###
        ##############################################

        # Retrieve dependent variables to save
        dependent_variables_to_save = Util.get_dependent_variable_save_settings()
        # Check whether there are any
        are_dependent_variables_to_save = False if not dependent_variables_to_save else True

        # Integrator Settings

        step_size = 3600  # s     # find best step size

        # Create integrator settings
        integrator_settings = Util.get_integrator_settings(initial_propagation_time, step_size)

        # Problem class is created
        current_jupiter_arrival_problem = JupiterArrivalProblem(bodies,
                                                      integrator_settings,
                                                      specific_impulse,
                                                      # minimum_mars_distance,
                                                      time_buffer,
                                                      vehicle_mass,
                                                      decision_variable_range,
                                                      initial_propagation_time,
                                                      True)
        # NOTE: Propagator settings, termination settings, and initial_propagation_time are defined in the fitness function
        fitness = current_jupiter_arrival_problem.fitness(trajectory_parameters)
        objectives_and_constraints[simulation_index] = fitness

        ### OUTPUT OF THE SIMULATION ###
        # Retrieve propagated state and dependent variables
        state_history = current_jupiter_arrival_problem.get_last_run_dynamics_simulator().state_history
        dependent_variable_history = current_jupiter_arrival_problem.get_last_run_dynamics_simulator().dependent_variable_history

        # Get output path
        if use_sobol_sampling or design_space_method == 'factorial_design' or design_space_method == 'monte_carlo_one_at_a_time':
            subdirectory = '/DesignSpace_%s/Run_%s' % (design_space_method, simulation_index)
        else:
            subdirectory = '/DesignSpace_%s_seed_%s/Run_%s' % (design_space_method, str(random_seed), simulation_index)

        # Decide if output writing is required
        if write_results_to_file:
            output_path = current_dir + subdirectory
        else:
            output_path = None

        # If desired, write output to a file
        if write_results_to_file:
            save2txt(state_history, 'state_history.dat', output_path)
            save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)

    if write_results_to_file:
        if use_sobol_sampling or design_space_method == 'factorial_design' or design_space_method == 'monte_carlo_one_at_a_time':
            subdirectory = '/DesignSpace_%s' % design_space_method
        else:
            subdirectory = '/DesignSpace_%s_seed_%s' % (design_space_method, str(random_seed))
        output_path = current_dir + subdirectory
        save2txt(parameters, 'parameter_values.dat', output_path)
        save2txt(objectives_and_constraints, 'objectives_constraints.dat', output_path)