'''
Copyright (c) 2010-2021, Delft University of Technology
All rights reserved
This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

This module defines useful functions that will be called by the main script, where the optimization is executed.
'''

# Problem-specific imports
import numpy as np
import CapsuleEntryUtilities as Util

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory

import numpy.linalg as LA

###########################################################################
# CREATE PROBLEM CLASS ####################################################
###########################################################################

class JupiterArrivalProblem:
    """
    Class to initialize, simulate and optimize the Jupiter aerocapture trajectory.
    The class is created specifically for this problem. This is done to provide better integration with Pagmo/Pygmo,
    where the optimization process (assignment 3) will be done.

    Attributes
    ----------
    bodies
    integrator_settings
    propagator_settings
    specific_impulse
    # minimum_mars_distance
    # time_buffer
    perform_propagation

    Methods
    -------
    get_last_run_propagated_state_history()
    get_last_run_dependent_variable_history()
    get_last_run_dynamics_simulator()
    fitness(trajectory_parameters)
    get_hodographic_shaping()
    """

    def __init__(self,
                 bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                 integrator_settings: tudatpy.kernel.numerical_simulation.propagation_setup.integrator.IntegratorSettings,
                 specific_impulse: float,
                 # minimum_mars_distance: float,
                 # time_buffer: float,
                 vehicle_mass: float,
                 decision_variable_range,
                 epoch: float,
                 perform_propagation: bool = True):
        """
        Constructor for the LowThrustProblem class.
        Parameters
        ----------
        bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
            System of bodies present in the simulation.
        integrator_settings : tudatpy.kernel.numerical_simulation.propagation_setup.integrator.IntegratorSettings
            Integrator settings to be provided to the dynamics simulator.
        specific_impulse : float
            Constant specific impulse of the vehicle.
        minimum_mars_distance : float
            Minimum distance from Mars at which the propagation stops.
        time_buffer : float
            Time interval between the simulation start epoch and the beginning of the hodographic trajectory.
        perform_propagation : bool (default: True)
            If true, the propagation is performed.
        Returns
        -------
        none
        """
        # Copy arguments as attributes
        self.bodies_function = lambda : bodies
        self.integrator_settings_function = lambda : integrator_settings
        self.specific_impulse = specific_impulse
        # self.minimum_mars_distance = minimum_mars_distance
        # self.time_buffer = time_buffer
        self.vehicle_mass = vehicle_mass
        self.decision_variable_range = decision_variable_range
        self.perform_propagation = perform_propagation
        self.epoch = epoch

    def get_bounds(self):

        return self.decision_variable_range

    def get_last_run_propagated_cartesian_state_history(self) -> dict:
        """
        Returns the full history of the propagated state, converted to Cartesian states
        Parameters
        ----------
        none
        Returns
        -------
        dict
        """
        return self.dynamics_simulator_function( ).get_equations_of_motion_numerical_solution()

    def get_last_run_propagated_state_history(self) -> dict:
        """
        Returns the full history of the propagated state, not converted to Cartesian state
        (i.e. in the actual formulation that was used during the numerical integration).
        Parameters
        ----------
        none
        Returns
        -------
        dict
        """
        return self.dynamics_simulator_function( ).get_equations_of_motion_numerical_solution_raw()

    def get_last_run_dependent_variable_history(self) -> dict:
        """
        Returns the full history of the dependent variables.
        Parameters
        ----------
        none
        Returns
        -------
        dict
        """
        return self.dynamics_simulator_function( ).get_dependent_variable_history()

    def get_last_run_dynamics_simulator(self) -> tudatpy.kernel.numerical_simulation.SingleArcSimulator:
        """
        Returns the dynamics simulator object.
        Parameters
        ----------
        none
        Returns
        -------
        tudatpy.kernel.numerical_simulation.SingleArcSimulator
        """
        return self.dynamics_simulator_function( )

    def fitness(self,
                trajectory_parameters) -> float:
        """
        Propagate the trajectory with the parameters given as argument.
        This function uses the trajectory parameters to numerically propagate the trajectory.
        The fitness, currently set to zero, can be computed here: it will be used during the optimization process.
        Parameters
        ----------
        trajectory_parameters : list of floats
            List of trajectory parameters to optimize.
        Returns
        -------
        fitness : list
            # min_earth_distance,
            # max_thrust,
            # max_angle_rate,
            # max_thrust_acceleration,
            # maximum_distance_sun,
            # minimum_distance_sun,
            # max_perturbations_acceleration,
            # low_thrust_delta_v,
            # earth_escape_delta_v,
            # mars_insertion_delta_v,
            # sc_propellant_consumed,
            # earth_departure_fuel_consumed,
            # time_of_flight
        """

        # Create hodographic shaping object
        bodies = self.bodies_function()
        # hodographic_shaping = Util.create_hodographic_shaping_object(trajectory_parameters,
        #                                                              bodies)
        # self.hodographic_shaping_function = lambda : hodographic_shaping

        # Propagate trajectory only if required
        if self.perform_propagation:

            integrator_settings = self.integrator_settings_function( )

            termination_settings = Util.get_termination_settings(self.epoch)
            initial_propagation_time = Util.get_trajectory_initial_time(trajectory_parameters,
                                                                        self.time_buffer)
            # initial_propagation_time = self.epoch
            dependent_variables_to_save = Util.get_dependent_variable_save_settings(bodies)
            propagator_settings = Util.get_propagator_settings(trajectory_parameters, bodies, initial_propagation_time,
                                                               self.specific_impulse,
                                                               current_propagator=self.vehicle_mass,
                                                               galileo_propagator_settings=dependent_variables_to_save)


            # Create simulation object and propagate dynamics
            dynamics_simulator = numerical_simulation.SingleArcSimulator(bodies,
                                                                         integrator_settings,
                                                                         propagator_settings,
                                                                         print_dependent_variable_data = False)

            self.dynamics_simulator_function = lambda: dynamics_simulator

        # Add the objective and constraint values into the fitness vector

        # Dictionaries for computing objectives and constraints
        state_history = dynamics_simulator.state_history
        dependent_variables_history = dynamics_simulator.dependent_variable_history
        epochs_dep_var = list(dependent_variables_history.keys())

        #### Minimum Earth distance constraint ###################################################
        buffer_time = self.time_buffer / constants.JULIAN_DAY
        days = np.array(range(int(trajectory_parameters[1] - buffer_time)))
        days = days + buffer_time
        seconds_since_departure = list(days * constants.JULIAN_DAY)
        vehicle_hodographic_states = hodographic_shaping.get_trajectory(times=seconds_since_departure)
        # print(vehicle_hodographic_states)
        vehicle_cartesian_states = np.vstack(list(vehicle_hodographic_states.values()))
        epochs_hodograph = trajectory_parameters[0] * constants.JULIAN_DAY + np.vstack(seconds_since_departure)
        min_earth_distance = np.inf
        for i, epoch in enumerate(epochs_hodograph):
            earth_state = spice_interface.get_body_cartesian_state_at_epoch(
                target_body_name="Earth",
                observer_body_name='Sun',
                reference_frame_name='ECLIPJ2000',
                aberration_corrections="NONE",
                ephemeris_time=epoch
            )
            current_distance = (vehicle_cartesian_states[i,:] - earth_state)[0:3]
            current_distance_norm = LA.norm(current_distance)
            if current_distance_norm < min_earth_distance:
                min_earth_distance = current_distance_norm
        ##########################################################################################


        ### Max Thrust and Max Thrust angular rates Constraint ###################################
        # Retrieve acceleration vector
        thrust_profile = np.vstack(list(dependent_variables_history.values()))[:, 3:6]
        states = np.vstack(list(state_history.values()))
        thrust_acceleration = LA.norm(thrust_profile, axis=1)
        thrust_magnitude = states[:,-1] * thrust_acceleration
        # print('thurst',thrust_magnitude)
        max_thrust_acceleration = np.amax(thrust_acceleration)

        # Calculate max thrust used in the trajectory
        max_thrust = np.amax(thrust_magnitude)


        # Calculate max angle rates between thrusts
        max_angle_rate = 0
        for num, epoch in enumerate(epochs_dep_var):
            vector_1 = thrust_profile[num, :]
            vector_2 = thrust_profile[min(num+1, len(thrust_profile)-1), :]
            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
            dot_product = np.dot(unit_vector_1, unit_vector_2)

            if dot_product>=1:
                dot_product = 1.0
            angle = np.arccos(dot_product)
            if num < (len(epochs_dep_var)-1):
                angle_rate = abs(angle / (epochs_dep_var[num+1] - epochs_dep_var[num]))
            else:
                angle_rate = 0
            if angle_rate > max_angle_rate:
                max_angle_rate = angle_rate
        ##########################################################################################


        ### Minimum and Maximum distance to Sun ##################################################
        distance_to_sun = np.vstack(list(dependent_variables_history.values()))[:, 1]

        maximum_distance_sun = np.amax(distance_to_sun)
        minimum_distance_sun = np.amin(distance_to_sun)
        ##########################################################################################


        ### Max perturbations ####################################################################
        total_acceleration = np.vstack(list(dependent_variables_history.values()))[:, 6:9]
        sun_acceleration = np.vstack(list(dependent_variables_history.values()))[:, 9:12]

        perturbations_acceleration = total_acceleration - sun_acceleration - thrust_profile
        perturbations_acceleration_norm = LA.norm(perturbations_acceleration, axis=1)
        max_perturbations_acceleration = np.nanmax(perturbations_acceleration_norm)
        ##########################################################################################

        ### Low-Thrust delta v objective #########################################################
        low_thrust_delta_v = hodographic_shaping.compute_delta_v()
        ##########################################################################################

        ### High-Thrust delta v and fuel consumed objectives #####################################
        earth_escape_delta_v = Util.delta_v_earth_escape(state_history)
        mars_insertion_delta_v = Util.delta_v_mars_insertion(state_history)
        total_propellant_consumed = Util.total_fuel_consumed(state_history)
        sc_propellant_consumed = Util.sc_fuel_consumed(state_history)
        earth_departure_fuel_consumed = Util.earth_departure_fuel_consumed(state_history)
        ##########################################################################################

        ### Time of Flight #######################################################################
        time_of_flight = trajectory_parameters[1]
        # print('1',low_thrust_delta_v,'2', min_earth_distance, '3',max_thrust, '4',max_angle_rate, '5',maximum_distance_sun,'6', minimum_distance_sun,'7', max_perturbations_acceleration,'8',low_thrust_delta_v,'9', earth_escape_delta_v,'10', mars_insertion_delta_v,'11',total_propellant_consumed,'12',time_of_flight)

        return [min_earth_distance,
                max_thrust, max_angle_rate, max_thrust_acceleration,
                maximum_distance_sun, minimum_distance_sun,
                max_perturbations_acceleration,
                low_thrust_delta_v, earth_escape_delta_v, mars_insertion_delta_v,sc_propellant_consumed,
                earth_departure_fuel_consumed,
                time_of_flight,]