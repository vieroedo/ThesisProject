'''
Copyright (c) 2010-2021, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

AE4866 Propagation and Optimization in Astrodynamics
Shape Optimization
First name: ***COMPLETE HERE***
Last name: ***COMPLETE HERE***
Student number: ***COMPLETE HERE***

This module defines useful functions that will be called by the main script, where the optimization is executed.
'''

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np

from handle_functions import *


# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.math import root_finders
from tudatpy.kernel.math import geometry

###########################################################################
# PROPAGATION SETTING UTILITIES ###########################################
###########################################################################


def get_initial_state_old() -> np.ndarray:
    """
    A sample initial state just for trying out things.
    """
    initial_state_vector = np.array([-46846525005.13016, 11342093939.996948, 0.0, 5421.4069183825695, -1402.9779133377706, 0.0])
    return initial_state_vector


def get_galileo_initial_state() -> np.ndarray:
    """
    Retrieves the Galileo probe atmospheric entry initial state.

    The initial state is expressed in global (inertial) cartesian coordinates.

    Returns
    -------
    initial_state_inertial_coordinates : np.ndarray
        The initial state of the vehicle expressed in inertial coordinates.
    """

    # Galileo atmospheric entry data (obtained from https://adsabs.harvard.edu/pdf/2006ESASP.631E...6R)

    entry_velocity = galileo_entry_velocity  # m/s
    entry_fpa = galileo_entry_fpa  # rad  # galileo entry fpa wrt atmosphere: -8.4104 deg   -6.69
    latitude = galileo_entry_latitude  # rad
    longitude = galileo_entry_longitude  # rad
    heading = galileo_entry_heading  # rad
    altitude = galileo_entry_altitude  # m
    radius = jupiter_radius + altitude

    # Calculate Galileo probe positon
    x_pos, y_pos, z_pos = cartesian_3d_from_polar(radius, latitude, longitude)
    position = np.array([x_pos, y_pos, z_pos])

    # Calculate velocity vector tangent to surface and eastward-pointing
    if z_pos > 0.:
        vel_eastward_pointing = unit_vector(np.cross(position, np.array([x_pos, y_pos, 0.])))
    elif z_pos < 0:
        vel_eastward_pointing = unit_vector(np.cross(np.array([x_pos, y_pos, 0.]), position))
    else:
        vel_eastward_pointing = unit_vector(np.cross(z_axis, position))

    # Rotate velocity vector by heading angle
    rot_matrix_heading = rotation_matrix(unit_vector(position), heading)
    vel_heading = rotate_vectors_by_given_matrix(rot_matrix_heading, vel_eastward_pointing)

    # Rotate velocity vector by flight path angle
    rot_axis_fpa = unit_vector(np.cross(position, vel_heading))
    rot_matrix_fpa = rotation_matrix(rot_axis_fpa, -entry_fpa)
    vel_final = rotate_vectors_by_given_matrix(rot_matrix_fpa, vel_heading)

    # Calculate initial velocity
    velocity = vel_final * entry_velocity

    # Build initial state vector
    initial_state_vector = np.concatenate((position, velocity))

    return initial_state_vector


def get_initial_state(atmosphere_entry_fpa: float,
                      atmosphere_entry_altitude: float = 400e3,
                      jupiter_arrival_v_inf: float = 5600,
                      verbose: bool = False) -> np.ndarray:
    """
    Calculates the initial state given the initial distance and speed, and the arrival fpa.

    The initial state is expressed in Jupiter-centered cartesian coordinates.

    Parameters
    ----------
    atmosphere_entry_fpa : float
        Flight path angle of the spacecraft at the atmosphere interface.
    atmosphere_entry_altitude : float
        Altitude of the spacecraft at the atmosphere interface.
    jupiter_arrival_v_inf : float
        Interplanetary excees velocity in Jupiter frame.
    verbose: bool
        Choose whether to plot analytical arc info or not.

    Returns
    -------
    initial_state_vector : np.ndarray
        The initial state of the vehicle expressed in inertial coordinates.
    """

    # Problem parameters
    arrival_fpa = np.deg2rad(atmosphere_entry_fpa)  # rad
    departure_radius = jupiter_SOI_radius
    departure_velocity_norm = jupiter_arrival_v_inf

    # Calculate orbital energy
    initial_orbital_energy = orbital_energy(departure_radius, departure_velocity_norm)

    # Calculate arrival radius and speed
    arrival_radius = jupiter_radius + atmosphere_entry_altitude
    arrival_velocity_norm = velocity_from_energy(initial_orbital_energy, arrival_radius)

    # Calculate angular momentum
    angular_momentum_norm = arrival_radius * arrival_velocity_norm * np.cos(arrival_fpa)
    # angular_momentum = z_axis * angular_momentum_norm

    # Calculate other orbit elements
    semilatus_rectum = angular_momentum_norm ** 2 / jupiter_gravitational_parameter
    semimajor_axis = - jupiter_gravitational_parameter / (2 * initial_orbital_energy)
    eccentricity = np.sqrt(1 - semilatus_rectum / semimajor_axis)

    # Set arrival position on x axis
    arrival_position = x_axis * arrival_radius

    circ_vel_at_atm_entry = np.sqrt(
        jupiter_gravitational_parameter / (jupiter_radius + atmosphere_entry_altitude))

    # Prints for debugging
    if verbose:
        print('\nAtmospheric entry (pre-aerocapture) analytical conditions:\n'
              f'- altitude: {atmosphere_entry_altitude / 1e3} km\n'
              f'- velocity: {arrival_velocity_norm / 1e3:.3f} km/s\n'
              f'- ref circular velocity: {circ_vel_at_atm_entry / 1e3:.3f} km/s\n'
              f'- flight path angle: {atmosphere_entry_fpa} deg\n'
              f'- eccentricity: {eccentricity:.10f} ')

    # Calculate delta true anomaly spanned by the spacecraft
    departure_true_anomaly = true_anomaly_from_radius(departure_radius, eccentricity, semimajor_axis)
    arrival_true_anomaly = true_anomaly_from_radius(arrival_radius, eccentricity,semimajor_axis)
    delta_true_anomaly = arrival_true_anomaly - departure_true_anomaly

    # Calculate the departure position of the spacecraft
    pos_rotation_matrix = rotation_matrix(z_axis, -delta_true_anomaly)
    departure_position = rotate_vectors_by_given_matrix(pos_rotation_matrix, unit_vector(
        arrival_position)) * departure_radius

    # Calculate departure fpa (useful to obtain departure velocity)
    departure_fpa = - np.arccos(
        angular_momentum_norm / (departure_radius * departure_velocity_norm))

    # Calculate departure velocity
    vel_rotation_matrix = rotation_matrix(z_axis, np.pi / 2 - departure_fpa)
    departure_velocity = rotate_vectors_by_given_matrix(vel_rotation_matrix, unit_vector(
        departure_position)) * departure_velocity_norm

    # Build the initial state vector
    initial_state_vector = np.concatenate((departure_position, departure_velocity))

    # Print the state vector for debugging
    if verbose:
        print('\nDeparture state:')
        print(f'{list(initial_state_vector)}')

    return initial_state_vector


def get_termination_settings(simulation_start_epoch: float,
                             maximum_duration: float = 200*constants.JULIAN_DAY,
                             galileo_termination_settings: bool = False) \
        -> tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings:
    """
    Get the termination settings for the simulation.

    Termination settings currently include:
    - Nominal termination settings, to be satisfied all at once:
        . Apoapsis greater than 0 termination
        . Negative fpa termination
        . No aerodynamic force termination
    - Time termination (safety setting)
    - Maximum distance termination (if spacecraft doesn't get captured)

    Parameters
    ----------
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    maximum_duration : float
        Maximum duration of the simulation [s].
    galileo_termination_settings: bool
        Switches to the termination settings for the Galileo Mission.

    Returns
    -------
    hybrid_termination_settings : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
        Propagation termination settings object.
    """

    # Return terminatino altitude for the Galileo mission
    if galileo_termination_settings:
        termination_settings = propagation_setup.propagator.dependent_variable_termination(
            dependent_variable_settings=propagation_setup.dependent_variable.altitude('Capsule', 'Jupiter'),
            limit_value=20e3,
            use_as_lower_limit=True,
            terminate_exactly_on_final_condition=False
        )
        return termination_settings

    # Create single PropagationTerminationSettings objects
    # Time
    time_termination_settings = propagation_setup.propagator.time_termination(
        simulation_start_epoch + maximum_duration,
        terminate_exactly_on_final_condition=False
    )

    # Maximum altitude
    maximum_altitude_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.altitude('Capsule', 'Jupiter'),
        limit_value=jupiter_SOI_radius,
        use_as_lower_limit=False,
        terminate_exactly_on_final_condition=False
    )

    # Minimum apoapsis (if greater than zero te orbit is closed)
    minimum_apoapsis_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.apoapsis_altitude('Capsule', 'Jupiter'),
        limit_value=0.,
        use_as_lower_limit=False,
        terminate_exactly_on_final_condition=False
    )

    # A negative fpa would trigger termination
    fpa_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.flight_path_angle('Capsule', 'Jupiter'),
        limit_value=0.,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False
    )

    # No aerodynamic force would trigger termination
    aero_force_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.aerodynamic_type,'Capsule', 'Jupiter'),
        limit_value=1e-4,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False
    )

    # List of settings that trigger termination if satisfied all at once
    nominal_settings_list = [minimum_apoapsis_termination_settings,
                             fpa_termination_settings,
                             aero_force_termination_settings]

    # Create termination settings object for the nominal conditions termination
    nominal_termination_settings = propagation_setup.propagator.hybrid_termination(nominal_settings_list,
                                                                                   fulfill_single_condition=False)

    # List of all settings that terminate the propagation
    termination_list = [nominal_termination_settings,
                        maximum_altitude_termination_settings,
                        time_termination_settings]

    # Create termination settings object that terminates when either nominal or non-nominal conditions are reached
    termination_settings = propagation_setup.propagator.hybrid_termination(termination_list,
                                                                           fulfill_single_condition=True)

    return termination_settings


def get_dependent_variable_save_settings() -> list:
    """
    Retrieves the dependent variables to save.

    Currently, the dependent variables saved include:
    - Aerodynamic acceleration (3-vector)
    - Point-mass acceleration norm
    - Altitude wrt Jupiter 1-bar level
    - Flight path angle
    - Airspeed
    - Mach number
    - Density of Jupiter's atmosphere

    Returns
    -------
    dependent_variables_to_save : list[tudatpy.kernel.numerical_simulation.propagation_setup.dependent_variable]
        List of dependent variables to save.
    """
    dependent_variables_to_save = [propagation_setup.dependent_variable.single_acceleration(propagation_setup.acceleration.aerodynamic_type,'Capsule', 'Jupiter'),
                                   propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.point_mass_gravity_type, 'Capsule', 'Jupiter'),
                                   propagation_setup.dependent_variable.altitude('Capsule', 'Jupiter'),
                                   propagation_setup.dependent_variable.flight_path_angle('Capsule', 'Jupiter'),
                                   propagation_setup.dependent_variable.airspeed('Capsule', 'Jupiter'),
                                   propagation_setup.dependent_variable.mach_number('Capsule', 'Jupiter'),
                                   propagation_setup.dependent_variable.density('Capsule', 'Jupiter')
                                   ]
    return dependent_variables_to_save


def get_integrator_settings(settings_index: int,
                            simulation_start_epoch: float,
                            galileo_integration_settings: bool = False,
                            galileo_step_size: float = 0.5) \
        -> tudatpy.kernel.numerical_simulation.propagation_setup.integrator.IntegratorSettings:
    """

    Retrieves the integrator settings. An RKF7(8) integration scheme is used.

    Parameters
    ----------
    settings_index : int
        Index that selects the tolerance of the integrator.
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    galileo_integration_settings: bool
        Loads the integration settings for the Galileo mission. An RK7 with fixed step size is used.
    galileo_step_size: float
        Sets the step size of the integration for the Galileo mission.

    Returns
    -------
    integrator_settings : tudatpy.kernel.numerical_simulation.propagation_setup.integrator.IntegratorSettings
        Integrator settings to be provided to the dynamics simulator.
    """

    # Select variable-step integrator
    current_coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_78
    # Compute current tolerance
    current_tolerance = 10.0 ** (-10.0 + settings_index)
    # Create integrator settings
    integrator = propagation_setup.integrator
    integrator_settings = integrator.runge_kutta_variable_step_size(
        simulation_start_epoch,
        40000.0,
        current_coefficient_set,
        np.finfo(float).eps,
        5*constants.JULIAN_DAY,
        current_tolerance,
        current_tolerance)

    # Return integration settings for the galileo probe entry case
    if galileo_integration_settings:
        return integrator.runge_kutta_variable_step_size(
                simulation_start_epoch,
                galileo_step_size,
                current_coefficient_set,
                galileo_step_size,
                galileo_step_size,
                np.inf,
                np.inf)

    return integrator_settings


def get_propagator_settings(atm_entry_fpa: float,
                            atm_entry_alt: float,
                            bodies,
                            termination_settings,
                            dependent_variables_to_save,
                            current_propagator=propagation_setup.propagator.unified_state_model_quaternions,
                            jupiter_interpl_excees_vel: float = 5600.,
                            initial_state_perturbation=np.zeros(6),
                            galileo_propagator_settings: bool = False,
                            model_choice: int = 0):

    # Define bodies that are propagated and their central bodies of propagation
    bodies_to_propagate = ['Capsule']
    central_bodies = ['Jupiter']

    # Define accelerations for the nominal case and for other variations of the model
    if model_choice == 0:
        acceleration_settings_on_vehicle = {'Jupiter': [propagation_setup.acceleration.point_mass_gravity(),
                                                        propagation_setup.acceleration.aerodynamic()]}
    elif model_choice == 1:
        acceleration_settings_on_vehicle = {'Jupiter': [propagation_setup.acceleration.spherical_harmonic_gravity(2, 0),
                                                        propagation_setup.acceleration.aerodynamic()]}
    elif model_choice == 2 or model_choice == 3:
        acceleration_settings_on_vehicle = {'Jupiter': [propagation_setup.acceleration.spherical_harmonic_gravity(8, 0),
                                                        propagation_setup.acceleration.aerodynamic()]}
    else:
        return -1

    # Create global accelerations' dictionary
    acceleration_settings = {'Capsule': acceleration_settings_on_vehicle}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    # Set vehicle body orientation (constant angle of attack, zero sideslip and bank angle)
    # environment_setup.set_constant_aerodynamic_orientation(
    #     bodies.get_body('Capsule'),shape_parameters[5], 0.0, 0.0,
    #     silence_warnings=True )

    # def aero_angles(t):
    #     return np.array([10*np.pi/180, 0, 0])

    # rotation_model_settings = environment_setup.rotation_model.aerodynamic_angle_based(
    #     'Jupiter', '', 'Capsule_Fixed', aero_angles)
    # environment_setup.add_rotation_model(bodies, 'Capsule', rotation_model_settings)

    # Retrieve initial state
    initial_state = get_initial_state(atm_entry_fpa, atm_entry_alt, jupiter_interpl_excees_vel, verbose=True) + initial_state_perturbation

    # Retrieve galileo probe entry initial state instead if needed
    if galileo_propagator_settings:
        initial_state = get_galileo_initial_state()

    # Create propagation settings for the benchmark
    propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                     acceleration_models,
                                                                     bodies_to_propagate,
                                                                     initial_state,
                                                                     termination_settings,
                                                                     current_propagator,
                                                                     output_variables=dependent_variables_to_save)
    return propagator_settings


###########################################################################
# CAPSULE SHAPE/AERODYNAMICS UTILITIES ####################################
###########################################################################


def get_capsule_coefficient_interface(capsule_shape: tudatpy.kernel.math.geometry.Capsule) \
        -> tudatpy.kernel.numerical_simulation.environment.HypersonicLocalInclinationAnalysis:
    """
    Function that creates an aerodynamic database for a capsule, based on a set of shape parameters.

    The Capsule shape consists of four separate geometrical components: a sphere segment for the nose, a torus segment
    for the shoulder/edge, a conical frustum for the rear body, and a sphere segment for the rear cap (see Dirkx and
    Mooij, 2016). The code used in this function discretizes these surfaces into a structured mesh of quadrilateral
    panels. The parameters number_of_points and number_of_lines define the number of discretization points (for each
    part) in both independent directions (lengthwise and circumferential). The list selectedMethods defines the type of
    aerodynamic analysis method that is used.

    Parameters
    ----------
    capsule_shape : tudatpy.kernel.math.geometry.Capsule
        Object that defines the shape of the vehicle.

    Returns
    -------
    hypersonic_local_inclination_analysis : tudatpy.kernel.environment.HypersonicLocalInclinationAnalysis
        Database created through the local inclination analysis method.
    """

    # Define settings for surface discretization of the capsule
    number_of_lines = [31, 31, 31, 11]
    number_of_points = [31, 31, 31, 11]
    # Set side of the vehicle (DO NOT CHANGE THESE: setting to true will turn parts of the vehicle 'inside out')
    invert_order = [0, 0, 0, 0]

    # Define moment reference point. NOTE: This value is chosen somewhat arbitrarily, and will only impact the
    # results when you consider any aspects of moment coefficients
    moment_reference = np.array([-0.6624, 0.0, 0.1369])

    # Define independent variable values
    independent_variable_data_points = []
    # Mach
    mach_points = environment.get_default_local_inclination_mach_points()
    independent_variable_data_points.append(mach_points)
    # Angle of attack
    angle_of_attack_points = np.linspace(np.deg2rad(-40),np.deg2rad(40),17)
    independent_variable_data_points.append(angle_of_attack_points)
    # Angle of sideslip
    angle_of_sideslip_points = environment.get_default_local_inclination_sideslip_angle_points()
    independent_variable_data_points.append(angle_of_sideslip_points)

    # Define local inclination method to use (index 0=Newtonian flow)
    selected_methods = [[0, 0, 0, 0], [0, 0, 0, 0]]

    # Get the capsule middle radius
    capsule_middle_radius = capsule_shape.middle_radius
    # Calculate reference area
    reference_area = np.pi * capsule_middle_radius ** 2

    # Create aerodynamic database
    hypersonic_local_inclination_analysis = environment.HypersonicLocalInclinationAnalysis(
        independent_variable_data_points,
        capsule_shape,
        number_of_lines,
        number_of_points,
        invert_order,
        selected_methods,
        reference_area,
        capsule_middle_radius,
        moment_reference)
    return hypersonic_local_inclination_analysis


def set_capsule_shape_parameters(shape_parameters: list,
                                 bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                                 capsule_density: float):
    """
    It computes and creates the properties of the capsule (shape, mass, aerodynamic coefficient interface...).

    Parameters
    ----------
    shape_parameters : list of floats
        List of shape parameters to be optimized.
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    capsule_density : float
        Constant density of the vehicle.

    Returns
    -------
    none
    """
    # Compute shape constraint
    length_limit = shape_parameters[1] - shape_parameters[4] * (1 - np.cos(shape_parameters[3]))
    length_limit /= np.tan(- shape_parameters[3])
    # Add safety factor
    length_limit -= 0.01
    # Apply constraint
    if shape_parameters[2] >= length_limit:
        shape_parameters[2] = length_limit

    # Create capsule
    new_capsule = geometry.Capsule(*shape_parameters[0:5])
    # Compute new body mass
    new_capsule_mass = capsule_density * new_capsule.volume
    # Set capsule mass
    bodies.get_body('Capsule').set_constant_mass(new_capsule_mass)
    # Create aerodynamic interface from shape parameters (this calls the local inclination analysis)
    new_aerodynamic_coefficient_interface = get_capsule_coefficient_interface(new_capsule)
    # Update the Capsule's aerodynamic coefficient interface
    bodies.get_body('Capsule').aerodynamic_coefficient_interface = new_aerodynamic_coefficient_interface


# NOTE TO STUDENTS: if and when making modifications to the capsule shape, do include them in this function and not in
# the main code.
def add_capsule_to_body_system(bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                               shape_parameters: list,
                               capsule_density: float):
    """
    It creates the capsule body object and adds it to the body system, setting its shape based on the shape parameters
    provided.

    Parameters
    ----------
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    shape_parameters : list of floats
        List of shape parameters to be optimized.
    capsule_density : float
        Constant density of the vehicle.

    Returns
    -------
    none
    """
    # Create new vehicle object and add it to the existing system of bodies
    bodies.create_empty_body('Capsule')
    # Update the capsule shape parameters
    set_capsule_shape_parameters(shape_parameters,
                                 bodies,
                                 capsule_density)



###########################################################################
# BENCHMARK UTILITIES #####################################################
###########################################################################


# NOTE TO STUDENTS: THIS FUNCTION CAN BE EXTENDED TO GENERATE A MORE ROBUST BENCHMARK (USING MORE THAN 2 RUNS)
def generate_benchmarks(benchmark_step_size,
                        simulation_start_epoch: float,
                        bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                        benchmark_propagator_settings: tudatpy.kernel.numerical_simulation.propagation_setup.propagator.TranslationalStatePropagatorSettings,
                        are_dependent_variables_present: bool,
                        output_path: str = None,
                        step_size_name: str = '',
                        benchmark_case: int = 3,
                        benchmark_initial_state: np.ndarray = np.zeros(6),
                        termination_epoch: float = 0.,
                        divide_step_size_of: float = 1.,
                        benchmark_coeff_set = propagation_setup.integrator.CoefficientSets.rkdp_87,
                        galileo_mission: bool = False):
    """
    Function to generate to accurate benchmarks.

    This function runs two propagations with two different integrator settings that serve as benchmarks for
    the nominal runs. The state and dependent variable history for both benchmarks are returned and, if desired, 
    they are also written to files (to the directory ./SimulationOutput/benchmarks/) in the following way:
    * benchmark_1_states.dat, benchmark_2_states.dat
        The numerically propagated states from the two benchmarks.
    * benchmark_1_dependent_variables.dat, benchmark_2_dependent_variables.dat
        The dependent variables from the two benchmarks.

    Parameters
    ----------
    benchmark_step_size :
        Time step of the benchmark that will be used. Two benchmark simulations will be run, both fixed-step 8th order
         (first benchmark uses benchmark_step_size, second benchmark uses 2.0 * benchmark_step_size)
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
        System of bodies present in the simulation.
    benchmark_propagator_settings
        Propagator settings object which is used to run the benchmark propagations.
    are_dependent_variables_present : bool
        If there are dependent variables to save.
    output_path : str (default: None)
        If and where to save the benchmark results (if None, results are NOT written).

    Returns
    -------
    return_list : list
        List of state and dependent variable history in this order: state_1, state_2, dependent_1_ dependent_2.
    """

    ### CREATION OF THE TWO BENCHMARKS ###

    if galileo_mission:
        benchmark_case = 4

    time_termination_settings = propagation_setup.propagator.time_termination(
        termination_epoch,
        terminate_exactly_on_final_condition=True
    )

    fpa_termination_settings_arc_0 = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.flight_path_angle('Capsule', 'Jupiter'),
        limit_value=0.,
        use_as_lower_limit=False,
        terminate_exactly_on_final_condition=False,
    )

    fpa_termination_settings_arc_1 = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.flight_path_angle('Capsule', 'Jupiter'),
        limit_value=0.,
        use_as_lower_limit=False,
        terminate_exactly_on_final_condition=False
    )

    atmosph_altitude_termination_settings_arc_0 = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.altitude('Capsule', 'Jupiter'),
        limit_value=4.5e6*1e3,  # + benchmark_step_size*2e2*1e3,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=True,
        termination_root_finder_settings=root_finders.bisection(relative_variable_tolerance=1e-16, absolute_variable_tolerance=1e-16, root_function_tolerance=1e-16)
    )

    atmosph_altitude_termination_settings_arc_1 = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.altitude('Capsule', 'Jupiter'),
        limit_value=4.5e6*1e3,  # + benchmark_step_size*2e2*1e3 + 10,
        use_as_lower_limit=False,
        terminate_exactly_on_final_condition=True,
        termination_root_finder_settings=root_finders.bisection(relative_variable_tolerance=1e-16,
                                                                absolute_variable_tolerance=1e-16,
                                                                root_function_tolerance=1e-16)
    )

    hybrid_termination_settings_arc_0 = propagation_setup.propagator.hybrid_termination([atmosph_altitude_termination_settings_arc_0,fpa_termination_settings_arc_0],
                                                                                        fulfill_single_condition=True)
    hybrid_part_termination_settings_arc_1 = propagation_setup.propagator.hybrid_termination([atmosph_altitude_termination_settings_arc_1, fpa_termination_settings_arc_1],
                                                                                             fulfill_single_condition=False)

    number_of_arcs = 1
    if benchmark_case == 0:
        first_benchmark_step_size = np.array([benchmark_step_size])  # s
    elif benchmark_case == 1:
        first_benchmark_step_size = np.array([benchmark_step_size])  # s
        if not benchmark_initial_state.any():
            return -1
        benchmark_propagator_settings.initial_states = benchmark_initial_state
    elif benchmark_case == 2:
        first_benchmark_step_size = np.array([benchmark_step_size])  # s
        if not benchmark_initial_state.any():
            return -1
        benchmark_propagator_settings.initial_states = benchmark_initial_state
    elif benchmark_case == 3:
        if type(benchmark_step_size) is list:
            first_benchmark_step_size = np.array(benchmark_step_size)
        else:
            first_benchmark_step_size = np.array([benchmark_step_size, benchmark_step_size / divide_step_size_of, benchmark_step_size])  # s
        number_of_arcs = 3
    elif benchmark_case == 4:  # galileo case
        first_benchmark_step_size = np.array([benchmark_step_size])  # s
    else:
        return Warning('Wrong case parameter chosen for the benchmark. Allowed values are integers 0, 1, 2, 3, 4')

    # Define benchmarks' step sizes
    second_benchmark_step_size = 2.0 * first_benchmark_step_size

    first_benchmark_states = dict()
    second_benchmark_states = dict()
    first_benchmark_dependent_variables = dict()
    second_benchmark_dependent_variables = dict()

    simulation_start_epoch2 = simulation_start_epoch
    fun_initial_state = benchmark_propagator_settings.initial_states
    fun_initial_state2 = fun_initial_state
    for i in range(number_of_arcs):

        if benchmark_case == 0 or (i == 0 and benchmark_case == 3):
            if termination_epoch != 0:
                benchmark_propagator_settings.termination_settings = time_termination_settings
            else:
                benchmark_propagator_settings.termination_settings = hybrid_termination_settings_arc_0
        if benchmark_case == 1 or (i == 1 and benchmark_case == 3):
            if termination_epoch != 0:
                benchmark_propagator_settings.termination_settings = time_termination_settings
            else:
                benchmark_propagator_settings.termination_settings = hybrid_part_termination_settings_arc_1
        if benchmark_case == 2 or (i == 2 and benchmark_case == 3):
            benchmark_propagator_settings.termination_settings = get_termination_settings(simulation_start_epoch)

        # Create integrator settings for the first benchmark, using a fixed step size RKDP8(7) integrator
        # (the minimum and maximum step sizes are set equal, while both tolerances are set to inf)
        benchmark_integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
            simulation_start_epoch,
            first_benchmark_step_size[i],
            benchmark_coeff_set,  #.CoefficientSets.rkdp_87,
            first_benchmark_step_size[i],
            first_benchmark_step_size[i],
            np.inf,
            np.inf)

        benchmark_propagator_settings.initial_states = fun_initial_state
        print(f'Running first benchmark...   (step size: {first_benchmark_step_size[i]} s)')
        first_dynamics_simulator = numerical_simulation.SingleArcSimulator(
            bodies,
            benchmark_integrator_settings,
            benchmark_propagator_settings, print_dependent_variable_data=False)

        # Create integrator settings for the second benchmark in the same way
        benchmark_integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
            simulation_start_epoch2,
            second_benchmark_step_size[i],
            benchmark_coeff_set,   #CoefficientSets.rkdp_87,
            second_benchmark_step_size[i],
            second_benchmark_step_size[i],
            np.inf,
            np.inf)

        benchmark_propagator_settings.initial_states = fun_initial_state2
        print(f'Running second benchmark...   (step size: {second_benchmark_step_size[i]} s)')
        second_dynamics_simulator = numerical_simulation.SingleArcSimulator(
            bodies,
            benchmark_integrator_settings,
            benchmark_propagator_settings, print_dependent_variable_data=False)


        ### WRITE BENCHMARK RESULTS TO FILE ###
        # Retrieve state history
        first_benchmark_partial_states = first_dynamics_simulator.state_history
        first_benchmark_partial_dependent_variables = first_dynamics_simulator.dependent_variable_history
        first_benchmark_states = first_benchmark_states | first_benchmark_partial_states
        first_benchmark_dependent_variables = first_benchmark_dependent_variables | first_benchmark_partial_dependent_variables

        second_benchmark_partial_states = second_dynamics_simulator.state_history
        second_benchmark_partial_dependent_variables = second_dynamics_simulator.dependent_variable_history
        second_benchmark_states = second_benchmark_states | second_benchmark_partial_states
        second_benchmark_dependent_variables = second_benchmark_dependent_variables | second_benchmark_partial_dependent_variables

        final_epoch_st = list(first_benchmark_partial_dependent_variables.keys())[-1]
        print(f'First benchmark final conditions:\n'
              f'- altitude: {list(first_benchmark_partial_dependent_variables.values())[-1][4]/1e3:.3f} km  \n'
              f'- f.p.a.: {list(first_benchmark_partial_dependent_variables.values())[-1][5]*180/np.pi:.3f} deg\n'
              f'- epoch: {final_epoch_st} s  ({(final_epoch_st-simulation_start_epoch)/constants.JULIAN_DAY:.3f} days since departure)')
        final_epoch_nd = list(second_benchmark_partial_dependent_variables.keys())[-1]
        print(f'Second benchmark final conditions:\n'
              f'- altitude: {list(second_benchmark_partial_dependent_variables.values())[-1][4] / 1e3:.3f} km  \n'
              f'- f.p.a.: {list(second_benchmark_partial_dependent_variables.values())[-1][5] * 180 / np.pi:.3f} deg\n'
              f'- epoch: {final_epoch_nd} s  ({(final_epoch_nd - simulation_start_epoch2) / constants.JULIAN_DAY:.3f} days since departure)')

        simulation_start_epoch = list(first_benchmark_partial_states.keys())[-1]
        fun_initial_state = np.vstack(list(first_benchmark_partial_states.values()))[-1, :]

        simulation_start_epoch2 = list(second_benchmark_partial_states.keys())[-1]
        fun_initial_state2 = np.vstack(list(second_benchmark_partial_states.values()))[-1, :]

    # Write results to files
    if output_path is not None:
        save2txt(first_benchmark_states, 'benchmark_1_states'+ step_size_name +'.dat', output_path)
        save2txt(second_benchmark_states, 'benchmark_2_states'+ step_size_name +'.dat', output_path)
    # Add items to be returned
    return_list = [first_benchmark_states,
                   second_benchmark_states]

    ### DO THE SAME FOR DEPENDENT VARIABLES ###
    if are_dependent_variables_present:
        # Retrieve dependent variable history
        first_benchmark_dependent_variable = first_benchmark_dependent_variables
        second_benchmark_dependent_variable = second_benchmark_dependent_variables
        # Write results to file
        if output_path is not None:
            save2txt(first_benchmark_dependent_variable, 'benchmark_1_dependent_variables'+ step_size_name +'.dat',  output_path)
            save2txt(second_benchmark_dependent_variable,  'benchmark_2_dependent_variables'+ step_size_name +'.dat',  output_path)
        # Add items to be returned
        return_list.append(first_benchmark_dependent_variable)
        return_list.append(second_benchmark_dependent_variable)

    return return_list, simulation_start_epoch, fun_initial_state


def compare_benchmarks(first_benchmark: dict,
                       second_benchmark: dict,
                       output_path: str,
                       filename: str) -> dict:
    """
    It compares the results of two benchmark runs.

    It uses an 8th-order Lagrange interpolator to compare the state (or the dependent variable, depending on what is
    given as input) history. The difference is returned in form of a dictionary and, if desired, written to a file named
    filename and placed in the directory output_path.

    Parameters
    ----------
    first_benchmark : dict
        State (or dependent variable history) from the first benchmark.
    second_benchmark : dict
        State (or dependent variable history) from the second benchmark.
    output_path : str
        If and where to save the benchmark results (if None, results are NOT written).
    filename : str
        Name of the output file.

    Returns
    -------
    benchmark_difference : dict
        Interpolated difference between the two benchmarks' state (or dependent variable) history.
    """
    # Create 8th-order Lagrange interpolator for first benchmark
    benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        first_benchmark, interpolators.lagrange_interpolation(8))
    # Calculate the difference between the benchmarks
    print('Calculating benchmark differences...')
    # Initialize difference dictionaries
    benchmark_difference = dict()
    # Calculate the difference between the states and dependent variables in an iterative manner
    for second_epoch in second_benchmark.keys():
        benchmark_difference[second_epoch] = benchmark_interpolator.interpolate(second_epoch) - \
                                             second_benchmark[second_epoch]
    # Write results to files
    if output_path is not None:
        save2txt(benchmark_difference, filename, output_path)
    # Return the interpolator
    return benchmark_difference


def compare_models(first_model: dict,
                   second_model: dict,
                   interpolation_epochs: np.ndarray,
                   output_path: str,
                   filename: str) -> dict:
    """
    It compares the results of two runs with different model settings.
    It uses an 8th-order Lagrange interpolator to compare the state (or the dependent variable, depending on what is
    given as input) history. The difference is returned in form of a dictionary and, if desired, written to a file named
    filename and placed in the directory output_path.
    Parameters
    ----------
    first_model : dict
        State (or dependent variable history) from the first run.
    second_model : dict
        State (or dependent variable history) from the second run.
    interpolation_epochs : np.ndarray
        Vector of epochs at which the two runs are compared.
    output_path : str
        If and where to save the benchmark results (if None, results are NOT written).
    filename : str
        Name of the output file.
    Returns
    -------
    model_difference : dict
        Interpolated difference between the two simulations' state (or dependent variable) history.
    """
    # Create interpolator settings
    interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation=interpolators.use_boundary_value)
    # Create 8th-order Lagrange interpolator for both cases
    first_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        first_model, interpolator_settings)
    second_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        second_model, interpolator_settings)
    # Calculate the difference between the first and second model at specific epochs
    model_difference = {epoch: second_interpolator.interpolate(epoch) - first_interpolator.interpolate(epoch)
                        for epoch in interpolation_epochs}
    # Write results to files
    if output_path is not None:
        save2txt(model_difference,
                 filename,
                 output_path)
    # Return the model difference
    return model_difference


###########################################################################
# OTHER UTILITIES #########################################################
###########################################################################


def plot_base_trajectory(state_history,
                         fig = plt.figure(),
                         ax = plt.axes(projection='3d')):
    if type(state_history) == dict:
        sim_result = np.vstack(list(state_history.values()))
    else:
        sim_result = state_history[:,1:]
    # Plot 3-D Trajectory
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    # draw jupiter
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = jupiter_radius * np.cos(u) * np.sin(v)
    y = jupiter_radius * np.sin(u) * np.sin(v)
    z = jupiter_radius * np.cos(v)
    ax.plot_wireframe(x, y, z, color="saddlebrown")

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Jupiter aerocapture trajectory')

    # draw post-ae possibly flyby moon orbit
    for moon in galilean_moons_data.keys():
        moon_sma = galilean_moons_data[moon]['SMA']
        theta_angle = np.linspace(0, 2 * np.pi, 200)
        x_m = moon_sma * np.cos(theta_angle)
        y_m = moon_sma * np.sin(theta_angle)
        z_m = np.zeros(len(theta_angle))
        ax.plot3D(x_m, y_m, z_m, 'b')

    xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
    XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim * 0.75)
    ax.set_aspect('auto')

    ax.plot3D(sim_result[:, 0], sim_result[:, 1], sim_result[:, 2], 'gray')

    return fig, ax


def plot_time_step(state_history):
    epochs_vector = np.vstack(list(state_history.keys()))
    epochs_plot = (epochs_vector - epochs_vector[0]) / constants.JULIAN_DAY

    fig, ax = plt.subplots(figsize=(6, 5))
    time_steps = np.diff(epochs_vector, n=1, axis=0)
    ax.plot(epochs_plot[:-1], time_steps)
    ax.scatter(epochs_plot[:-1], time_steps)

    return fig, ax