import numpy as np
from JupiterTrajectory_GlobalParameters import *
from handle_functions import *
import CapsuleEntryUtilities as Util

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel import numerical_simulation

class SingleFlybyApproach:

    def __init__(self,
                 decision_variable_range,
                 epoch_of_flyby,
                 moon_of_flyby,
                 flyby_safety_altitude: float = 0,
                 orbit_datapoints: int = 200,
                 verbose: bool = False):

        if moon_of_flyby not in ['Io', 'Europa', 'Ganymede', 'Callisto']:
            raise Exception('Moon name is invalid.')

        self.flyby_epoch = epoch_of_flyby
        self.flyby_moon = moon_of_flyby
        self.decision_variable_range = decision_variable_range
        self.flyby_safety_altitude = flyby_safety_altitude
        self.orbit_datapoints = orbit_datapoints
        self.verbose = verbose

    def get_cartesian_state_history(self) -> dict:
        """
        Returns the full history of the propagated state, converted to Cartesian states
        Parameters
        ----------
        none
        Returns
        -------
        dict
        """
        return self.state_history_function()

    def get_dependent_variables_history(self) -> dict:
        """
        Returns the full history of the propagated dependent variables.
        Parameters
        ----------
        none
        Returns
        -------
        dict
        """
        return self.dependent_variable_history_function()

    def get_bounds(self):

        return self.decision_variable_range

    def get_moon_data(self):
        moon_data = galilean_moons_data[self.flyby_moon]
        return moon_data

    def set_flyby_epoch(self, epoch_of_flyby):
        warnings.warn('rerun fitness function for changes to be effective')
        self.flyby_epoch = epoch_of_flyby

    def fitness(self,
                orbital_parameters
                ):
        """
        Calculates the trajectory with the orbital parameters given as argument.
        This function uses the orbital parameters to compute the arrival trajectory.
        The fitness, currently set to zero, can be computed here: it will be used during the
        optimization process.
        Parameters
        ----------
        orbital_parameters : list of floats
            List of orbital parameters to be optimized.
            0: interplanetary arrival velocity
            1: atmospheric entry flight path angle
            2: angle between interplanetary velocity vector and negative position vector
        Returns
        -------
        fitness : float
            Fitness value, for optimization.
        """

        equatorial_orbit = True
        debug = False

        interplanetary_arrival_velocity = orbital_parameters[0]
        atmospheric_entry_fpa = orbital_parameters[1]
        delta_angle_from_hohmann_trajectory = orbital_parameters[2]

        moon_data = self.get_moon_data()
        moon_radius = moon_data['Radius']
        moon_SOI_radius = moon_data['SOI_Radius']
        mu_moon = moon_data['mu']

        # Retrieve moon state
        moon_flyby_state = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name=self.flyby_moon,
            observer_body_name="Jupiter",
            reference_frame_name=global_frame_orientation,
            aberration_corrections="NONE",
            ephemeris_time=self.flyby_epoch)
        moon_position = moon_flyby_state[0:3]
        moon_velocity = moon_flyby_state[3:6]
        if equatorial_orbit:
            moon_position[2], moon_velocity[2] = 0., 0.

        orbit_axis = unit_vector(np.cross(moon_position, moon_velocity))

        first_arc_angular_momentum = orbit_axis * jupiter_SOI_radius * interplanetary_arrival_velocity * np.sin(delta_angle_from_hohmann_trajectory)
        first_arc_orbital_energy = orbital_energy(jupiter_SOI_radius, interplanetary_arrival_velocity, jupiter_gravitational_parameter)


        first_arc_semilatus_rectum = first_arc_angular_momentum ** 2 / jupiter_gravitational_parameter
        first_arc_semimajor_axis = - jupiter_gravitational_parameter / (2 * first_arc_orbital_energy)
        first_arc_eccentricity = np.sqrt(1 - first_arc_semilatus_rectum / first_arc_semimajor_axis)

        if debug:
            sigma_angles = np.linspace(0,np.pi,100)
            function_values_lol = np.zeros(len(sigma_angles))
            for i, sigma_angle in enumerate(sigma_angles):
                fpa_function = calculate_fpa_from_flyby_geometry(sigma_angle,
                                                                 arc_1_initial_velocity=interplanetary_arrival_velocity,
                                                                 arc_1_initial_radius=jupiter_SOI_radius,
                                                                 delta_hoh=delta_angle_from_hohmann_trajectory,
                                                                 arc_2_final_radius=atmospheric_entry_altitude+jupiter_radius,
                                                                 flyby_moon=self.flyby_moon,
                                                                 flyby_epoch=self.flyby_epoch,
                                                                 equatorial_approximation=equatorial_orbit)
                function_values_lol[i] = fpa_function - atmospheric_entry_fpa

            plt.plot(sigma_angles, function_values_lol * 180 / np.pi)
            plt.show()

        # Finding flyby pericenter
        tolerance = 1e-12
        c_point, f_c, i = regula_falsi_illinois((0., np.pi), calculate_fpa_from_flyby_geometry,
                                                atmospheric_entry_fpa, tolerance,
                                                illinois_addition=True,
                                                arc_1_initial_velocity=interplanetary_arrival_velocity,
                                                arc_1_initial_radius=jupiter_SOI_radius,
                                                delta_hoh=delta_angle_from_hohmann_trajectory,
                                                arc_2_final_radius=jupiter_radius + atmospheric_entry_altitude,
                                                flyby_moon=self.flyby_moon,
                                                flyby_epoch=self.flyby_epoch,
                                                equatorial_approximation=True)
        sigma_angle = c_point
        calculated_fpa = f_c + atmospheric_entry_fpa

        # Debugging
        if self.verbose:
            print(f'Number of iterations: {i}')
            # print(f'Flyby pericenter altitude: {(flyby_pericenter-moon_radius)/1e3} km')
            print(f'Sigma angle used: {sigma_angle * 180 / np.pi} deg')
            print(f'f.p.a. result of root finder: {calculated_fpa * 180 / np.pi} deg')



        flyby_initial_position = rotate_vectors_by_given_matrix(rotation_matrix(orbit_axis, -sigma_angle),
                                                                unit_vector(moon_velocity)) * moon_SOI_radius

        # Energy and angular momentum for first arc are calculated above

        first_arc_arrival_position = moon_position + flyby_initial_position
        first_arc_arrival_radius = LA.norm(first_arc_arrival_position)
        first_arc_arrival_velocity = np.sqrt(
            2 * (first_arc_orbital_energy + jupiter_gravitational_parameter / first_arc_arrival_radius))

        first_arc_arrival_fpa = - np.arccos(
            LA.norm(first_arc_angular_momentum) / (first_arc_arrival_radius * first_arc_arrival_velocity))
        first_arc_arrival_velocity_vector = rotate_vectors_by_given_matrix(
            rotation_matrix(orbit_axis, np.pi / 2 - first_arc_arrival_fpa),
            unit_vector(first_arc_arrival_position)) * first_arc_arrival_velocity

        flyby_initial_velocity_vector = first_arc_arrival_velocity_vector - moon_velocity
        flyby_v_inf_t = LA.norm(flyby_initial_velocity_vector)

        flyby_axis = unit_vector(np.cross(flyby_initial_position, flyby_initial_velocity_vector))

        phi_2_angle = np.arccos(np.dot(unit_vector(-moon_velocity), unit_vector(flyby_initial_velocity_vector)))
        if np.dot(np.cross(-moon_velocity, flyby_initial_velocity_vector), flyby_axis) < 0:
            phi_2_angle = - phi_2_angle + 2 * np.pi

        delta_angle = np.arccos(np.dot(unit_vector(-moon_velocity), unit_vector(flyby_initial_position)))
        if np.dot(np.cross(-moon_velocity, flyby_initial_position), flyby_axis) < 0:
            delta_angle = 2 * np.pi - delta_angle

        B_parameter = moon_SOI_radius * np.sin(phi_2_angle - delta_angle)
        flyby_alpha_angle = 2 * np.arcsin(1 / np.sqrt(1 + (B_parameter ** 2 * flyby_v_inf_t ** 4) / mu_moon ** 2))
        beta_angle = phi_2_angle + flyby_alpha_angle / 2 - np.pi / 2

        position_rot_angle = 2 * (- delta_angle + beta_angle)

        flyby_final_position = rotate_vectors_by_given_matrix(rotation_matrix(flyby_axis, position_rot_angle),
                                                              flyby_initial_position)

        flyby_final_velocity_vector = rotate_vectors_by_given_matrix(rotation_matrix(flyby_axis, flyby_alpha_angle),
                                                                     flyby_initial_velocity_vector)

        flyby_pericenter = mu_moon / (flyby_v_inf_t ** 2) * (
                np.sqrt(1 + (B_parameter ** 2 * flyby_v_inf_t ** 4) / (mu_moon ** 2)) - 1)
        flyby_altitude = flyby_pericenter - moon_radius
        if flyby_altitude < 0:
            print(f'Flyby impact! Altitude: {flyby_altitude / 1e3} km     Sigma: {sigma_angle * 180 / np.pi} deg')
        #     arc_2_final_fpa = arc_2_final_fpa + 1000
        flyby_orbital_energy = orbital_energy(LA.norm(flyby_initial_position), flyby_v_inf_t, mu_parameter=mu_moon)
        flyby_sma = - mu_moon / (2 * flyby_orbital_energy)
        flyby_eccentricity = 1 - flyby_pericenter / flyby_sma

        true_anomaly_boundary = 2 * np.pi - delta_angle + beta_angle
        true_anomaly_boundary = true_anomaly_boundary if true_anomaly_boundary < 2 * np.pi else true_anomaly_boundary - 2 * np.pi
        true_anomaly_range = np.array([-true_anomaly_boundary, true_anomaly_boundary])
        flyby_elapsed_time = delta_t_from_delta_true_anomaly(true_anomaly_range,
                                                             eccentricity=flyby_eccentricity,
                                                             semi_major_axis=flyby_sma,
                                                             mu_parameter=mu_moon)
        flyby_final_epoch = self.flyby_epoch + flyby_elapsed_time
        moon_flyby_final_state = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name=self.flyby_moon,
            observer_body_name="Jupiter",
            reference_frame_name=global_frame_orientation,
            aberration_corrections="NONE",
            ephemeris_time=flyby_final_epoch)
        moon_final_position = moon_flyby_final_state[0:3]
        moon_final_velocity = moon_flyby_final_state[3:6]
        if equatorial_orbit:
            moon_final_position[2], moon_final_velocity[2] = 0., 0.

        second_arc_departure_position = moon_final_position + flyby_final_position
        second_arc_departure_velocity_vector = moon_final_velocity + flyby_final_velocity_vector

        second_arc_departure_radius = LA.norm(second_arc_departure_position)
        second_arc_departure_velocity = LA.norm(second_arc_departure_velocity_vector)

        second_arc_angular_momentum_vector = np.cross(second_arc_departure_position,
                                                      second_arc_departure_velocity_vector)
        second_arc_angular_momentum = LA.norm(second_arc_angular_momentum_vector)
        second_arc_orbital_energy = orbital_energy(second_arc_departure_radius, second_arc_departure_velocity, jupiter_gravitational_parameter)

        print(f'\n\nFirst arc orbital specific energy: {first_arc_orbital_energy / 1e3:.3f} kJ/kg')
        print(f'Second arc orbital specific energy: {second_arc_orbital_energy/1e3:.3f} kJ/kg')

        second_arc_semilatus_rectum = second_arc_angular_momentum ** 2 / jupiter_gravitational_parameter
        second_arc_semimajor_axis = - jupiter_gravitational_parameter / (2 * second_arc_orbital_energy)
        second_arc_eccentricity = np.sqrt(1 - second_arc_semilatus_rectum / second_arc_semimajor_axis)

        second_arc_arrival_velocity = np.sqrt(
            2 * (second_arc_orbital_energy + jupiter_gravitational_parameter / (jupiter_radius + atmospheric_entry_altitude)))

        second_arc_arrival_fpa = - np.arccos(
            np.clip(second_arc_angular_momentum / ((jupiter_radius + atmospheric_entry_altitude) * second_arc_arrival_velocity), -1, 1))

        flyby_pericenter = mu_moon / (flyby_v_inf_t ** 2) * (np.sqrt(1 + (B_parameter ** 2 * flyby_v_inf_t ** 4) / (mu_moon ** 2)) - 1)
        flyby_altitude = flyby_pericenter - moon_radius

        flyby_delta_v = LA.norm(flyby_final_velocity_vector - flyby_initial_velocity_vector)

        alpha_max = 2 * np.arcsin(1 / (1 + flyby_v_inf_t ** 2 / (mu_moon / moon_radius)))
        flyby_delta_v_max = 2 * flyby_v_inf_t * np.sin(alpha_max / 2)

        # Checks if the flyby pericenter is above minimum safety altitude set at the beginning
        if self.verbose:
            print(f'\nFlyby altitude: {flyby_altitude / 1e3:.3f} km')
            print(f'Flyby alpha angle: {flyby_alpha_angle * 180 / np.pi:.3f} deg')
            print(f'Flyby delta_v: {flyby_delta_v / 1e3:.3f} km/s')
            print(f'Max delta_v achievable for {self.flyby_moon}: {flyby_delta_v_max / 1e3:.3f} km/s')
            if flyby_altitude < self.flyby_safety_altitude:
                warnings.warn(f'\nMOON IMPACT - FLYBY FAILED')

        self.flyby_approach_parameters_function = lambda: [second_arc_semimajor_axis, second_arc_eccentricity, flyby_pericenter, flyby_delta_v, flyby_delta_v_max, second_arc_arrival_fpa]


        # Add the objective and constraint values into the fitness vector
        fitness = 0.0
        return [fitness]