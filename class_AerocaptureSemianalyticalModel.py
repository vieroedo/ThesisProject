import numpy as np
from JupiterTrajectory_GlobalParameters import *
from handle_functions import *
import CapsuleEntryUtilities as Util
import Just_aerocapture.Analytical_approach.second_order_equations_aerocapture as ae_second_order
import Just_aerocapture.Analytical_approach.first_order_equations_aerocapture as ae_first_order

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel import numerical_simulation


def entry_velocity_from_interplanetary(interplanetary_arrival_velocity_in_jupiter_frame):
    pre_ae_departure_radius = jupiter_SOI_radius
    pre_ae_departure_velocity_norm = interplanetary_arrival_velocity_in_jupiter_frame
    pre_ae_orbital_energy = orbital_energy(pre_ae_departure_radius, pre_ae_departure_velocity_norm, jupiter_gravitational_parameter)
    pre_ae_arrival_radius = jupiter_radius + atmospheric_entry_altitude
    pre_ae_arrival_velocity_norm = velocity_from_energy(pre_ae_orbital_energy, pre_ae_arrival_radius, jupiter_gravitational_parameter)
    return pre_ae_arrival_velocity_norm


class AerocaptureSemianalyticalModel:

    def __init__(self,
                 decision_variable_range,
                 orbit_datapoints: int,
                 equations_order: int,
                 atmospheric_interface_altitude: float
                 ):
        """
                Constructor for the AerocaptureSemianalyticalSecondOrder class.
                Parameters
                ----------

                Returns
                -------
                none
                """

        # Set arguments as attributes
        self.decision_variable_range = decision_variable_range
        # self.initial_epoch = epoch
        # self.number_of_epochs_to_plot = number_of_epochs_to_plot
        self.atmospheric_interface_altitude = atmospheric_interface_altitude
        self.orbit_datapoints = orbit_datapoints
        if equations_order == 1 or equations_order == 2:
            self.equations_order = equations_order
        else:
            raise Exception('The requested order for the aerocapture entry is unavailable.')

        self.state_history_function = None
        self.last_run_entry_parameters = None

    def get_cartesian_state_history(self, aerocapture_initial_position: np.ndarray, orbital_axis: np.ndarray) -> dict:
        """
        Returns the full history of the propagated state, converted to Cartesian states
        Parameters
        ----------
        none
        Returns
        -------
        dict
        """

        self.create_cartesian_state_history(aerocapture_initial_position, orbital_axis)
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

    def create_cartesian_state_history(self, aerocapture_initial_position: np.ndarray, orbital_axis: np.ndarray):
        if np.shape(orbital_axis) != (3,):
            raise Exception('Wrong orbital axis inserted')
        if np.shape(aerocapture_initial_position) != (3,):
            raise Exception('Wrong aerocapture_initial_position shape inserted. maybe it is just transposed?')
        if self.last_run_entry_parameters is None:
            raise Exception('Fitness function has never been run')

        atmospheric_entry_velocity_norm = self.last_run_entry_parameters[0]
        atmospheric_entry_fpa = self.last_run_entry_parameters[1]

        aerocapture_problem_parameters = self.aerocapture_parameters_function()
        atmospheric_exit_phase_angle = aerocapture_problem_parameters[5]

        pre_ae_angular_momentum = orbital_axis * self.atmospheric_interface_altitude * \
                                  atmospheric_entry_velocity_norm * np.cos(atmospheric_entry_fpa)

        atmosph_entry_rot_matrix = rotation_matrix(orbital_axis, atmospheric_exit_phase_angle)
        atmospheric_entry_final_position = rotate_vectors_by_given_matrix(atmosph_entry_rot_matrix, aerocapture_initial_position)

        # dependent_variables = self.dependent_variable_history_function()
        dependent_variables = np.vstack(list(self.dependent_variable_history_function().values()))
        ae_fpas = dependent_variables[:, 0]
        ae_velocities = dependent_variables[:, 1]
        ae_radii = dependent_variables[:, 2]
        ae_range_angles = dependent_variables[:, 7]

        # x_unal, y_unal, z_unal = cartesian_3d_from_polar(ae_radii,np.zeros(len(ae_radii)),ae_range_angles)
        #
        # position_states_unaligned = np.vstack((x_unal, y_unal, z_unal))

        # atmospheric_entry_reference_angle = np.arcsin(LA.norm(np.cross(x_axis, self.atmospheric_entry_initial_position))/LA.norm(self.atmospheric_entry_initial_position))
        # if np.dot(z_axis, np.cross(x_axis, self.atmospheric_entry_initial_position)) < 0:
        #     atmospheric_entry_reference_angle = atmospheric_entry_reference_angle + np.pi

        # entry_positions_rot_matrix = rotation_matrix(pre_ae_angular_momentum, atmospheric_entry_reference_angle)
        # position_states = rotate_vectors_by_given_matrix(entry_positions_rot_matrix,
        #                                                  position_states_unaligned)

        position_states = np.zeros((len(ae_radii),3))
        for i in range(len(ae_radii)):
            rot_matrix_pos = rotation_matrix(orbital_axis,ae_range_angles[i])
            position_states[i,:] = rotate_vectors_by_given_matrix(rot_matrix_pos,unit_vector(aerocapture_initial_position)) * ae_radii[i]


        velocity_states =np.zeros(np.shape(position_states))
        if max(np.shape(velocity_states)) <= 3:
            raise Exception('too few instances of the reentry trajectory (less or equal than 3)')
        for i in range(len(ae_radii)):
            entry_velocities_rot_matrix = rotation_matrix(orbital_axis, np.pi/2 - ae_fpas[i])
            velocity_states[i,:] = rotate_vectors_by_given_matrix(entry_velocities_rot_matrix, unit_vector(position_states[i, :])) * ae_velocities[i]

        cartesian_states = np.concatenate((position_states, velocity_states), axis=1)
        self.state_history_function = lambda: dict(zip(ae_range_angles, cartesian_states))


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
            2: ...
        Returns
        -------
        fitness : float
            Fitness value, for optimization.
        """

        interplanetary_arrival_velocity = orbital_parameters[0]
        atmospheric_entry_velocity_norm = entry_velocity_from_interplanetary(interplanetary_arrival_velocity)

        atmospheric_entry_fpa = orbital_parameters[1]

        self.last_run_entry_parameters = [atmospheric_entry_velocity_norm,atmospheric_entry_fpa]

        if self.equations_order == 2:
            tau_entry, tau_minimum_altitude, tau_exit, a1, a2, a3 = \
                ae_second_order.calculate_tau_boundaries_second_order_equations(
                                                                atmospheric_entry_altitude + jupiter_radius,
                                                                atmospheric_entry_velocity_norm, atmospheric_entry_fpa,
                                                                K_hypersonic=Util.vehicle_hypersonic_K_parameter)

            tau_linspace = np.linspace(tau_entry, tau_exit, self.orbit_datapoints)

            # (radius, velocity, flight_path_angle, density, drag, lift, wall_heat_flux)
            aerocapture_quantities, empty_variable = ae_second_order.second_order_approximation_aerocapture(
                                                                    tau_linspace, tau_minimum_altitude, a1, a2, a3,
                                                                    atmospheric_entry_altitude + jupiter_radius,
                                                                    atmospheric_entry_velocity_norm,
                                                                    atmospheric_entry_fpa,
                                                                    K_hypersonic=Util.vehicle_hypersonic_K_parameter)

            # Minimum altitude
            x_tau_min_alt = ae_second_order.x_tau_function(tau_minimum_altitude, a1, a2, a3)
            minimum_altitude = - jupiter_scale_height * x_tau_min_alt + atmospheric_entry_altitude
        elif self.equations_order == 1:

            fpa_entry, fpa_minimum_altitude, fpa_exit = ae_first_order.calculate_fpa_boundaries(atmospheric_entry_fpa)
            fpa_linspace = np.linspace(fpa_entry, fpa_exit, self.orbit_datapoints)

            aerocapture_quantities, other_data = ae_first_order.first_order_approximation_aerocapture(
                                                                        fpa_linspace, fpa_entry, fpa_minimum_altitude,
                                                                        atmospheric_entry_altitude + jupiter_radius,
                                                                        atmospheric_entry_velocity_norm)

            # Minimum altitude
            minimum_altitude = other_data[0]

        else:
            raise Exception('The requested order for the aerocapture entry is unavailable.')

        ae_radii = aerocapture_quantities[0]
        ae_velocities = aerocapture_quantities[1]
        ae_fpas = aerocapture_quantities[2]
        ae_densities = aerocapture_quantities[3]
        ae_drag = aerocapture_quantities[4]
        ae_lift = aerocapture_quantities[5]
        ae_wall_hfx = aerocapture_quantities[6]
        ae_range_angles = aerocapture_quantities[7]

        # Numerically calculate elapsed time - trapezoidal integration
        integrand_function = ae_radii / (ae_velocities*np.cos(ae_fpas))
        delta_t = np.trapz(integrand_function, ae_range_angles)

        # Simple integration
        # delta_t_coarse = 0
        # theta_0 = 0
        # for no,theta in enumerate(ae_range_angles):
        #     delta_t_coarse = delta_t_coarse + integrand_function[no]*(theta-theta_0)
        #     theta_0 = theta

        # Atmosphere exit fpa
        atmospheric_exit_fpa = ae_fpas[-1]

        # Atmosphere exit velocity
        atmospheric_exit_velocity_norm = ae_velocities[-1]

        # Travelled distance (assumed at surface)
        atmospheric_entry_final_phase_angle = ae_range_angles[-1]
        final_distance_travelled = atmospheric_entry_final_phase_angle * jupiter_radius

        atmospheric_exit_radius = ae_radii[-1]

        dependent_variables = np.vstack((ae_fpas, ae_velocities, ae_radii, ae_densities, ae_drag, ae_lift, ae_wall_hfx, ae_range_angles)).T

        # The range angle is the independent variable
        self.dependent_variable_history_function = lambda: dict(zip(ae_range_angles, dependent_variables))
        self.aerocapture_parameters_function = lambda: [atmospheric_exit_fpa, atmospheric_exit_velocity_norm,
                                                        final_distance_travelled, minimum_altitude, atmospheric_exit_radius, atmospheric_entry_final_phase_angle, delta_t]

        # Add the objective and constraint values into the fitness vector
        fitness = 0.0
        return [fitness]