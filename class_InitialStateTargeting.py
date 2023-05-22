import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from JupiterTrajectory_GlobalParameters import *
import class_AerocaptureSemianalyticalModel as ae_analytical
from handle_functions import *
from class_GalileanMoon import GalileanMoon

from tudatpy.kernel.astro import two_body_dynamics

# Load spice kernels
spice_interface.load_standard_kernels()


class InitialStateTargeting:

    def __init__(self,
                 atmosphere_entry_fpa: float,
                 atmosphere_entry_altitude: float,
                 B_parameter,
                 flyby_moon: str,
                 flyby_epoch: float,
                 jupiter_arrival_v_inf: float,
                 perturb_fpa: float = 0.,
                 perturb_entry_velocity_magnitude: float = 0.,
                 start_at_entry_interface: bool = False,
                 # start_at_exit_interface: bool = False,
                 verbose: bool = False
                 ):
        """
            Constructor class
        """
        if perturb_entry_velocity_magnitude != 0. and not start_at_entry_interface:
            raise Exception('Warning: you are calculating a wrong initial state!'
                          'Entry velocity magnitude perturation might not work well under these conditions.')


        self.atmosphere_entry_fpa = atmosphere_entry_fpa + perturb_fpa
        self.atmosphere_entry_altitude = atmosphere_entry_altitude
        self.B_parameter = B_parameter
        self.flyby_moon = flyby_moon
        if self.flyby_moon not in ['Io', 'Europa', 'Ganymede', 'Callisto']:
            self.flyby_moon = 'Ganymede'
            self.B_parameter = 1 / 2 * (galilean_moons_data[self.flyby_moon]['Radius'] +
                     galilean_moons_data[self.flyby_moon]['SOI_Radius'])
            warnings.warn(f'No moon selected. Using Ganymede as default. NOTE: the inserted B parameter: {B_parameter} [m] has no effect. Default value is {self.B_parameter} [m]')


        self.flyby_epoch = flyby_epoch
        self.jupiter_arrival_v_inf = jupiter_arrival_v_inf
        self.perturb_entry_velocity_magnitude = perturb_entry_velocity_magnitude
        self.start_at_entry_interface = start_at_entry_interface
        self.verbose = verbose

        self.initial_state = self.calculate_initial_state()
        if type(self.initial_state) == float or type(self.initial_state) == int:
            if self.initial_state == -1:
                self.trajectory_is_feasible = False
                self.escape_trajectory = False
            elif self.initial_state == -2:
                self.trajectory_is_feasible = True
                self.escape_trajectory = True
            else:
                self.simulation_start_epoch = self.get_simulation_start_epoch()
                self.trajectory_is_feasible = True
                self.escape_trajectory = False

        else:
            self.trajectory_is_feasible = True
            self.escape_trajectory = False

        self.entire_trajectory_cartesian_states = None
        self.final_orbit_cartesian_states = None
        self.trajectory_state_history = None
        self.trajectory_state_history_custom = None

    def calculate_initial_state(self):

        flyby_moon_state = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name=self.flyby_moon,
            observer_body_name="Jupiter",
            reference_frame_name=global_frame_orientation,
            aberration_corrections="NONE",
            ephemeris_time=self.flyby_epoch)
        moon_position = flyby_moon_state[0:3]
        moon_velocity = flyby_moon_state[3:6]
        moon_orbital_axis = unit_vector(np.cross(moon_position, moon_velocity))
        moon_eccentricity_vector = eccentricity_vector_from_cartesian_state(flyby_moon_state)
        moon_orbital_energy = orbital_energy(LA.norm(moon_position), LA.norm(moon_velocity), jupiter_gravitational_parameter)
        moon_sma = galilean_moons_data[self.flyby_moon]['SMA']
        moon_period = galilean_moons_data[self.flyby_moon]['Orbital_Period']
        moon_SOI = galilean_moons_data[self.flyby_moon]['SOI_Radius']
        moon_radius = galilean_moons_data[self.flyby_moon]['Radius']
        mu_moon = galilean_moons_data[self.flyby_moon]['mu']

        # epoch_interval = [flyby_epoch-moon_period/2, flyby_epoch+moon_period/2]
        orbital_axis = moon_orbital_axis

        arrival_fpa_deg = self.atmosphere_entry_fpa
        # Problem parameters
        first_arc_arrival_fpa = np.deg2rad(arrival_fpa_deg)  # rad
        first_arc_departure_radius = jupiter_SOI_radius
        first_arc_departure_velocity_norm = self.jupiter_arrival_v_inf

        # Calculate orbital energy
        first_arc_orbital_energy = orbital_energy(first_arc_departure_radius, first_arc_departure_velocity_norm,
                                                  jupiter_gravitational_parameter)

        # Calculate arrival radius and speed
        first_arc_arrival_radius = jupiter_radius + self.atmosphere_entry_altitude
        first_arc_arrival_velocity_norm = velocity_from_energy(first_arc_orbital_energy, first_arc_arrival_radius,
                                                               jupiter_gravitational_parameter) + self.perturb_entry_velocity_magnitude

        # Calculate angular momentum
        first_arc_angular_momentum_norm = angular_momentum(first_arc_arrival_radius, first_arc_arrival_velocity_norm, first_arc_arrival_fpa)

        # Calculate p, a, e
        first_arc_semilatus_rectum = first_arc_angular_momentum_norm ** 2 / jupiter_gravitational_parameter
        first_arc_semimajor_axis = - jupiter_gravitational_parameter / (2 * first_arc_orbital_energy)
        first_arc_eccentricity = np.sqrt(1 - first_arc_semilatus_rectum / first_arc_semimajor_axis)

        first_arc_departure_fpa = fpa_from_angular_momentum(first_arc_angular_momentum_norm, first_arc_departure_radius, first_arc_departure_velocity_norm,
                                                            is_fpa_positive=False)

        # ATMOSPHERIC ENTRY PROBLEM CLASS
        aerocapture_analytical_problem = ae_analytical.AerocaptureSemianalyticalModel([0., 0.], orbit_datapoints=200,
                                                                                      equations_order=2,
                                                                                      atmospheric_interface_altitude=atmospheric_entry_altitude)
        # Set the problem parameters and compute atmospheric entry
        orbital_parameters = [first_arc_departure_velocity_norm, first_arc_arrival_fpa]
        aerocapture_analytical_problem.fitness(orbital_parameters)

        self.aerocapture_dependent_variable_history = aerocapture_analytical_problem.get_dependent_variables_history()

        # Retrieve problem parameters
        aerocapture_problem_parameters = aerocapture_analytical_problem.aerocapture_parameters_function()
        self.aerocapture_problem_parameters = aerocapture_problem_parameters

        # Atmosphere exit fpa
        atmospheric_exit_fpa = aerocapture_problem_parameters[0]
        # Atmosphere exit velocity
        atmospheric_exit_velocity_norm = aerocapture_problem_parameters[1]
        # Final radius after aerocapture
        atmospheric_exit_radius = aerocapture_problem_parameters[4]
        # Phase angle of the atmospehric entry
        atmospheric_entry_final_phase_angle = aerocapture_problem_parameters[5]
        # Elapsed time of the atmospheric entry
        aerocapture_elapsed_time = aerocapture_problem_parameters[6]

        # Calculate second arc h and E
        second_arc_angular_momentum = angular_momentum(atmospheric_exit_radius, atmospheric_exit_velocity_norm,
                                                       atmospheric_exit_fpa)
        second_arc_orbital_energy = orbital_energy(atmospheric_exit_radius, atmospheric_exit_velocity_norm,
                                                   jupiter_gravitational_parameter)

        # Calculate p, a, e
        second_arc_semilatus_rectum = second_arc_angular_momentum ** 2 / jupiter_gravitational_parameter
        second_arc_semimajor_axis = - jupiter_gravitational_parameter / (2 * second_arc_orbital_energy)
        second_arc_eccentricity = np.sqrt(1 - second_arc_semilatus_rectum / second_arc_semimajor_axis)

        # Verify that post-ae orbit intersects moon's orbit
        second_arc_apocenter = second_arc_semimajor_axis * (1 + second_arc_eccentricity)
        if second_arc_apocenter < 0:
            return -2
        elif second_arc_apocenter < moon_sma + moon_SOI:
            return -1

        # Calculate arrival velocity mag at flyby
        second_arc_arrival_velocity_norm = velocity_from_energy(second_arc_orbital_energy, moon_sma,
                                                                jupiter_gravitational_parameter)

        # Calculate fpa at flyby, it's positive
        second_arc_arrival_fpa = fpa_from_angular_momentum(second_arc_angular_momentum,moon_sma,second_arc_arrival_velocity_norm,is_fpa_positive=True)

        # Calculate inertial velocity vector in Jupiter frame
        second_arc_arrival_velocity = velocity_vector_from_position(moon_position,orbital_axis,second_arc_arrival_fpa,second_arc_arrival_velocity_norm)

        # Calculate velocity vector and magnitude in moon's frame
        flyby_initial_velocity_vector = second_arc_arrival_velocity - moon_velocity
        flyby_v_inf_t = LA.norm(flyby_initial_velocity_vector)

        # Calculate flyby angles
        phi_2_angle = np.arccos(np.dot(unit_vector(-moon_velocity), unit_vector(flyby_initial_velocity_vector)))
        if np.dot(np.cross(-moon_velocity, flyby_initial_velocity_vector), orbital_axis) < 0:
            phi_2_angle = - phi_2_angle + 2 * np.pi

        delta_angle = phi_2_angle - (np.pi - np.arcsin(self.B_parameter / moon_SOI))
        # delta_angle = (phi_2_angle - np.arcsin(B_parameter_abs/moon_SOI)) WRONG WRONG WRONG WEONG
        if delta_angle < 0:
            delta_angle = delta_angle+2*np.pi

        # Calculate initial position of flyby in moon's frame
        flyby_initial_position = rotate_vector(unit_vector(-moon_velocity)*moon_SOI, orbital_axis, delta_angle)

        # Now the final position of the post-ae arc is defined
        second_arc_final_position = moon_position + flyby_initial_position

        flyby_alpha_angle = 2 * np.arcsin(1 / np.sqrt(1 + (self.B_parameter ** 2 * flyby_v_inf_t ** 4) / mu_moon ** 2))

        if self.B_parameter < 0:
            flyby_alpha_angle = - flyby_alpha_angle

        beta_angle = phi_2_angle + flyby_alpha_angle / 2 - np.pi / 2

        # position_rot_angle_old = 2 * (- delta_angle + beta_angle)

        flyby_pericenter = mu_moon / (flyby_v_inf_t ** 2) * (
                np.sqrt(1 + (self.B_parameter ** 2 * flyby_v_inf_t ** 4) / (mu_moon ** 2)) - 1)
        flyby_orbital_energy = orbital_energy(LA.norm(flyby_initial_position), flyby_v_inf_t, mu_parameter=mu_moon)
        flyby_sma = - mu_moon / (2 * flyby_orbital_energy)

        flyby_angular_momentum = self.B_parameter * flyby_v_inf_t
        flyby_semilatus_rectum = flyby_angular_momentum ** 2 / mu_moon
        flyby_eccentricity = np.sqrt(1 - flyby_semilatus_rectum / flyby_sma)

        position_rot_angle = 2 * true_anomaly_from_radius(LA.norm(flyby_initial_position), flyby_eccentricity,flyby_sma, True)

        if self.B_parameter < 0:
            position_rot_angle = - position_rot_angle

        flyby_axis = orbital_axis

        flyby_final_position = rotate_vector(flyby_initial_position, flyby_axis, position_rot_angle)
        flyby_final_velocity_vector = rotate_vector(flyby_initial_velocity_vector, flyby_axis, flyby_alpha_angle)

        # For debugging
        true_anomaly_boundary_total_theoretical_angle = np.arccos(-1/flyby_eccentricity)
        true_anomaly_doublecheck = true_anomaly_from_radius(LA.norm(flyby_initial_position), flyby_eccentricity,flyby_sma, True)

        # Calculate true anomaly space of the orbit, and derive the flyby elapsed time
        true_anomaly_boundary = true_anomaly_from_radius(LA.norm(flyby_initial_position), flyby_eccentricity, flyby_sma,True)
        true_anomaly_range = np.array([-true_anomaly_boundary, true_anomaly_boundary])
        flyby_elapsed_time = delta_t_from_delta_true_anomaly(true_anomaly_range,
                                                             eccentricity=flyby_eccentricity,
                                                             semi_major_axis=flyby_sma,
                                                             mu_parameter=mu_moon)
        self.flyby_elapsed_time = flyby_elapsed_time
        flyby_final_epoch = self.flyby_epoch + flyby_elapsed_time

        # Calculate the new moon state after the flyby
        moon_flyby_final_state = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name=self.flyby_moon,
            observer_body_name="Jupiter",
            reference_frame_name=global_frame_orientation,
            aberration_corrections="NONE",
            ephemeris_time=flyby_final_epoch)
        moon_final_position = moon_flyby_final_state[0:3]
        moon_final_velocity = moon_flyby_final_state[3:6]
        # moon_final_position = flyby_moon_state[0:3]
        # moon_final_velocity = flyby_moon_state[3:6]

        if self.B_parameter > moon_SOI-150000:
            ...  # just to stop here when debugging

        # Calculate starting conditions of post-flyby arc
        fourth_arc_departure_position = flyby_final_position + moon_final_position
        fourth_arc_departure_velocity = flyby_final_velocity_vector + moon_final_velocity
        fourth_arc_departure_velocity_norm = LA.norm(fourth_arc_departure_velocity)

        # Calculate post-flyby arc departure flight path angle
        fourth_arc_departure_fpa = fpa_from_cartesian_state(fourth_arc_departure_position,fourth_arc_departure_velocity)

        # Calculate post-flyby arc h and E
        fourth_arc_orbital_energy = orbital_energy(LA.norm(fourth_arc_departure_position), fourth_arc_departure_velocity_norm, jupiter_gravitational_parameter)
        fourth_arc_angular_momentum = angular_momentum(LA.norm(fourth_arc_departure_position), fourth_arc_departure_velocity_norm, fourth_arc_departure_fpa)

        # Calculate post-flyby arc p, a, e
        fourth_arc_semilatus_rectum = fourth_arc_angular_momentum ** 2 / jupiter_gravitational_parameter
        fourth_arc_semimajor_axis = - jupiter_gravitational_parameter / (2 * fourth_arc_orbital_energy)
        fourth_arc_eccentricity = np.sqrt(1 - fourth_arc_semilatus_rectum / fourth_arc_semimajor_axis)

        # Calculate post-flyby r_p
        fourth_arc_pericenter = fourth_arc_semimajor_axis * (1 - fourth_arc_eccentricity)

        fourth_arc_orbital_period = orbital_period(fourth_arc_semimajor_axis, jupiter_gravitational_parameter)

        # Calculate true anomaly spanned by the spacecraft in the first arc
        first_arc_departure_true_anomaly = true_anomaly_from_radius(first_arc_departure_radius,first_arc_eccentricity, first_arc_semimajor_axis, return_positive=False)
        first_arc_arrival_true_anomaly = true_anomaly_from_radius(first_arc_arrival_radius, first_arc_eccentricity,first_arc_semimajor_axis, return_positive=False)
        first_arc_delta_true_anomaly = first_arc_arrival_true_anomaly - first_arc_departure_true_anomaly

        first_arc_elapsed_time = delta_t_from_delta_true_anomaly(np.array([first_arc_departure_true_anomaly,first_arc_arrival_true_anomaly]), first_arc_eccentricity, first_arc_semimajor_axis, jupiter_gravitational_parameter)

        aerocapture_delta_phase_angle = atmospheric_entry_final_phase_angle

        # Calculate true anomaly spanned by the spacecraft in the second arc
        second_arc_departure_true_anomaly = true_anomaly_from_radius(first_arc_arrival_radius, second_arc_eccentricity,second_arc_semimajor_axis, return_positive=True)
        second_arc_arrival_true_anomaly = true_anomaly_from_radius(moon_sma, second_arc_eccentricity,second_arc_semimajor_axis, return_positive=True)
        second_arc_delta_true_anomaly = second_arc_arrival_true_anomaly - second_arc_departure_true_anomaly

        second_arc_elapsed_time = delta_t_from_delta_true_anomaly(np.array([second_arc_departure_true_anomaly,second_arc_arrival_true_anomaly]),second_arc_eccentricity,second_arc_semimajor_axis,jupiter_gravitational_parameter)

        # Calculate total true anomaly and elapsed time of the orbit up to the flyby
        delta_true_anomaly = first_arc_delta_true_anomaly + aerocapture_delta_phase_angle + second_arc_delta_true_anomaly
        total_elapsed_time = first_arc_elapsed_time + aerocapture_elapsed_time + second_arc_elapsed_time

        second_arc_initial_position = rotate_vector(atmospheric_exit_radius*unit_vector(second_arc_final_position), orbital_axis, -second_arc_delta_true_anomaly)
        second_arc_initial_velocity = velocity_vector_from_position(second_arc_initial_position,orbital_axis,atmospheric_exit_fpa,atmospheric_exit_velocity_norm)

        first_arc_final_position = rotate_vector(first_arc_arrival_radius*unit_vector(second_arc_initial_position), orbital_axis, -aerocapture_delta_phase_angle)
        first_arc_final_velocity = velocity_vector_from_position(first_arc_final_position, orbital_axis, first_arc_arrival_fpa, first_arc_arrival_velocity_norm)

        # Compute initial state of first arc
        first_arc_initial_position = rotate_vector(jupiter_SOI_radius*unit_vector(first_arc_final_position),orbital_axis, -first_arc_delta_true_anomaly)
        first_arc_initial_velocity = velocity_vector_from_position(first_arc_initial_position, orbital_axis, first_arc_departure_fpa, first_arc_departure_velocity_norm)

        # Variables for debugging
        flyby_pericenter_altitude = flyby_pericenter-moon_radius
        final_orbit_pericenter_altitude = fourth_arc_pericenter-jupiter_radius
        delta_v_mag = 2*flyby_v_inf_t*np.sin(flyby_alpha_angle/2)
        interpl_delta_v = LA.norm(fourth_arc_departure_velocity-second_arc_arrival_velocity)
        fpa_diff = fourth_arc_departure_fpa-second_arc_arrival_fpa

        self.calculated_orbit_parameters = [phi_2_angle, delta_angle, flyby_alpha_angle, beta_angle, flyby_pericenter_altitude,final_orbit_pericenter_altitude, fourth_arc_departure_velocity_norm, delta_v_mag, interpl_delta_v, fpa_diff, flyby_elapsed_time]

        # Prints for debugging
        if self.verbose:
            print(f'Impact parameter B of choice: {self.B_parameter/1e3:.3f} km')
            print(f'Phi 2 angle: {np.rad2deg(phi_2_angle):.3f} deg')
            print(f'Delta angle: {np.rad2deg(delta_angle):.3f} deg')
            print(f'Alpha angle: {np.rad2deg(flyby_alpha_angle):.3f} deg')
            print(f'Beta angle:  {np.rad2deg(beta_angle):.3f} deg')

            if self.B_parameter > 0:
                print('Orbit type: ccw')
            else:
                print('Orbit type: cw')

            print(f'Flyby pericenter altitude: {flyby_pericenter_altitude/1e3:.3f} km')
            print(f'Final orbit pericenter altitude: {final_orbit_pericenter_altitude/1e3:.3f} km')
            print(f'\n Pre-flyby energy: {second_arc_orbital_energy/1e3:.3f} kJ/kg      Post-flyby energy: {fourth_arc_orbital_energy/1e3:.3f} kJ/kg')


        # Build the initial state vector
        initial_state_vector = np.concatenate((first_arc_initial_position, first_arc_initial_velocity))

        sphere_of_influence_entry_start_epoch = self.flyby_epoch - total_elapsed_time

        self.arcs_dictionary = {
            'First': (
                jupiter_SOI_radius, first_arc_final_position, first_arc_eccentricity, first_arc_semimajor_axis,
                first_arc_arrival_fpa, first_arc_final_velocity, first_arc_orbital_energy),
            'Second': (
                atmospheric_exit_radius, second_arc_final_position, second_arc_eccentricity,
                second_arc_semimajor_axis,
                second_arc_arrival_fpa,second_arc_arrival_velocity, second_arc_orbital_energy),
            # 'Third': (
            # moon_sma, third_arc_final_position, third_arc_eccentricity, third_arc_semimajor_axis,
            # third_arc_arrival_fpa),
        }

        first_arc_start_epoch = sphere_of_influence_entry_start_epoch
        aerocapture_start_epoch = first_arc_start_epoch + first_arc_elapsed_time
        second_arc_start_epoch = aerocapture_start_epoch + aerocapture_elapsed_time
        flyby_start_epoch = second_arc_start_epoch + second_arc_elapsed_time
        final_orbit_start_epoch = flyby_start_epoch + flyby_elapsed_time

        self.arcs_time_information = np.asarray([(first_arc_start_epoch, first_arc_elapsed_time),
                                      (aerocapture_start_epoch, aerocapture_elapsed_time),
                                      (second_arc_start_epoch, second_arc_elapsed_time),
                                      (flyby_start_epoch, flyby_elapsed_time),
                                      (final_orbit_start_epoch, fourth_arc_orbital_period)])

        first_arc_initial_state = np.concatenate((first_arc_initial_position,first_arc_initial_velocity))
        aerocapture_initial_state = np.concatenate((first_arc_final_position,first_arc_final_velocity))
        second_arc_initial_state = np.concatenate((second_arc_initial_position,second_arc_initial_velocity))
        flyby_initial_state = np.concatenate((second_arc_final_position,second_arc_arrival_velocity))
        final_orbit_initial_state = np.concatenate((fourth_arc_departure_position,fourth_arc_departure_velocity))

        self.arcs_initial_states = np.array([first_arc_initial_state,
                                             aerocapture_initial_state,
                                             second_arc_initial_state,
                                             flyby_initial_state,
                                             final_orbit_initial_state])



        self.final_orbit_data = [fourth_arc_eccentricity, fourth_arc_semimajor_axis,fourth_arc_departure_velocity, fourth_arc_departure_position,fourth_arc_orbital_energy]


        aerocapture_start_epoch = self.flyby_epoch - second_arc_elapsed_time - aerocapture_elapsed_time
        self.aerocapture_state_history = aerocapture_analytical_problem.get_cartesian_state_history(
            first_arc_final_position, aerocapture_start_epoch, orbital_axis)

        self.simulation_start_epoch = sphere_of_influence_entry_start_epoch

        # Build the atmospheric entry interface state vector
        if self.start_at_entry_interface:
            initial_state_vector = np.concatenate((first_arc_final_position, first_arc_final_velocity))
            self.simulation_start_epoch = aerocapture_start_epoch

        # Print the state vector for debugging
        if self.verbose:
            print('\nDeparture state:')
            print(f'{list(initial_state_vector)}')
        return initial_state_vector

    def get_initial_state(self):
        return self.initial_state

    def get_simulation_start_epoch(self):
        return self.simulation_start_epoch

    def get_trajectory_cartesian_states(self):
        return self.entire_trajectory_cartesian_states, self.final_orbit_cartesian_states

    def get_trajectory_state_history(self):
        if self.trajectory_state_history is None:
            self.create_state_history()
        return self.trajectory_state_history

    def get_trajectory_state_history_from_epochs(self, epochs):
        if self.trajectory_state_history_custom is None:
            self.create_state_history_from_epochs(epochs)
        return self.trajectory_state_history_custom

    def get_aerocapture_dependent_variable_history(self, aerocapture_start_epoch: float = 0.):
        if type(aerocapture_start_epoch) not in [float, np.float64, int]:
            raise TypeError('wrong parameter inserted')
        if aerocapture_start_epoch == 0.:
            return self.aerocapture_dependent_variable_history

        elapsed_time_epochs = np.array(list(self.aerocapture_dependent_variable_history.keys()))
        dependent_variables = np.vstack(list(self.aerocapture_dependent_variable_history.values()))
        epochs = elapsed_time_epochs + aerocapture_start_epoch
        return dict(zip(epochs, dependent_variables))




    def create_state_history(self):
        flyby_moon_state = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name=self.flyby_moon,
            observer_body_name="Jupiter",
            reference_frame_name=global_frame_orientation,
            aberration_corrections="NONE",
            ephemeris_time=self.flyby_epoch)
        moon_position = flyby_moon_state[0:3]
        moon_velocity = flyby_moon_state[3:6]
        moon_orbital_axis = unit_vector(np.cross(moon_position, moon_velocity))
        moon_eccentricity_vector = eccentricity_vector_from_cartesian_state(flyby_moon_state)
        # moon_orbital_energy = orbital_energy(LA.norm(moon_position), LA.norm(moon_velocity), jupiter_gravitational_parameter)
        moon_sma = galilean_moons_data[self.flyby_moon]['SMA']
        moon_period = galilean_moons_data[self.flyby_moon]['Orbital_Period']
        moon_SOI = galilean_moons_data[self.flyby_moon]['SOI_Radius']
        moon_radius = galilean_moons_data[self.flyby_moon]['Radius']
        mu_moon = galilean_moons_data[self.flyby_moon]['mu']

        arcs_dictionary = self.arcs_dictionary
        # arcs_dictionary = {
        #     'First': (
        #         jupiter_SOI_radius, first_arc_final_position, eccentricity, semimajor_axis,
        #         arrival_fpa),
        #     'Second': (
        #         LA.norm(atmospheric_entry_final_position), second_arc_final_position, second_arc_eccentricity,
        #         second_arc_semimajor_axis,
        #         second_arc_arrival_fpa),
        # }
        number_of_epochs_to_plot = 200
        arc_number_of_points = number_of_epochs_to_plot
        arc_cartesian_state_history = {}
        total_state_history = {}

        arc_final_epoch = self.simulation_start_epoch # the first arc does not have a previous arc with a final epoch, so final_epoch is set as sim start epoch

        arcs_sequence = ['First', 'Aerocapture', 'Second']
        if self.start_at_entry_interface:
            arcs_sequence = ['Aerocapture', 'Second']
            arcs_dictionary.pop('First')

        for arc in arcs_sequence:
            if arc == 'First':
                initial_epoch = self.simulation_start_epoch # simulation start epoch
            elif arc == 'Second':
                aerocapture_elapsed_time = self.aerocapture_problem_parameters[6]
                initial_epoch = self.simulation_start_epoch + self.arcs_time_information[0,1] + aerocapture_elapsed_time # aerocapture exit epoch
            elif arc == 'Aerocapture':
                # COMPUTE AEROCAPTURE STATE HISTORY
                total_state_history.update(self.aerocapture_state_history)
                continue
            else:
                warnings.warn('no arcs wound with names First or Second')
                continue

            arc_departure_radius = arcs_dictionary[arc][0]
            arc_arrival_position = arcs_dictionary[arc][1]
            arc_arrival_radius = LA.norm(arc_arrival_position)

            arc_eccentricity = arcs_dictionary[arc][2]
            arc_semimajor_axis = arcs_dictionary[arc][3]
            arc_arrival_fpa = arcs_dictionary[arc][4]
            arc_arrival_velocity = arcs_dictionary[arc][5]
            arc_orbital_energy = arcs_dictionary[arc][6]

            arc_eccentricity_vector = eccentricity_vector_from_cartesian_state(np.concatenate((arc_arrival_position,arc_arrival_velocity)))

            orbital_axis = unit_vector(np.cross(arc_arrival_position, arc_arrival_velocity))
            line_of_nodes = unit_vector(np.cross(z_axis, orbital_axis))

            inclination = np.arccos(orbital_axis[2] / LA.norm(orbital_axis))

            # Find true anomalies at the first arc boundaries
            arc_arrival_true_anomaly = np.sign(arc_arrival_fpa) * true_anomaly_from_radius(arc_arrival_radius, arc_eccentricity, arc_semimajor_axis, True)
            arc_departure_true_anomaly = np.sign(arc_arrival_fpa) * true_anomaly_from_radius(arc_departure_radius, arc_eccentricity, arc_semimajor_axis, True)

            # Calculate phase angle of first arc
            arc_phase_angle = arc_arrival_true_anomaly - arc_departure_true_anomaly

            # Calculate coordinate points of the first arc to be plotted
            arc_true_anomaly_vector = np.linspace(arc_departure_true_anomaly, arc_arrival_true_anomaly,arc_number_of_points)
            radius_vector = radius_from_true_anomaly(arc_true_anomaly_vector, arc_eccentricity, arc_semimajor_axis)

            flight_path_angles = np.arctan(arc_eccentricity*np.sin(arc_true_anomaly_vector)/(1+arc_eccentricity*np.cos(arc_true_anomaly_vector)))
            velocity_magnitudes = velocity_from_energy(arc_orbital_energy,radius_vector,jupiter_gravitational_parameter)

            arc_rotated_position_states = rotate_vector(unit_vector(arc_eccentricity_vector), orbital_axis, arc_true_anomaly_vector) * radius_vector.reshape((len(radius_vector),1))
            arc_rotated_velocity_states = velocity_vector_from_position(arc_rotated_position_states, orbital_axis, flight_path_angles, velocity_magnitudes)

            epochs_vector = np.zeros(len(arc_true_anomaly_vector))
            for i in range(len(epochs_vector)):
                true_anomaly_interval_i = np.array([arc_true_anomaly_vector[0], arc_true_anomaly_vector[i]])
                delta_t = delta_t_from_delta_true_anomaly(true_anomaly_interval_i, arc_eccentricity, arc_semimajor_axis, jupiter_gravitational_parameter)
                epochs_vector[i] = initial_epoch + delta_t

            arc_cartesian_state_history[arc] = np.concatenate((arc_rotated_position_states, arc_rotated_velocity_states), axis=1)
            arc_state_history_temp = dict(zip(epochs_vector,arc_cartesian_state_history[arc]))
            total_state_history.update(arc_state_history_temp)
            arc_final_epoch = epochs_vector[-1]

        if arc != 'Second':
            raise Exception('wrong initial epoch for final orbit is getting calculated')
        final_orbit_initial_epoch = arc_final_epoch + self.flyby_elapsed_time

        # FINAL ORBIT #########################
        final_orbit_number_of_points = 2 * number_of_epochs_to_plot

        final_orbit_eccentricity = self.final_orbit_data[0]
        final_orbit_semimajor_axis = self.final_orbit_data[1]
        final_orbit_reference_velocity = self.final_orbit_data[2]
        final_orbit_reference_position = self.final_orbit_data[3]
        final_orbit_orbital_energy = self.final_orbit_data[4]

        # final_orbit_eccentricity_vector = eccentricity_vector_from_cartesian_state(np.concatenate((final_orbit_reference_position,final_orbit_reference_velocity)))

        final_orbit_orbital_axis = unit_vector(np.cross(final_orbit_reference_position, final_orbit_reference_velocity))
        final_orbit_line_of_nodes = unit_vector(np.cross(z_axis, final_orbit_orbital_axis))

        reference_cartesian_state = np.concatenate((final_orbit_reference_position,final_orbit_reference_velocity))
        final_orbit_eccentricity_vector = eccentricity_vector_from_cartesian_state(reference_cartesian_state)

        post_flyby_true_anomaly = true_anomaly_from_radius(LA.norm(final_orbit_reference_position),
                                                           final_orbit_eccentricity, final_orbit_semimajor_axis,
                                                           return_positive=True)

        if final_orbit_eccentricity < 1:
            final_orbit_true_anomaly_vector = np.linspace(post_flyby_true_anomaly, 2*np.pi + post_flyby_true_anomaly, final_orbit_number_of_points)
        else:
            soi_edge_true_anomaly = true_anomaly_from_radius(jupiter_SOI_radius, final_orbit_eccentricity,
                                                             final_orbit_semimajor_axis, return_positive=True)
            final_orbit_true_anomaly_vector = np.linspace(post_flyby_true_anomaly, soi_edge_true_anomaly, final_orbit_number_of_points)


        final_orbit_radius_vector = radius_from_true_anomaly(final_orbit_true_anomaly_vector, final_orbit_eccentricity,
                                                             final_orbit_semimajor_axis)

        final_orbit_flight_path_angles = np.arctan(final_orbit_eccentricity * np.sin(final_orbit_true_anomaly_vector) / (
                    1 + final_orbit_eccentricity * np.cos(final_orbit_true_anomaly_vector)))
        final_orbit_velocity_magnitudes = velocity_from_energy(final_orbit_orbital_energy, final_orbit_radius_vector, jupiter_gravitational_parameter)

        # Calculate the rotated position and velocity states of the final orbit
        final_orbit_rotated_position_states = rotate_vector(unit_vector(final_orbit_eccentricity_vector), final_orbit_orbital_axis, final_orbit_true_anomaly_vector) * final_orbit_radius_vector.reshape((len(final_orbit_radius_vector),1))
        final_orbit_rotated_velocity_states = velocity_vector_from_position(final_orbit_rotated_position_states, final_orbit_orbital_axis, final_orbit_flight_path_angles, final_orbit_velocity_magnitudes)

        final_orbit_cartesian_states = np.concatenate((final_orbit_rotated_position_states, final_orbit_rotated_velocity_states), axis=1)

        final_orbit_epochs_vector = np.zeros(len(final_orbit_true_anomaly_vector))
        for i in range(len(final_orbit_epochs_vector)):
            true_anomaly_range_i = np.array([final_orbit_true_anomaly_vector[0], final_orbit_true_anomaly_vector[i]])
            delta_t = delta_t_from_delta_true_anomaly(true_anomaly_range_i, final_orbit_eccentricity,
                                                      final_orbit_semimajor_axis, jupiter_gravitational_parameter)
            final_orbit_epochs_vector[i] = final_orbit_initial_epoch + delta_t

        final_orbit_state_history_temp = dict(zip(final_orbit_epochs_vector, final_orbit_cartesian_states))
        total_state_history.update(final_orbit_state_history_temp)

        arcs_cartesian_state_history_values = list(arc_cartesian_state_history.values())
        first_arc_cartesian_state_history = arcs_cartesian_state_history_values[0]
        second_arc_cartesian_state_history = arcs_cartesian_state_history_values[1]

        first_arc_final_position = first_arc_cartesian_state_history[-1, 0:3]

        aerocapture_cartesian_state_history = np.vstack(list(self.aerocapture_state_history.values()))

        self.trajectory_state_history = total_state_history # LACK AE STATE HISTORY
        self.entire_trajectory_cartesian_states = np.vstack((first_arc_cartesian_state_history, aerocapture_cartesian_state_history, second_arc_cartesian_state_history))
        self.final_orbit_cartesian_states = final_orbit_cartesian_states

    def create_state_history_from_epochs(self, epochs_vector: np.ndarray):
        if self.simulation_start_epoch != epochs_vector[0]:
            raise ValueError(f'Initial epoch differs from simulation start, by {abs(epochs_vector[0] - self.simulation_start_epoch)} s')

        flyby_moon = GalileanMoon(self.flyby_moon, self.flyby_epoch)
        flyby_moon_state = flyby_moon.cartesian_state
        moon_position, moon_velocity = flyby_moon_state[0:3], flyby_moon_state[3:6]

        arc_initial_states = self.arcs_initial_states
        arc_initial_epochs = self.arcs_time_information[:,0]
        arc_time_durations = self.arcs_time_information[:, 1]

        total_state_history = {}
        FIRST_ARC, AEROCAPTURE_ARC, SECOND_ARC, FLYBY_ARC, FINAL_ORBIT = 0, 1, 2, 3, 4

        # cartesian_states = np.zeros((len(epochs_vector),6))
        total_arc_epochs_list = []
        for arc_no, arc_initial_state in enumerate(arc_initial_states):
            mu_parameter = jupiter_gravitational_parameter
            if arc_no == FIRST_ARC and self.start_at_entry_interface:
                continue
            if arc_no == AEROCAPTURE_ARC:
                warnings.warn('might be wrong the aerocapture state history')
                total_state_history.update(self.aerocapture_state_history)
                continue
            if arc_no == FLYBY_ARC:
                flyby_position = arc_initial_state[0:3] - moon_position
                flyby_velocity = arc_initial_state[3:6] - moon_velocity
                arc_initial_state = np.concatenate((flyby_position, flyby_velocity))
                mu_parameter = flyby_moon.gravitational_parameter

            arc_initial_kepler_elements = element_conversion.cartesian_to_keplerian(arc_initial_state, mu_parameter)
            arc_initial_epoch = arc_initial_epochs[arc_no]
            arc_time_of_flight = arc_time_durations[arc_no]

            arc_epochs = epochs_vector[np.asarray(epochs_vector-arc_initial_epoch < arc_time_of_flight).nonzero()[0]]
            arc_epochs = arc_epochs[np.asarray(arc_epochs-arc_initial_epoch>0).nonzero()[0]]

            arc_cartesian_states = np.zeros((len(arc_epochs), 6))
            for epoch_index, arc_current_epoch in enumerate(arc_epochs):
                arc_elapsed_time = arc_current_epoch - arc_initial_epoch

                arc_new_kepler_elements = two_body_dynamics.propagate_kepler_orbit(arc_initial_kepler_elements, arc_elapsed_time, mu_parameter)
                arc_new_cartesian_state = element_conversion.keplerian_to_cartesian(arc_new_kepler_elements, mu_parameter)
                arc_cartesian_states[epoch_index, :] = arc_new_cartesian_state

            if arc_no == FLYBY_ARC:
                arc_cartesian_states[:,0:3] = arc_cartesian_states[:,0:3] + moon_position
                arc_cartesian_states[:,3:6] = arc_cartesian_states[:,3:6] + moon_velocity

            # arc_cartesian_states = np.vstack(arc_cartesian_states_list)
            arc_state_history = dict(zip(arc_epochs,arc_cartesian_states)) # fix what you do with verification_epochs
            total_state_history.update(arc_state_history)
            total_arc_epochs_list = total_arc_epochs_list + list(arc_epochs)

        total_arc_epochs = np.array(total_arc_epochs_list)
        if not np.asarray(total_arc_epochs == epochs_vector).all():
            warnings.warn('Not all verification_epochs have been used. Epochs vector extends further than the analytical orbit timespan. Extrapolation needed')
            # raise Exception('Not all verification_epochs have been assigned')

        self.trajectory_state_history_custom = total_state_history


    def plot_trajectory(self):
        # for debugging
        multiply_vector = 1e3
        show_vectors = True

        if self.entire_trajectory_cartesian_states is None or self.final_orbit_cartesian_states is None or self.trajectory_state_history is None:
            self.create_state_history()

        flyby_moon_state = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name=self.flyby_moon,
            observer_body_name="Jupiter",
            reference_frame_name=global_frame_orientation,
            aberration_corrections="NONE",
            ephemeris_time=self.flyby_epoch)
        moon_position = flyby_moon_state[0:3]
        moon_velocity = flyby_moon_state[3:6]
        # moon_orbital_axis = unit_vector(np.cross(moon_position, moon_velocity))
        # moon_eccentricity_vector = eccentricity_vector_from_cartesian_state(flyby_moon_state)
        # # moon_orbital_energy = orbital_energy(LA.norm(moon_position), LA.norm(moon_velocity), jupiter_gravitational_parameter)
        # moon_sma = galilean_moons_data[self.flyby_moon]['SMA']
        # moon_period = galilean_moons_data[self.flyby_moon]['Orbital_Period']
        # moon_SOI = galilean_moons_data[self.flyby_moon]['SOI_Radius']
        moon_radius = galilean_moons_data[self.flyby_moon]['Radius']
        # mu_moon = galilean_moons_data[self.flyby_moon]['mu']

        arcs_dictionary = self.arcs_dictionary

        # Plot 3-D Trajectory
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        trajectory_cartesian_states = np.vstack(list(self.trajectory_state_history.values()))
        trajectory_epochs = np.vstack(list(self.trajectory_state_history.keys()))

        x_plot, y_plot, z_plot = trajectory_cartesian_states[:,0], trajectory_cartesian_states[:,1], trajectory_cartesian_states[:,2]
        ax.plot3D(x_plot, y_plot, z_plot, 'gray')

        for arc in arcs_dictionary.keys():

            # arc_departure_radius = arcs_dictionary[arc][0]
            arc_arrival_position = arcs_dictionary[arc][1]
            # arc_arrival_radius = LA.norm(arc_arrival_position)
            # arc_eccentricity = arcs_dictionary[arc][2]
            # arc_semimajor_axis = arcs_dictionary[arc][3]
            # arc_arrival_fpa = arcs_dictionary[arc][4]
            arc_arrival_velocity = arcs_dictionary[arc][5]

            orbital_axis = unit_vector(np.cross(arc_arrival_position, arc_arrival_velocity))
            line_of_nodes = unit_vector(np.cross(z_axis,orbital_axis))

            if show_vectors:
                self.plot_vector(ax,np.zeros(3),line_of_nodes,1e9)

                self.plot_vector(ax, arc_arrival_position, arc_arrival_velocity, multiply_vector, color='g')

                flyby_initial_velocity = arc_arrival_velocity - moon_velocity
                self.plot_vector(ax, arc_arrival_position, flyby_initial_velocity, multiply_vector, color='r')

        # old: plots final orbit with a different color (red)
        # ax.plot3D(self.entire_trajectory_cartesian_states[:, 0], self.entire_trajectory_cartesian_states[:, 1], self.entire_trajectory_cartesian_states[:, 2], 'gray')

        # FINAL ORBIT ##########################################################################################################


        # final_orbit_eccentricity = self.final_orbit_data[0]
        # final_orbit_semimajor_axis = self.final_orbit_data[1]
        # # final_orbit_reference_position = moon_position
        final_orbit_reference_velocity = self.final_orbit_data[2]
        final_orbit_reference_position =  self.final_orbit_data[3]

        # final_orbit_orbital_axis = unit_vector(np.cross(final_orbit_reference_position, final_orbit_reference_velocity))
        # final_orbit_line_of_nodes = unit_vector(np.cross(z_axis, final_orbit_orbital_axis))

        # Old: plots final orbit with a different color (red)
        # ax.plot3D(self.final_orbit_cartesian_states[:,0], self.final_orbit_cartesian_states[:,1], self.final_orbit_cartesian_states[:,2], 'r')

        if show_vectors:
            ax.plot3D([0, final_orbit_reference_position[0]], [0, final_orbit_reference_position[1]],
                      [0, final_orbit_reference_position[2]], 'k')
            self.plot_vector(ax,final_orbit_reference_position,final_orbit_reference_velocity,multiply_vector, 'g')

            flyby_final_velocity = final_orbit_reference_velocity-moon_velocity
            self.plot_vector(ax, final_orbit_reference_position, flyby_final_velocity, multiply_vector, 'r')

        # PLOT FIGURE ARRANGEMENT ##############################################################################################

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        ax.set_title('Jupiter full arrival trajectory')

        lines = ax.get_lines()
        last_line = lines[-1]

        # Get the data for the last line
        line_data = last_line.get_data_3d()

        # Get the data limits for the last line
        x_data_limits = line_data[0].min(), line_data[0].max()
        y_data_limits = line_data[1].min(), line_data[1].max()
        z_data_limits = line_data[2].min(), line_data[2].max()

        xyzlim = np.array([x_data_limits, y_data_limits, z_data_limits]).T
        # xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
        XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
        ax.set_xlim3d(XYZlim)
        ax.set_ylim3d(XYZlim)
        ax.set_zlim3d(XYZlim * 0.75)
        ax.set_aspect('auto')

        # draw jupiter
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = jupiter_radius * np.cos(u) * np.sin(v)
        y = jupiter_radius * np.sin(u) * np.sin(v)
        z = jupiter_radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color="saddlebrown")

        # draw moon
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x_0 = flyby_moon_state[0]
        y_0 = flyby_moon_state[1]
        z_0 = flyby_moon_state[2]
        x = x_0 + moon_radius * np.cos(u) * np.sin(v)
        y = y_0 + moon_radius * np.sin(u) * np.sin(v)
        z = z_0 + moon_radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color="b")

        if show_vectors:
            self.plot_vector(ax,moon_position,moon_velocity,multiply_vector)

        ########################################################################################################################
        # RE-ENTRY PLOTS  ######################################################################################################
        ########################################################################################################################

        # fpa_vector = np.linspace(arrival_fpa, atmospheric_exit_fpa, 200)
        #
        # altitude_vector = ae_radii - jupiter_radius
        # # atmospheric_entry_trajectory_altitude(fpa_vector, atmospheric_entry_fpa, density_at_atmosphere_entry,
        # #                                                     reference_density, ballistic_coefficient_times_g_acc,
        # #                                                     atmospheric_entry_g_acc, jupiter_beta_parameter)
        # downrange_vector = tau_linspace * np.sqrt(
        #     jupiter_scale_height / (atmospheric_entry_altitude + jupiter_radius)) * jupiter_radius
        # # atmospheric_entry_trajectory_distance_travelled(fpa_vector, atmospheric_entry_fpa, effective_entry_fpa, scale_height)
        #
        # fig2, ax2 = plt.subplots(figsize=(5, 6))
        # ax2.plot(downrange_vector / 1e3, altitude_vector / 1e3)
        # ax2.set(xlabel='downrange [km]', ylabel='altitude [km]')

        # ax2.plot(fpa_vector,downrange_vector)

        # ax2.set_aspect('equal', 'box')

        plt.show()

    def plot_vector(self,axes, origin, vector, multiply_vector = 1., color='b'):
        axes.plot3D([origin[0],vector[0]*multiply_vector+origin[0]], [origin[1],vector[1]*multiply_vector+origin[1]], [origin[2],vector[2]*multiply_vector+origin[2]],color)

