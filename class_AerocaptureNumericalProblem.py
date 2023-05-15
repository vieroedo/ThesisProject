import numpy as np
import random
from JupiterTrajectory_GlobalParameters import *
from handle_functions import *
import CapsuleEntryUtilities as Util

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel import numerical_simulation

import class_InitialStateTargeting as ist


class AerocaptureNumericalProblem:

    def __init__(self,
                 decision_variable_range,
                 environment_model: int = 0,
                 integrator_settings_index: int = -4,
                 # post_ae_flyby_moon: str = '',
                 do_flyby: bool = False,
                 fly_galileo: bool = False,
                 arc_to_compute: int = -1,
                 initial_state_perturbation=np.zeros(6),
                 entry_parameters_perturbation=np.zeros(2)
                 ):
        """
                Constructor for the AerocaptureNumericalProblem class.
                Parameters
                ----------
                simulation_start_epoch: float
                    start epoch of the simulation
                decision_variable_range
                    define the boundaries of the decision variables (for optimization)
                environment_model: int
                    choose the environment model to be used
                integrator_settings_index: int
                    choose the preferred integrator settings
                fly_galileo: bool
                    choose whether to fly the galileo mission or do aerocapture
                initial_state_perturbation:
                    perturbation of the initial state of the simulation
                entry_parameters_perturbation:
                    perturbation of the atmospheric entry parameters. Can be used only if the propagation starts at
                    atmospheric interface. Entry parameters: [entry_velocity_norm, entry_fpa]
                Returns
                -------
                none
                """

        # Set arguments as attributes
        self.flyby_occurs = do_flyby
        self.decision_variable_range = decision_variable_range
        self.galileo_mission_case = fly_galileo
        self.arc_to_compute = arc_to_compute
        self.environment_model = environment_model
        self.integrator_settings_index = integrator_settings_index
        self.initial_state_perturbation = initial_state_perturbation
        self.entry_parameters_perturbation = entry_parameters_perturbation

        if self.arc_to_compute == 0:
            self._stop_before_aerocapture = True
            self._start_at_entry_interface = False
            self._stop_after_aerocapture = False
            # self._start_at_exit_interface = False
        elif self.arc_to_compute == 1:
            self._stop_before_aerocapture = False
            self._start_at_entry_interface = True
            self._stop_after_aerocapture = True
            # self._start_at_exit_interface = False
        elif self.arc_to_compute == 12:
            self._stop_before_aerocapture = False
            self._start_at_entry_interface = True
            self._stop_after_aerocapture = False
            # self._start_at_exit_interface = False
        # elif self.arc_to_compute == 2:
        #     self._stop_before_aerocapture = False
        #     self._start_at_entry_interface = False
        #     self._stop_after_aerocapture = False
        #     self._start_at_exit_interface = True
        elif self.arc_to_compute == -1:
            self._stop_before_aerocapture = False
            self._start_at_entry_interface = False
            self._stop_after_aerocapture = False
            # self._start_at_exit_interface = False
        else:
            raise Exception(f'Wrong parameter inserted for the arc to compute ({self.arc_to_compute}). '
                            f'Allowed values are -1, 0, 1, 12, and 2. (Default: -1)')

        if self.arc_to_compute != -1 and self.galileo_mission_case:
            raise Exception('Incompatible combination of arguments. '
                            'Cannot compute just a part of the Galileo trajectory. '
                            'Leave the argument arc_to_compute unexpressed, or set it to the default value. '
                            '(Default: -1)')

        if entry_parameters_perturbation.any() and not self.start_at_entry_interface:
            raise Exception('Entry parameters cannot be perturbed when the propagation does not start at atmospheric entry.'
                            'Set the perturbation of the entry parameters to 0 or the propagation to start at the entry interface.')


        body_settings = self.create_body_settings_environment()
        bodies = self.create_bodies_environment(body_settings)
        bodies = self.add_aerodynamic_interface(bodies)
        self.bodies_function = lambda: bodies

        self.trajectory_is_feasible = None
        # integrator_settings = self.create_integrator_settings()
        # self.integrator_settings_function = lambda: integrator_settings
        #
        # termination_settings = self.create_termination_settings()
        # self.termination_settings_function = lambda: termination_settings

        # self.dynamics_simulator_function = lambda: None

    @property
    def stop_before_aerocapture(self):
        return self._stop_before_aerocapture

    @property
    def start_at_entry_interface(self):
        return self._start_at_entry_interface

    @property
    def stop_after_aerocapture(self):
        return self._stop_after_aerocapture

    # @property
    # def start_at_exit_interface(self):
    #     return self._start_at_exit_interface

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

        if self.dynamics_simulator_function() is None:
            raise Exception('Unable to retrieve state history. Trajectory not yet propagated.')

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

        if self.dynamics_simulator_function() is None:
            raise Exception('Unable to retrieve state history. Trajectory not yet propagated.')

        return self.dynamics_simulator_function().get_equations_of_motion_numerical_solution_raw()

    def get_bounds(self):

        return self.decision_variable_range

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

        if self.dynamics_simulator_function() is None:
            raise Exception('Unable to retrieve dependent variable history. Trajectory not yet propagated.')
        return self.dynamics_simulator_function( ).dependent_variable_history

    def get_last_run_dynamics_simulator(self):
        """
        Returns the dynamics simulator object.
        Parameters
        ----------
        none
        Returns
        -------
        tudatpy.kernel.simulation.propagation_setup.SingleArcDynamicsSimulator
        """

        if self.dynamics_simulator_function() is None:
            raise Exception('Unable to retrieve the dynamics simulator. Trajectory not yet propagated.')
        return self.dynamics_simulator_function( )

    def get_bodies(self):
        return self.bodies_function()

    # def get_integrator_settings(self):
    #     integrator_settings = self.integrator_settings_function
    #     return integrator_settings
    #
    # def get_termination_settings(self):
    #     termination_settings = self.termination_settings_function
    #     return termination_settings

    def get_initial_state_perturbation(self):
        return self.initial_state_perturbation

    # def set_termination_settings(self, termination_settings):
    #     self.termination_settings_function = lambda: termination_settings

    def get_simulation_start_epoch(self):
        return self.simulation_start_epoch

    def get_initial_state(self):
        return self.initial_state

    def set_environment_model(self, choose_model, verbose = True):
        self.environment_model = choose_model
        if verbose:
            print('Run the \'fitness\' function for the changes to be effective in the environment model')

    def set_initial_state_perturbation(self, initial_state_perturbation):
        self.initial_state_perturbation = initial_state_perturbation

    def set_entry_parameters_perturbation(self, entry_parameters_perturbation):
        self.entry_parameters_perturbation = entry_parameters_perturbation

    def set_bodies(self, bodies):
        self.bodies_function = lambda: bodies

    def create_body_settings_environment(self):
        if self.flyby_occurs:
            bodies_to_create = ['Jupiter', 'Sun',
                                'Io', 'Europa', 'Ganymede', 'Callisto']
        else:
            bodies_to_create = ['Jupiter', 'Sun']
        global_frame_origin = 'Jupiter'
        global_frame_orientation = 'ECLIPJ2000'

        # Create body settings
        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create,
            global_frame_origin,
            global_frame_orientation)

        # Add Jupiter exponential atmosphere
        density_scale_height = Util.jupiter_scale_height
        density_at_zero_altitude = Util.jupiter_1bar_density
        g_0 = Util.jupiter_gravitational_parameter / Util.jupiter_radius ** 2
        constant_temperature = density_scale_height * g_0 / Util.jupiter_gas_constant

        if self.environment_model == 4 or self.environment_model == 0:
            body_settings.get('Jupiter').atmosphere_settings = environment_setup.atmosphere.custom_constant_temperature(
                Util.jupiter_atmosphere_density_model,
                constant_temperature,
                Util.jupiter_gas_constant,
                Util.jupiter_specific_heats_ratio)
        else:
            body_settings.get('Jupiter').atmosphere_settings = environment_setup.atmosphere.exponential(
                density_scale_height, density_at_zero_altitude)

        # Maybe add it, yes, but later, cs now jupiter's already rotating
        # target_frame = 'IAU_Jupiter_Simplified'
        # target_frame_spice = "IAU_Jupiter"
        # body_settings.get('Jupiter').rotation_model_settings = environment_setup.rotation_model.simple_from_spice(global_frame_orientation, target_frame, target_frame_spice,simulation_start_epoch)

        return body_settings

    def create_bodies_environment(self, body_settings):
        # Create bodies_env
        bodies_env = environment_setup.create_system_of_bodies(body_settings)
        # Create vehicle object
        bodies_env.create_empty_body('Capsule')
        # Set mass of vehicle
        if self.galileo_mission_case:
            bodies_env.get_body('Capsule').mass = Util.galileo_mass  # kg
        else:
            bodies_env.get_body('Capsule').mass = Util.vehicle_mass  # kg
        return bodies_env

    def add_aerodynamic_interface(self, bodies, coefficients_variation: np.ndarray = None):
        # Create aerodynamic coefficients interface (drag and lift only)
        if self.galileo_mission_case:
            reference_area = Util.galileo_ref_area  # m^2
            drag_coefficient = Util.galileo_cd
            lift_coefficient = Util.galileo_cl
        else:
            reference_area = Util.vehicle_reference_area  # m^2
            drag_coefficient = Util.vehicle_cd
            lift_coefficient = Util.vehicle_cl
        aero_coefficients = np.array([drag_coefficient, lift_coefficient])

        # For uncertainty propagation
        if coefficients_variation is not None:
            if len(coefficients_variation) == 2:
                # coeff_variation = C_D, C_L
                coeff_variation = coefficients_variation
            else:
                raise Exception('Wrong coefficients uncertainty given to the function')
            aero_coefficients = aero_coefficients * coeff_variation

        drag_coefficient = aero_coefficients[0]
        lift_coefficient = aero_coefficients[1]

        aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
            reference_area, [drag_coefficient, 0.0, lift_coefficient])  # [Drag, Side-force, Lift]
        environment_setup.add_aerodynamic_coefficient_interface(
            bodies, 'Capsule', aero_coefficient_settings)

        return bodies

    def create_integrator_settings(self, simulation_start_epoch):
        integrator_settings = Util.get_integrator_settings(self.integrator_settings_index,
                                                           simulation_start_epoch,
                                                           galileo_integration_settings=self.galileo_mission_case,
                                                           galileo_step_size=0.1)

        return integrator_settings

    def create_propagator_settings(self, initial_state,
                                   termination_settings,
                                   moon_of_flyby=''
                                   # entry_fpa,
                                   # interplanetary_arrival_velocity,
                                   # orbit_axis,
                                   # initial_state_perturbation = np.zeros(6),
                                   # entry_parameters_perturbation = np.zeros(2)
                                   ):
        if moon_of_flyby == '' and self.flyby_occurs:
            raise Exception('The current instance is set to have a flyby. please select a moon.')
        if moon_of_flyby != '' and not self.flyby_occurs:
            raise Exception('Aerocapture problem instancce declared to have no flyby. You can\'t select a moon')

        dependent_variables_to_save = Util.get_dependent_variable_save_settings(self.bodies_function())
        # propagator_settings = Util.get_propagator_settings(entry_fpa, atm_entry_alt, self.bodies_function(),
        #                                                    self.termination_settings_function(),
        #                                                    dependent_variables_to_save,
        #                                                    galileo_propagator_settings=self.galileo_mission_case,
        #                                                    model_choice=self.environment_model)
        #                                                    # start_at_exit_interface=self._start_at_exit_interface)
        propagator_settings = Util.get_propagator_settings(initial_state, self.bodies_function(), termination_settings,
                                                           dependent_variables_to_save,
                                                           moon_of_flyby=moon_of_flyby,
                                                           galileo_propagator_settings=self.galileo_mission_case,
                                                           model_choice=self.environment_model)
        return propagator_settings

    def create_termination_settings(self, simulation_start_epoch, entry_fpa_deg=0., bodies=None):
        termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                             galileo_termination_settings=self.galileo_mission_case,
                                                             stop_before_aerocapture=self._stop_before_aerocapture,
                                                             stop_after_aerocapture=self._stop_after_aerocapture,
                                                             entry_fpa_deg=entry_fpa_deg, bodies=bodies)
        return termination_settings

    def get_entry_heat_fluxes(self, return_history_dictionary = False):
        dependent_variable_history = self.get_last_run_dependent_variable_history()
        dependent_variable_values = np.vstack(list(dependent_variable_history.values()))
        dependent_variable_epochs = np.array(list(dependent_variable_history.keys()))
        airspeed = dependent_variable_values[:,6]
        density = dependent_variable_values[:,8]
        nose_radius = Util.vehicle_radius
        convective_hfx, radiative_hfx = Util.calculate_trajectory_heat_fluxes(density, airspeed, nose_radius)
        if return_history_dictionary:
            heat_fluxes_dict = Util.calculate_trajectory_heat_fluxes_history(dependent_variable_epochs, convective_hfx, radiative_hfx)
            return heat_fluxes_dict

        return convective_hfx, radiative_hfx

    def get_peak_heat_flux_and_heat_load(self):
        convective, radiative = self.get_entry_heat_fluxes()
        total= convective+radiative
        dependent_variable_history = self.get_last_run_dependent_variable_history()
        dependent_variable_epochs = np.array(list(dependent_variable_history.keys()))
        peak_hfx, heat_load = Util.calculate_peak_hfx_and_heat_load(dependent_variable_epochs, total)
        return peak_hfx, heat_load

    def get_tps_mass_fraction(self):
        peak, heat_load = self.get_peak_heat_flux_and_heat_load()
        return Util.calculate_tps_mass_fraction(heat_load)

    def fitness(self,
                orbital_parameters,
                ):
        """
        Propagates the trajectory with the orbital parameters given as argument.
        This function uses the orbital parameters to set a new aerodynamic coefficient interface, subsequently propagating
        the trajectory. The fitness, currently set to zero, can be computed here: it will be used during the
        optimization process.
        Parameters
        ----------
        orbital_parameters : list of floats
            List of orbital parameters to be optimized:
                0: interplanetary arrival velocity
                1: atmospheric entry flight path angle
                2: moon of flyby [0: None, 1: 'Io', 2: 'Europa', 3: 'Ganymede, 4: 'Callisto']
                3: epoch of moon flyby
                4: flyby impact parameter B
        Returns
        -------
        fitness : float
            Fitness value, for optimization.
        """
        bodies = self.bodies_function()

        # Delete existing capsule
        # bodies.remove_body('Capsule')
        # Create new capsule with a new coefficient interface based on the current parameters, add it to the body system
        # aerodynamic_analysis = Util.add_capsule_to_body_system(bodies,
        #                            shape_parameters,
        #                            self.capsule_density)
        orbital_parameters_length = len(orbital_parameters)
        # moons_dict = {1: 'Io', 2: 'Europa', 3: 'Ganymede', 4: 'Callisto'}

        interplanetary_arrival_velocity = orbital_parameters[0]  # m/s
        atm_entry_fpa_deg = orbital_parameters[1]  # deg

        if orbital_parameters_length > 3:
            flyby_moon_no = orbital_parameters[2]
        elif orbital_parameters_length == 3:
            flyby_moon_no = 0
        else:
            raise Exception('Wrong number of orbital parameters inserted.')

        if 1 <= flyby_moon_no <= 4:
            if not self.flyby_occurs:
                raise Exception('Current instance is set to have no flyby. change settings')
            flyby_epoch = orbital_parameters[3]
            flyby_moon = moons_optimization_parameter_dict[flyby_moon_no]
            B_parameter = orbital_parameters[4]

            flyby_moon_state = spice_interface.get_body_cartesian_state_at_epoch(
                target_body_name=flyby_moon,
                observer_body_name="Jupiter",
                reference_frame_name=global_frame_orientation,
                aberration_corrections="NONE",
                ephemeris_time=flyby_epoch)
            moon_position = flyby_moon_state[0:3]
            moon_velocity = flyby_moon_state[3:6]
            moon_orbital_axis = unit_vector(np.cross(moon_position,moon_velocity))
            moon_eccentricity_vector = eccentricity_vector_from_cartesian_state(flyby_moon_state)


            initial_state_targeting_problem = ist.InitialStateTargeting(atmosphere_entry_fpa=atm_entry_fpa_deg,
                                                                        atmosphere_entry_altitude=atmospheric_entry_altitude,
                                                                        B_parameter=B_parameter, flyby_moon=flyby_moon,
                                                                        flyby_epoch=flyby_epoch,
                                                                        jupiter_arrival_v_inf=interplanetary_arrival_velocity,
                                                                        perturb_fpa=self.entry_parameters_perturbation[1],
                                                                        perturb_entry_velocity_magnitude=self.entry_parameters_perturbation[0],
                                                                        start_at_entry_interface=self.start_at_entry_interface)

            if initial_state_targeting_problem.trajectory_is_feasible == False:
                self.trajectory_is_feasible = False
            else:
                self.trajectory_is_feasible = True

            simulation_start_epoch = initial_state_targeting_problem.get_simulation_start_epoch()
            initial_state = initial_state_targeting_problem.get_initial_state()

            self.simulation_start_epoch = simulation_start_epoch
            self.initial_state = initial_state

        elif flyby_moon_no == 0:
            if self.flyby_occurs:
                raise Exception('Current instance is set to have a post-aerocapture flyby. change settings')
            flyby_moon = ''
            orbital_axis = z_axis
            simulation_start_epoch = orbital_parameters[2]  # it's correct cs the third one is time when no flyby happens
            initial_state = Util.get_initial_state(atm_entry_fpa_deg, atmospheric_entry_altitude,
                                                   orbital_axis, interplanetary_arrival_velocity,
                                                   self.entry_parameters_perturbation[1],
                                                   self.entry_parameters_perturbation[0],
                                                   self.start_at_entry_interface) + self.initial_state_perturbation
            self.simulation_start_epoch = simulation_start_epoch
            self.initial_state = initial_state
        else:
            raise Exception('Wrong value for flyby_moon')

        integrator_settings = self.create_integrator_settings(simulation_start_epoch)
        # if self._stop_before_aerocapture or self._stop_after_aerocapture:
        termination_settings = self.create_termination_settings(simulation_start_epoch, atm_entry_fpa_deg, bodies)

        # Create propagator settings
        propagator_settings = self.create_propagator_settings(initial_state, termination_settings,
                                                              moon_of_flyby=flyby_moon)

        # Create simulation object and propagate dynamics
        dynamics_simulator = numerical_simulation.SingleArcSimulator(
            bodies,
            integrator_settings,
            propagator_settings,
            print_dependent_variable_data = False)

        self.dynamics_simulator_function = lambda: dynamics_simulator

        # if self._stop_after_aerocapture:
        #     state_history_final_state = list(dynamics_simulator.state_history.values())[-1]
        #     self.entry_interface_nominal_state = state_history_final_state

        # Add the objective and constraint values into the fitness vector
        fitness = 0.0
        return [fitness]
