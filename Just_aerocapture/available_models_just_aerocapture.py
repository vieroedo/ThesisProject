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


class AerocaptureNumericalProblem:

    def __init__(self,
                 bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                 integrator_settings: tudatpy.kernel.numerical_simulation.propagation_setup.integrator.IntegratorSettings,
                 termination_settings,
                 # capsule_density: float,
                 simulation_start_epoch: float,
                 decision_variable_range
                 ):
        """
                Constructor for the AerocaptureNumericalProblem class.
                Parameters
                ----------
                bodies : tudatpy.kernel.simulation.environment_setup.SystemOfBodies
                    System of bodies present in the simulation.
                integrator_settings : tudatpy.kernel.simulation.propagation_setup.integrator.IntegratorSettings
                    Integrator settings to be provided to the dynamics simulator.
                propagator_settings : tudatpy.kernel.simulation.propagation_setup.propagator.MultiTypePropagatorSettings
                    Propagator settings object.
                capsule_density : float
                    Constant density of the vehicle.
                Returns
                -------
                none
                """

        # Set arguments as attributes
        self.bodies_function = lambda: bodies
        self.integrator_settings_function = lambda: integrator_settings
        self.termination_settings_function = lambda: termination_settings
        # self.capsule_density = capsule_density
        self.simulation_start_epoch = simulation_start_epoch
        self.decision_variable_range = decision_variable_range


    def fitness(self,
                orbital_parameters):
        """
        Propagates the trajectory with the orbital parameters given as argument.
        This function uses the orbital parameters to set a new aerodynamic coefficient interface, subsequently propagating
        the trajectory. The fitness, currently set to zero, can be computed here: it will be used during the
        optimization process.
        Parameters
        ----------
        orbital_parameters : list of floats
            List of orbital parameters to be optimized.
        Returns
        -------
        fitness : float
            Fitness value (for optimization, see assignment 3).
        """
        bodies = self.bodies_function()
        integrator_settings = self.integrator_settings_function()

        # Delete existing capsule
        # bodies.remove_body('Capsule')
        # Create new capsule with a new coefficient interface based on the current parameters, add it to the body system
        # aerodynamic_analysis = Util.add_capsule_to_body_system(bodies,
        #                            shape_parameters,
        #                            self.capsule_density)

        atm_entry_fpa = orbital_parameters[0]
        atm_entry = orbital_parameters[1]

        # Create propagator settings for benchmark (Cowell)
        dependent_variables_to_save = Util.get_dependent_variable_save_settings()
        termination_settings = self.termination_settings_function( )
        propagator_settings = Util.get_propagator_settings(atm_entry_fpa,
                                                           atm_entry_alt,
                                                           bodies,
                                                           termination_settings,
                                                           dependent_variables_to_save,
                                                           jupiter_interpl_excees_vel=)

        # Create simulation object and propagate dynamics
        dynamics_simulator = numerical_simulation.SingleArcSimulator(
            bodies,
            integrator_settings,
            propagator_settings,
            print_dependent_variable_data = False)

        self.dynamics_simulator_function = lambda: dynamics_simulator

        # Add the objective and constraint values into the fitness vector
        fitness = 0.0
        return [fitness]