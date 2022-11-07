import numpy as np

# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators


def get_propagator_settings(bodies,
                            termination_settings,
                            dependent_variables_to_save,
                            current_propagator = propagation_setup.propagator.cowell,
                            initial_state_perturbation = np.zeros( 6 ) ):

    # Define bodies that are propagated and their central bodies of propagation
    bodies_to_propagate = ['Capsule']
    central_bodies = ['Jupiter']

    # Define accelerations for the nominal case
    acceleration_settings_on_vehicle = {'Jupiter': [propagation_setup.acceleration.point_mass_gravity()]}

    # Create global accelerations' dictionary
    acceleration_settings = {'Capsule': acceleration_settings_on_vehicle}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    # Retrieve initial state (set as a hohmann arrival from the Jupiter SoI)
    initial_state = np.array([-46846525005.13016, 11342093939.996948, 0.0, 5421.4069183825695, -1402.9779133377706, 0.0])

    # Create propagation settings for the benchmark
    propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                     acceleration_models,
                                                                     bodies_to_propagate,
                                                                     initial_state,
                                                                     termination_settings,
                                                                     current_propagator,
                                                                     output_variables=dependent_variables_to_save)
    return propagator_settings


# Load spice kernels
spice_interface.load_standard_kernels()

simulation_start_epoch = 11000 # random initial date
simulation_duration = 200 * constants.JULIAN_DAY

# Define settings for celestial bodies
bodies_to_create = ['Jupiter']
# Define coordinate system
global_frame_origin = 'Jupiter'
global_frame_orientation = 'ECLIPJ2000'

# Create body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Create vehicle object
bodies.create_empty_body('Capsule')

# # Set mass of vehicle
# bodies.get_body('Capsule').mass = 2000  # kg


########################################################################################################################
# ERROR TRIGGER ########################################################################################################
########################################################################################################################

termination_settings = propagation_setup.propagator.time_termination(
        simulation_start_epoch + simulation_duration,
        terminate_exactly_on_final_condition=False)
dependent_variables_to_save = []
current_propagator = propagation_setup.propagator.unified_state_model_quaternions

current_propagator_settings = get_propagator_settings(bodies,
                                                      termination_settings,
                                                      dependent_variables_to_save,
                                                      current_propagator)

propagator_settings_2 = current_propagator_settings

print('The two vectors should be identical:')
print(propagator_settings_2.initial_states)

current_propagator_settings.initial_states = np.zeros(6)

print(propagator_settings_2.initial_states)


