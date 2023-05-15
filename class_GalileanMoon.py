from JupiterTrajectory_GlobalParameters import *

from handle_functions import unit_vector
from handle_functions import eccentricity_vector_from_cartesian_state
from handle_functions import orbital_energy


class GalileanMoon:

    def __init__(self, moon_name, epoch=0.):
        if moon_name not in ['Io', 'Europa', 'Ganymede', 'Callisto']:
            raise ValueError('Unexpected moon name. Allowed values are Io, Europa, Ganymede, Callisto')

        self._moon = moon_name
        self._semimajor_axis = galilean_moons_data[moon_name]['SMA']
        self._orbital_period = galilean_moons_data[moon_name]['Orbital_Period']
        self._SOI_radius = galilean_moons_data[moon_name]['SOI_Radius']
        self._radius = galilean_moons_data[moon_name]['Radius']
        self._gravitational_parameter = galilean_moons_data[moon_name]['mu']

        self.epoch = epoch
        self._cartesian_state = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name=self._moon,
            observer_body_name="Jupiter",
            reference_frame_name=global_frame_orientation,
            aberration_corrections="NONE",
            ephemeris_time=self.epoch)

    @property
    def cartesian_state(self):
        return self._cartesian_state

    @property
    def orbital_axis(self) -> np.ndarray:
        moon_state = self.cartesian_state()
        moon_position, moon_velocity = moon_state[0:3], moon_state[3:6]
        moon_orbital_axis = unit_vector(np.cross(moon_position, moon_velocity))
        return moon_orbital_axis

    @property
    def eccentricity_vector(self) -> np.ndarray:
        moon_state = self.cartesian_state()
        return eccentricity_vector_from_cartesian_state(moon_state)

    @property
    def orbital_energy(self):
        moon_state = self.cartesian_state()
        moon_position, moon_velocity = moon_state[0:3], moon_state[3:6]
        return orbital_energy(LA.norm(moon_position), LA.norm(moon_velocity), jupiter_gravitational_parameter)

    @property
    def moon_name(self):
        return self._moon

    @property
    def semimajor_axis(self):
        return self._semimajor_axis

    @property
    def orbital_period(self):
        return self._orbital_period

    @property
    def SOI_radius(self):
        return self._SOI_radius

    @property
    def radius(self):
        return self._radius

    @property
    def gravitational_parameter(self):
        return self._gravitational_parameter

    def get_epoch(self):
        return self.epoch

    def set_epoch(self, epoch: float):
        self.epoch = epoch
        self._cartesian_state = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name=self._moon,
            observer_body_name="Jupiter",
            reference_frame_name=global_frame_orientation,
            aberration_corrections="NONE",
            ephemeris_time=self.epoch)
