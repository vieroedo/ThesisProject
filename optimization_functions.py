import numpy as np
import pandas as pd
import copy as cp

from JupiterTrajectory_GlobalParameters import *
from CapsuleEntryUtilities import *


ENTRY_SUPPORT_SYSTEM_MASS_FRACTION = 0.23
REQUIRED_CLOSED_ORBIT_ECCENTRICITY_CONVENTIONAL_BURN = 0.98
MINIMUM_ALTITUDE_FOR_INSERTION_BURN = 500e3  # m
NOMINAL_ALTITUDE_FOR_INSERTION_BURN = 2000e3  # m
BIPROPELLANT_ENGINE_SPECIFIC_IMPULSE = 320  # s
EARTH_G_ACCELERATION = 9.80665  # m/s2


def calculate_trajectory_heat_loads(density, velocity, nose_radius):

    convective_hfx, radiative_hfx = calculate_trajectory_heat_fluxes(density=density,
                                                                     velocity=velocity,
                                                                     nose_radius=nose_radius)
    total_wall_hfx = convective_hfx + radiative_hfx

    return total_wall_hfx


def calculate_trajectory_total_radiation_dose(epochs: np.ndarray, altitudes: np.ndarray):
    radiation_intensity_trend = radiation_intensity_function(altitudes)
    total_radiation_dose = np.trapz(radiation_intensity_trend, epochs)
    return total_radiation_dose


def aerocapture_payload_mass_fraction(epochs, wall_heat_flux):
    peak_heat_flux, total_heat_load = calculate_peak_hfx_and_heat_load(epochs, wall_heat_flux)
    tps_mass_fraction = calculate_tps_mass_fraction(total_heat_load)
    return 1 - ENTRY_SUPPORT_SYSTEM_MASS_FRACTION - tps_mass_fraction


def conventional_burn_payload_mass_fraction(interplanetary_arrival_excess_velocity: float, insertion_burn_pericenter_altitude:float):
    if np.asarray(interplanetary_arrival_excess_velocity < 0).any():
        raise ValueError('Excess velocity must be above 0 m/s')
    if np.asarray(insertion_burn_pericenter_altitude < MINIMUM_ALTITUDE_FOR_INSERTION_BURN).any():
        raise ValueError(f'Minimum altitude for insertion burn is set to {MINIMUM_ALTITUDE_FOR_INSERTION_BURN} m'
                         f'Use altitudes higher than that value.')

    pericenter_radius = jupiter_radius + insertion_burn_pericenter_altitude
    conv_orbit_energy = orbital_energy(jupiter_SOI_radius, interplanetary_arrival_excess_velocity, jupiter_gravitational_parameter)
    pericenter_velocity_mag = velocity_from_energy(conv_orbit_energy, pericenter_radius, jupiter_gravitational_parameter)

    final_orbit_eccentricity = REQUIRED_CLOSED_ORBIT_ECCENTRICITY_CONVENTIONAL_BURN
    final_orbit_sma = pericenter_radius / (1 - final_orbit_eccentricity)

    final_orbit_energy = - jupiter_gravitational_parameter/(2*final_orbit_sma)
    final_orbit_velocity_mag = velocity_from_energy(final_orbit_energy, pericenter_radius, jupiter_gravitational_parameter)

    # delta_v_check = np.sqrt(interplanetary_arrival_excess_velocity**2 + 2*jupiter_gravitational_parameter/pericenter_radius) - final_orbit_velocity_mag
    delta_v = pericenter_velocity_mag - final_orbit_velocity_mag
    if np.asarray(delta_v < 0).any():
        warnings.warn(r'$\Delta v$ is below 0! Arriving in a closed orbit already?')

    # prope_check = 1.12 * (1 - np.exp(-delta_v_check / (BIPROPELLANT_ENGINE_SPECIFIC_IMPULSE * EARTH_G_ACCELERATION)))
    propellant_mass_fraction = 1.12 * (1 - np.exp(-delta_v / (BIPROPELLANT_ENGINE_SPECIFIC_IMPULSE * EARTH_G_ACCELERATION)))
    payload_mass_fraction = 1 - propellant_mass_fraction
    return payload_mass_fraction


def aerocapture_payload_mass_fraction_benefit_over_insertion_burn(aerocapture_payload_mass_fraction, interplanetary_arrival_excess_velocity):

    insertion_burn_payload_mass_fraction = conventional_burn_payload_mass_fraction(interplanetary_arrival_excess_velocity, NOMINAL_ALTITUDE_FOR_INSERTION_BURN)

    return aerocapture_payload_mass_fraction - insertion_burn_payload_mass_fraction

# rp_alt_range = np.linspace(500e3,10000e3,1000)
# payload_mass_fraction = conventional_burn_payload_mass_fraction(5600., rp_alt_range)
# plt.plot(rp_alt_range/1e4, payload_mass_fraction)
# plt.xlabel('altitude [1000 km]')
# plt.show()


MAXIMUM_AERODYNAMIC_LOAD_ALLOWED = 30 * EARTH_G_ACCELERATION  # m/s2
MAXIMUM_PEAK_HEAT_FLUX_ALLOWED = 5000e4  # W/m2
MINIMUM_JUPITER_DISTANCE_ALLOWED = jupiter_radius  # m
MAXIMUM_JUPITER_DISTANCE_ALLOWED = jupiter_SOI_radius  # m
MAXIMUM_FINAL_ORBIT_ECCENTRICITY_ALLOWED = 0.99  # -

MAX_AERO_LOAD_NO = 0
PEAK_HEAT_FLUX_NO = 1
MIN_JUP_DISTANCE_NO = 2
MAX_JUP_DISTANCE_NO = 3
FINAL_ECCENTRICITY_NO = 4


def have_constraints_been_violated(constraint_values):
    maximum_aerodynamic_load = constraint_values[MAX_AERO_LOAD_NO]
    peak_heat_flux = constraint_values[PEAK_HEAT_FLUX_NO]
    minimum_jupiter_distance = constraint_values[MIN_JUP_DISTANCE_NO]
    maximum_jupiter_distance = constraint_values[MAX_JUP_DISTANCE_NO]
    final_orbit_eccentricity = constraint_values[FINAL_ECCENTRICITY_NO]

    is_one_violated = False
    constraints_violated = []

    if maximum_aerodynamic_load > MAXIMUM_AERODYNAMIC_LOAD_ALLOWED:
        is_one_violated = True
        constraints_violated = constraints_violated + [0]
        # return True, 0
    if peak_heat_flux > MAXIMUM_PEAK_HEAT_FLUX_ALLOWED:
        is_one_violated = True
        constraints_violated = constraints_violated + [1]
        # return True, 1
    if minimum_jupiter_distance < MINIMUM_JUPITER_DISTANCE_ALLOWED:
        is_one_violated = True
        constraints_violated = constraints_violated + [2]
        # return True, 2
    if maximum_jupiter_distance > MAXIMUM_JUPITER_DISTANCE_ALLOWED:
        is_one_violated = True
        constraints_violated = constraints_violated + [3]
        # return True, 3
    if final_orbit_eccentricity > MAXIMUM_FINAL_ORBIT_ECCENTRICITY_ALLOWED:
        is_one_violated = True
        constraints_violated = constraints_violated + [4]
        # return True, 4

    return is_one_violated, constraints_violated


is_constraint_character_upper_limit = (True, True, False, True, True)
constraint_vals = (MAXIMUM_AERODYNAMIC_LOAD_ALLOWED,
                   MAXIMUM_PEAK_HEAT_FLUX_ALLOWED,
                   MINIMUM_JUPITER_DISTANCE_ALLOWED,
                   MAXIMUM_JUPITER_DISTANCE_ALLOWED,
                   MAXIMUM_FINAL_ORBIT_ECCENTRICITY_ALLOWED)


def constraints_filter(grid_search_df: pd.DataFrame,
                       constraint_list: list,
                       is_limit_upper: list = is_constraint_character_upper_limit,
                       constraint_values: list = constraint_vals) -> pd.DataFrame:
    if not len(constraint_list) == len(is_limit_upper) == len(constraint_values):
        raise ValueError('wrong dimensions for constraints')

    filtered_df = cp.deepcopy(grid_search_df)
    for no, constraint in enumerate(constraint_list):
        constraint_limit = constraint_values[no]
        if is_limit_upper[no]:
            filtered_df = filtered_df[(filtered_df[constraint] < constraint_limit)]
        else:
            filtered_df = filtered_df[(filtered_df[constraint] > constraint_limit)]

    return filtered_df
