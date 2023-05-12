from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import constants
from tudatpy.kernel.math import interpolators

import numpy as np
import os
global_parameters_file_directory = os.path.dirname(__file__)

# Load standard spice kernels
spice_interface.load_standard_kernels()

# Load other spice kernels
# (add them here)

# Global interpolator settings
global_interpolator_settings = interpolators.lagrange_interpolation(
    8, boundary_interpolation=interpolators.extrapolate_at_boundary)

# Global frame orientation
global_frame_orientation = 'ECLIPJ2000'


# VEHICLE PARAMETERS ###################################################################################################
# Atmospheric entry interface
atmospheric_entry_altitude = 450e3  # m

# Vehicle properties
vehicle_mass = 3000  # kg
vehicle_reference_area = 5.  # m^2
vehicle_radius = np.sqrt(vehicle_reference_area/np.pi)
# vehicle_reference_area = np.pi * vehicle_radius**2    choose which one you want to use as parameter
vehicle_cd = 1.2
vehicle_cl = 0.6
vehicle_hypersonic_K_parameter = 0.000315  # galileo probe aerodynamic -paper
########################################################################################################################


# JUPITER ENVIRONMENT ##################################################################################################
# Jupiter physical values
jupiter_mass = 1898.6e24  # kg
jupiter_radius = spice_interface.get_average_radius('Jupiter') # m     legacy value: 69911e3
jupiter_SOI_radius = 48.2e9  # m
jupiter_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Jupiter')
jupiter_surface_acceleration = jupiter_gravitational_parameter / jupiter_radius ** 2
jupiter_molecular_weight = 2.22  # kg/kmol
jupiter_gas_constant = constants.MOLAR_GAS_CONSTANT * 1e3 / jupiter_molecular_weight
jupiter_specific_heats_ratio = 1.5

# Jupiter atmosphere exponential model parameters
jupiter_scale_height = 27e3  # m -> https://web.archive.org/web/20111013042045/http://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html
jupiter_1bar_density = 0.16  # kg/m^3

# Jupiter atmosphere exponential layered model
# Layers: 'layer_1' 'layer_2' 'layer_3' 'layer_4'  (1:lowestLayer 4:highestLayer)
# Parameters for each layer: T_0, h_0, rho_0, alpha
jupiter_atmosphere_exponential_layered_model = {
    'layer_1': [425.0, -132e3, 1.5, -0.56e3],
    'layer_2': [100.0,   50e3, 2e-2, 2.7e3],
    'layer_3': [200.0,  320e3, 2e-7, 0.9067e3],
    'layer_4': [950.0, 1000e3, 3e-11, np.inf]
}

# Orbital and physical data of galilean moons in SI units
# Moons: 'Io' 'Europa' 'Ganymede' 'Callisto'
# Entries for each moon: 'SMA' 'Mass' 'Radius' 'Orbital_Period' 'SOI_Radius' 'mu' 'g_0'
moons_optimization_parameter_dict = {1: 'Io', 2: 'Europa', 3: 'Ganymede', 4: 'Callisto'}
galilean_moons_data = {
                       'Io': {'SMA': 421.77e6,
                              'Mass': 893.3e20,
                              'Radius': 1821.3e3,
                              'Orbital_Period': 1.769138 * constants.JULIAN_DAY,
                              'SOI_Radius': 7836e3,
                              'mu': 893.3e20 * constants.GRAVITATIONAL_CONSTANT,
                              'g_0': 893.3e20 * constants.GRAVITATIONAL_CONSTANT / (1821.3e3**2)},
                       'Europa': {'SMA': 671.08e6,
                                  'Mass': 479.7e20,
                                  'Radius': 1565e3,
                                  'Orbital_Period': 3.551810 * constants.JULIAN_DAY,
                                  'SOI_Radius': 9723e3,
                                  'mu': 479.7e20 * constants.GRAVITATIONAL_CONSTANT,
                                  'g_0': 479.7e20 * constants.GRAVITATIONAL_CONSTANT / (1565e3**2)},
                       'Ganymede': {'SMA': 1070.4e6,
                                    'Mass': 1482e20,
                                    'Radius': 2634e3,
                                    'Orbital_Period': 7.154553 * constants.JULIAN_DAY,
                                    'SOI_Radius': 24351e3,
                                    'mu': 1482e20 * constants.GRAVITATIONAL_CONSTANT,
                                    'g_0': 1482e20 * constants.GRAVITATIONAL_CONSTANT / (2634e3**2)},
                       'Callisto': {'SMA': 1882.8e6,
                                    'Mass': 1076e20,
                                    'Radius': 2403e3,
                                    'Orbital_Period': 16.689018 * constants.JULIAN_DAY,
                                    'SOI_Radius': 37684e3,
                                    'mu': 1076e20 * constants.GRAVITATIONAL_CONSTANT,
                                    'g_0': 1076e20 * constants.GRAVITATIONAL_CONSTANT / (2403e3**2)}
                       }
########################################################################################################################


# Galileo Mission ######################################################################################################
# Galileo probe data
galileo_mass = 339.  # kg
galileo_radius = 0.632  # m
galileo_ref_area = galileo_radius**2 * np.pi  # m^2
galileo_cd = 1.05 #1.02
galileo_cl = 0.

# Galileo entry interface parameters
galileo_entry_velocity = 59.58e3  # m/s  # galileo entry velocity wrt atmosphere: 47.4054 km/s     59.6e3
galileo_entry_fpa = np.deg2rad(-6.69)  # rad  # galileo entry fpa wrt atmosphere: -8.4104 deg   -6.69
galileo_entry_latitude = np.deg2rad(6.57)  # rad
galileo_entry_longitude = np.deg2rad(4.88)  # rad
galileo_entry_heading = np.deg2rad(-2.6111)  # rad
galileo_entry_altitude = 450e3  # m

galileo_mission_directory = '/Just_aerocapture/Numerical_approach/GalileoMission/'

# Galileo in-flight data taken from Table 6 of DOI: 10.1029/98JE01766
galileo_flight_data_seiff1998 = np.loadtxt(global_parameters_file_directory + galileo_mission_directory + 'galileo_flight_data.txt')
galileo_flight_epoch = galileo_flight_data_seiff1998[:, 0]
galileo_flight_altitude = galileo_flight_data_seiff1998[:, 1] * 1e3
galileo_flight_velocity = galileo_flight_data_seiff1998[:, 2] * 1e3
galileo_flight_fpa = galileo_flight_data_seiff1998[:, 3]
galileo_flight_mach_no = galileo_flight_data_seiff1998[:, 6]
galileo_flight_cd = galileo_flight_data_seiff1998[:, 9]
galileo_flight_altitude_boundaries = [galileo_flight_altitude[-1], galileo_flight_altitude[0]]

# Galileo in-flight data taken from  DOI: 10.1029/98JE01766
# Table 8
jupiter_upper_atmosphere_data_seiff1998 = np.loadtxt(global_parameters_file_directory + galileo_mission_directory +
                                   'galileo_flight_data_2.txt')
# Table 7
jupiter_lower_atmosphere_data_seiff1998 = np.loadtxt(global_parameters_file_directory + galileo_mission_directory +
                                   'galileo_lower_atm_flight_data_2.txt')
jupiter_altitude_atmosphere_values_seiff1998 = np.concatenate((jupiter_upper_atmosphere_data_seiff1998[:, 0],jupiter_lower_atmosphere_data_seiff1998[:, 1]))
jupiter_density_atmosphere_values_seiff1998 = np.concatenate((jupiter_upper_atmosphere_data_seiff1998[:, 3],jupiter_lower_atmosphere_data_seiff1998[:, 3]))
atmosphere_altitude_values_boundaries_seiff1998 = [jupiter_altitude_atmosphere_values_seiff1998[-1],
                                                   jupiter_altitude_atmosphere_values_seiff1998[0]]  # 0=1000km  -1=-135km
########################################################################################################################


