import numpy as np
import scipy as sp

from handle_functions import *
current_dir = os.path.dirname(__file__)

def calculate_heat_flux(independent_variables, c1, m, n):
    # independent_variables = [nose_radius, density, velocity]
    nose_radius = independent_variables[0]
    density = independent_variables[1]
    velocity = independent_variables[2]
    heat_flux = c1 * (nose_radius**-0.5) * density**m * velocity**n
    # hot wall correction can be neglected in hypersonic flow
    return heat_flux


# plot_rarefield_results =
use_weird_heat_fluxes = False
use_galileo_simulated_heatloads = False
use_rarefield_data = True


if use_weird_heat_fluxes:
    filename = 'heat_fluxes_galileo_low_altitudes.txt'
    galileo_generic_data = np.loadtxt(handle_functions_directory + '/Just_aerocapture/GalileoMission/' + filename)
if use_galileo_simulated_heatloads:
    filename = 'galileo_simulated_heatloads.txt'
    galileo_generic_data = np.loadtxt(handle_functions_directory + '/Just_aerocapture/GalileoMission/' + filename)
if use_rarefield_data:
    rarefield_flight_data = np.loadtxt(current_dir + '/GalileoMission/rarefield_simulation_data/galileo_rarefield_entry_rarefield_data.txt')
    rarefield_heatloads_data = np.loadtxt(current_dir + '/GalileoMission/rarefield_simulation_data/galileo_rarefield_entry_simulation_heatloads.txt')
    rarefield_heatloads_data = rarefield_heatloads_data[0:7, :]

if use_rarefield_data:
    #nose_rad, density, velocity, heatflux
    # scaling_values = np.array([1e2, 1e10, 1e0, 1e0])
    # scaling_values = np.array([1e0, 1e0, 1e0, 1e0])
    rarefield_heatflux = rarefield_heatloads_data[:, 3] / 1e3 # kW/m^2
    rarefield_density = rarefield_flight_data[:, 6]  # kg/m^3
    rarefield_altitude = rarefield_flight_data[:, 1]  # km
    rarefield_velocity = galileo_velocity_from_altitude(rarefield_altitude*1e3)/1e3  #km/s
    # rarefield_velocity=47.450 * np.ones(len(rarefield_density))  # km/s
    nose_radius = galileo_nose_radius * np.ones(len(rarefield_density))  # m

    #nondimensionalizing
    scaling_values = np.array([1/nose_radius[0], 1/rarefield_density.max(), 1/rarefield_velocity.max(), 1/rarefield_heatflux.max()])

    x_values_unscaled = np.vstack((nose_radius, rarefield_density, rarefield_velocity))
    x_values = (x_values_unscaled.T * scaling_values[0:3]).T
    y_values = rarefield_heatflux * scaling_values[3]

    data_x = rarefield_altitude
    data_y = rarefield_heatflux

if use_weird_heat_fluxes:
    # qr_boundary_layer_edge = galileo_generic_data[:, 4] * 1e7  # W/m^2
    # qr_wall = galileo_generic_data[:, 5] * 1e7  # W/m^2
    qc_wall = galileo_generic_data[:, 6] * 1e7  # W/m^2
    heat_fluxes_altitudes = galileo_generic_data[:, 1] * 1e3  # m
    heat_fluxes_densities = np.real(jupiter_atmosphere_density_model(heat_fluxes_altitudes))
    heat_fluxes_velocities = galileo_velocity_from_altitude(heat_fluxes_altitudes)/1e3  # km/s

    altitude_boundaries = [heat_fluxes_altitudes[-1], heat_fluxes_altitudes[0]]
    nose_radius = galileo_nose_radius * np.ones(len(heat_fluxes_densities))  # m

    x_values = np.vstack((nose_radius, heat_fluxes_densities, heat_fluxes_velocities))
    y_values = qc_wall

    data_x = heat_fluxes_altitudes/1e3
    data_y = qc_wall/1e3

if use_galileo_simulated_heatloads:
    simulated_altitudes = galileo_generic_data[:,1]  # km
    simulated_velocities = galileo_generic_data[:,2]  # km/s
    simulated_densities = galileo_generic_data[:,3]  # kg/m^3
    simulated_qc = galileo_generic_data[:,8]  # MW/m^2
    simulated_qr = galileo_generic_data[:,9]  # MW/m^2

    nose_radius = galileo_nose_radius * np.ones(len(simulated_qc))

    x_values = np.vstack((nose_radius, simulated_densities, simulated_velocities))
    y_values = simulated_qc

    data_x = simulated_altitudes
    data_y = simulated_qc*1e3

# rarefield_flight_data = np.loadtxt(current_dir + '/GalileoMission/rarefield_simulation_data/galileo_rarefield_entry_rarefield_data.txt')
# rarefield_heatloads_data = np.loadtxt(current_dir + '/GalileoMission/rarefield_simulation_data/galileo_rarefield_entry_simulation_heatloads.txt')
# rarefield_heatloads_data = rarefield_heatloads_data[0:7, :]

# heatflux = rarefield_heatloads_data[:, 3]
# flight_density = rarefield_flight_data[:,6]
# flight_altitude = rarefield_flight_data[:,1]
# flight_veloctiy = 47450 * np.ones(len(flight_density))  # m/s

if use_rarefield_data:
    # nfvec = 11
    solution = sp.optimize.curve_fit(calculate_heat_flux, x_values, y_values, full_output=True)
    # bounds = ([-1e7, -1e3, -1e3], [1e7, 1e3, 1e3])
    # nfvec = 46
    # solution = sp.optimize.curve_fit(calculate_heat_flux, x_values, y_values, full_output=True)

else:
    solution = sp.optimize.curve_fit(calculate_heat_flux, x_values, y_values)

parameters = solution[0]

altitudes = np.linspace(100,800,1000)  # km

if use_rarefield_data or use_weird_heat_fluxes or use_galileo_simulated_heatloads:
    scaling_factor = 1e-3
x_val_2 = np.vstack((galileo_nose_radius * np.ones(len(altitudes)), jupiter_atmosphere_density_model(altitudes*1e3), galileo_velocity_from_altitude(altitudes*1e3)*scaling_factor))
calc_heatflux = np.zeros(len(altitudes))

if use_rarefield_data:
    x_val_2 = (x_val_2.T*scaling_values[0:3]).T
for i in range(len(altitudes)):
    calc_heatflux[i] = calculate_heat_flux(x_val_2[:, i], parameters[0], parameters[1], parameters[2])

if use_weird_heat_fluxes:
    calc_heatflux = calc_heatflux/1e3
if use_galileo_simulated_heatloads:
    calc_heatflux = calc_heatflux*1e3
if use_rarefield_data:
    calc_heatflux = calc_heatflux / scaling_values[3]

print(parameters)
plt.plot(data_x, data_y, label='nominal', linestyle='--')
plt.scatter(data_x, data_y, label='nominal', linestyle='--')
plt.plot(altitudes, calc_heatflux, label='calculated')
plt.yscale('log')
plt.ylabel('q_c [kW/m^2]')
plt.legend()
plt.show()