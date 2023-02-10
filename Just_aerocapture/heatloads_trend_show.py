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


remove_values = -7  # Used -7 for writing the thesis report!!

rarefield_flight_data = np.loadtxt(current_dir + '/GalileoMission/rarefield_simulation_data/galileo_rarefield_entry_rarefield_data.txt')
rarefield_heatloads_data = np.loadtxt(current_dir + '/GalileoMission/rarefield_simulation_data/galileo_rarefield_entry_simulation_heatloads.txt')
rarefield_heatloads_data = rarefield_heatloads_data[0:7, :]

#nose_rad, density, velocity, heatflux
rarefield_heatflux = rarefield_heatloads_data[:, 3] / 1e3 # kW/m^2
rarefield_density = rarefield_flight_data[:, 6]  # kg/m^3
rarefield_altitude = rarefield_flight_data[:, 1]  # km
rarefield_velocity = galileo_velocity_from_altitude(rarefield_altitude*1e3)/1e3  #km/s
# rarefield_velocity=47.450 * np.ones(len(rarefield_density))  # km/s


galileo_generic_data = np.loadtxt(handle_functions_directory + '/Just_aerocapture/GalileoMission/galileo_simulated_heatloads.txt')

simulated_altitudes = galileo_generic_data[:,1]  # km
simulated_velocities = galileo_generic_data[:,2]  # km/s
simulated_densities = galileo_generic_data[:,3]  # kg/m^3
simulated_qc = galileo_generic_data[:,8] *1e3 # kW/m^2
# simulated_qr = galileo_generic_data[:,9] *1e3  # kW/m^2

total_altitudes_data = np.concatenate((rarefield_altitude,simulated_altitudes))  # km
total_velocities_data = np.concatenate((rarefield_velocity, simulated_velocities))  # km/s
total_densities_data = np.concatenate((rarefield_density, simulated_densities))  # kg/m^3
total_qc_data = np.concatenate((rarefield_heatflux, simulated_qc))  # kW/m^2
nose_radius = galileo_radius * np.ones(len(total_qc_data))  # m

scaling_vector = np.array([1/nose_radius[0], 1/total_densities_data.max(), 1/total_velocities_data.max(), 1/total_qc_data.max()])
# scaling_vector = np.ones(4)

x_values_unscaled = np.vstack((nose_radius, total_densities_data, total_velocities_data))
x_values = (x_values_unscaled.T * scaling_vector[0:3]).T
y_values = total_qc_data * scaling_vector[3]

data_x = total_altitudes_data
data_y = total_qc_data


solution = sp.optimize.curve_fit(calculate_heat_flux, x_values[:,:remove_values], y_values[:remove_values])

parameters = solution[0]

altitudes = np.linspace(100,800,1000)  # km

x_val_2_unscaled = np.vstack((galileo_radius * np.ones(len(altitudes)), jupiter_atmosphere_density_model(altitudes*1e3), galileo_velocity_from_altitude(altitudes*1e3)/1e3))
calc_heatflux = np.zeros(len(altitudes))


x_val_2 = (x_val_2_unscaled.T*scaling_vector[0:3]).T
for i in range(len(altitudes)):
    calc_heatflux[i] = calculate_heat_flux(x_val_2[:, i], parameters[0], parameters[1], parameters[2])
calc_heatflux = calc_heatflux / scaling_vector[3]

print(parameters)
print(scaling_vector)
fig, ax = plt.subplots(figsize=(6,5), )
data_y = data_y/1e3
calc_heatflux = calc_heatflux/1e3

# ax.plot(data_x[:remove_values], data_y[:remove_values], label='nominal', linestyle='--')
ax.scatter(data_x[:remove_values], data_y[:remove_values], label='data points', color='steelblue')
if not remove_values == None:
    ax.scatter(data_x[remove_values:], data_y[remove_values:], label='unused data', facecolors='none', edgecolors='grey', linestyle='--')
    ax.axvline(x=150, color='grey', linestyle='--', linewidth=1.2, label='altitude limit of validity')
ax.plot(altitudes, calc_heatflux, label='curve fit', color='orange')
# ax.set_yscale('log')
ax.set_ylabel(r'$q_c$ [MW/m$^2$]')
ax.set_xlabel('Altitude [km]')
plt.legend()
plt.tight_layout()
plt.show()

