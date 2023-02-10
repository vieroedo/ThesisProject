# General imports
import os
import shutil
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from time import process_time as pt


# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators

import CapsuleEntryUtilities as Util
# import handle_functions as hanfun

# Make everything big
plt.rc('font', size=15)

show_galileo_flight = True
plot_galileo_tabulated_data = True
plot_heatfluxes_in_dep_var_plots = False

current_dir = os.path.dirname(__file__)
if show_galileo_flight:
    current_dir = current_dir + '/GalileoMission'
benchmark_output_path = current_dir + '/SimulationOutput/benchmarks/'


see_error_wrt_benchmark = False


# leave them like this for most script usages
benchmark_portion_to_evaluate = 3  # 0, 1, 2, 3 (3 means all three together)
step_size_study_benchmark = False  # leave false please
benchmark_number_to_retrieve = 1  # 1 or 2

if step_size_study_benchmark:
    benchmark_output_path = benchmark_output_path + 'step_size_study/'

if benchmark_portion_to_evaluate == 3:
    benchmark_output_path = benchmark_output_path + 'full_traj/'
else:
    benchmark_output_path = benchmark_output_path + 'portion_' + str(benchmark_portion_to_evaluate) + '/'

# Retrieve from file
state_history_matrix = np.loadtxt(current_dir + '/SimulationOutput/' + 'simulation_state_history.dat')
dependent_variable_history_matrix = np.loadtxt(current_dir + '/SimulationOutput/' + 'simulation_dependent_variable_history.dat')
state_history = dict(zip(state_history_matrix[:,0], state_history_matrix[:,1:]))
dependent_variable_history = dict(zip(dependent_variable_history_matrix[:,0], dependent_variable_history_matrix[:,1:]))

# Create matrices with data
# simulation_result = np.vstack(list(state_history.values()))
simulation_result = state_history_matrix[:,1:]
# epochs_vector = np.vstack(list(state_history.keys()))
epochs_vector = state_history_matrix[:,0]
epochs_plot = (epochs_vector - epochs_vector[0]) / constants.JULIAN_DAY
# dependent_variables = np.vstack(list(dependent_variable_history.values()))
dependent_variables = dependent_variable_history_matrix[:,1:]

# Slice to separate various dep variables and quantities
aero_acc = dependent_variables[:,0:3]
grav_acc = dependent_variables[:,3]
altitude = dependent_variables[:,4]
flight_path_angle = dependent_variables[:,5]
airspeed = dependent_variables[:, 6]
mach_number = dependent_variables[:,7]
atmospheric_density = dependent_variables[:,8]

spacecraft_velocity_states = simulation_result[:,3:6]

########################################################################################################################
# DEPENDENT VARIABLES PLOT #############################################################################################
########################################################################################################################

drag_direction = -Util.unit_vector(spacecraft_velocity_states)
# to_fix here
lift_direction = np.zeros((len(simulation_result[:,0]),3))
for i in range(len(simulation_result[:,0])):
    rotation_axis = Util.unit_vector(np.cross(simulation_result[i,0:3], simulation_result[i,3:6]))
    lift_direction[i] = Util.rotate_vectors_by_given_matrix(Util.rotation_matrix(rotation_axis, np.pi/2), drag_direction[i,:])
# lift_direction = Util.rotate_vectors_by_given_matrix(Util.rotation_matrix(Util.z_axis, np.pi/2), drag_direction)


drag_acc = np.zeros((len(dependent_variables), 3))
lift_acc = np.zeros((len(dependent_variables), 3))
for i in range(len(dependent_variables)):
    drag_acc[i, :] = np.dot(aero_acc[i, :], drag_direction[i, :]) * drag_direction[i, :]
    lift_acc[i, :] = np.dot(aero_acc[i, :], lift_direction[i, :]) * lift_direction[i, :]

noise_level = 1e-6

drag_acc = LA.norm(drag_acc, axis=1)
lift_acc = LA.norm(lift_acc, axis=1)
drag_acc_mod = np.delete(drag_acc, np.where(drag_acc <= noise_level))
# lift_acc_mod = np.delete(lift_acc, np.where(lift_acc <= noise_level))

entry_epochs_cells = list(np.where(drag_acc > noise_level)[0])

lift_acc_mod = lift_acc[entry_epochs_cells].reshape(len(drag_acc_mod))

# downrange = np.zeros(len(entry_epochs_cells))
# for i, cell in enumerate(entry_epochs_cells):
#     current_position = simulation_result[cell, 0:3]

atmosphere_interfaces_cell_no = np.where(drag_acc > 1e-2)[0][[0,-1]]

atmosphere_altitude_interfaces = altitude[atmosphere_interfaces_cell_no].reshape(2)
atmosphere_fpa_interfaces = (flight_path_angle[atmosphere_interfaces_cell_no[0]], flight_path_angle[atmosphere_interfaces_cell_no[1]])

if plot_heatfluxes_in_dep_var_plots:
    plots_number = 7
else:
    plots_number = 6
Fig_f, axs_f = plt.subplots(plots_number, 1, figsize=(7,9), sharex='col')

fig_hf, ax_hf = plt.subplots(figsize=(5,6))

epochs_ae_phase = epochs_vector[entry_epochs_cells].reshape(len(drag_acc_mod))
epochs_plot_ae_phase = epochs_ae_phase - epochs_ae_phase[0]


axs_f[0].plot(epochs_plot_ae_phase, drag_acc_mod, label='drag')
axs_f[0].plot(epochs_plot_ae_phase, lift_acc_mod, label='lift')
axs_f[0].plot(epochs_plot_ae_phase, grav_acc[entry_epochs_cells].reshape(len(drag_acc_mod)), label='gravity')
axs_f[0].set(ylabel='acceleration [m/s^2]')
axs_f[0].legend()

axs_f[1].plot(epochs_plot_ae_phase, altitude[entry_epochs_cells].reshape(len(drag_acc_mod))/1e3)
axs_f[1].axhline(y=atmosphere_altitude_interfaces[0]/1e3, color='grey', linestyle='dotted')
axs_f[1].axhline(y=atmosphere_altitude_interfaces[1]/1e3, color='grey', linestyle='dotted')
axs_f[1].set(ylabel='altitude [km]')

axs_f[2].plot(epochs_plot_ae_phase, flight_path_angle[entry_epochs_cells].reshape(len(drag_acc_mod))*180/np.pi)
axs_f[2].axhline(y=atmosphere_fpa_interfaces[0]*180/np.pi, color='grey', linestyle='dotted')
axs_f[2].axhline(y=atmosphere_fpa_interfaces[1]*180/np.pi, color='grey', linestyle='dotted')
axs_f[2].set( ylabel='f.p.a. [deg]')

axs_f[3].plot(epochs_plot_ae_phase, mach_number[entry_epochs_cells].reshape(len(drag_acc_mod)))
axs_f[3].set(ylabel='Mach number [-]')

axs_f[4].plot(epochs_plot_ae_phase, atmospheric_density[entry_epochs_cells].reshape(len(drag_acc_mod)))
axs_f[4].set(ylabel='Density [kg/m^3]', yscale='log')

axs_f[5].plot(epochs_plot_ae_phase, airspeed[entry_epochs_cells].reshape(len(drag_acc_mod)) / 1e3)
axs_f[5].set(ylabel='Airspeed [km/s]')

if not plot_heatfluxes_in_dep_var_plots:
    axs_f[5].set(xlabel='elapsed time [s]')


if show_galileo_flight:
    nose_radius = np.sqrt(Util.galileo_ref_area / np.pi)
else:
    nose_radius = np.sqrt(Util.vehicle_reference_area / np.pi)
convective_hf, radiative_hf, radiative_hf_w_blockage = Util.atmospheric_entry_heat_loads_correlations(
    atmospheric_density, airspeed, nose_radius=nose_radius)


# convective_hf_w_blockage = Util.heat_flux_with_blockage_from_blowing(radiative_hf_w_blockage+convective_hf, convective_hf, atmospheric_density, airspeed, )
convective_hf_w_blockage = Util.convective_heat_flux_with_blockage(convective_hf, atmospheric_density, airspeed)#), total_wall_hfx=convective_hf+radiative_hf)
radiative_hf_wall = Util.convective_heat_flux_with_blockage(radiative_hf_w_blockage, atmospheric_density, airspeed)


# radius, density, velocity, heat_flux
scaling_vector = np.array([1.58227848e+00, 4.92610837e+02, 2.09863589e-02, 6.49350649e-06])

custom_convective_hf = Util.custom_atm_convective_hfx_correlation(
    atmospheric_density*scaling_vector[1], airspeed*scaling_vector[2]/1e3, radius=nose_radius*scaling_vector[0])
custom_convective_hf = custom_convective_hf/scaling_vector[3] # kW/m^2
custom_convective_hf = custom_convective_hf*1e3  # W/m^2


if plot_heatfluxes_in_dep_var_plots:
    axs_f[6].plot(epochs_plot_ae_phase, convective_hf[entry_epochs_cells].reshape(len(drag_acc_mod)) / 1e3, label='conv')
    axs_f[6].plot(epochs_plot_ae_phase, convective_hf_w_blockage[entry_epochs_cells].reshape(len(drag_acc_mod)) / 1e3, label='conv_w_blockage')
    axs_f[6].plot(epochs_plot_ae_phase, radiative_hf[entry_epochs_cells].reshape(len(drag_acc_mod)) / 1e3, label='rad')
    axs_f[6].plot(epochs_plot_ae_phase, radiative_hf_w_blockage[entry_epochs_cells].reshape(len(drag_acc_mod)) / 1e3, label='rad_w_blockage')
    axs_f[6].plot(epochs_plot_ae_phase, radiative_hf_wall[entry_epochs_cells].reshape(len(drag_acc_mod)) / 1e3, label='rad_wall_hfx')

    axs_f[6].plot(epochs_plot_ae_phase, custom_convective_hf[entry_epochs_cells].reshape(len(drag_acc_mod)) / 1e3, label='conv_custom')


    axs_f[6].set(xlabel='elapsed time [s]', ylabel='Heat fluxes [kW/m^2]')
    axs_f[6].legend()

ax_hf.plot(epochs_plot_ae_phase, convective_hf[entry_epochs_cells].reshape(len(drag_acc_mod)) / 1e3, label='conv')
ax_hf.plot(epochs_plot_ae_phase, convective_hf_w_blockage[entry_epochs_cells].reshape(len(drag_acc_mod)) / 1e3, label='conv_w_blockage')
ax_hf.plot(epochs_plot_ae_phase, radiative_hf[entry_epochs_cells].reshape(len(drag_acc_mod)) / 1e3, label='rad')
ax_hf.plot(epochs_plot_ae_phase, radiative_hf_w_blockage[entry_epochs_cells].reshape(len(drag_acc_mod)) / 1e3, label='rad_w_blockage', linestyle='-.')
ax_hf.plot(epochs_plot_ae_phase, radiative_hf_wall[entry_epochs_cells].reshape(len(drag_acc_mod)) / 1e3, label='rad_wall_hfx')

ax_hf.plot(epochs_plot_ae_phase, custom_convective_hf[entry_epochs_cells].reshape(len(drag_acc_mod)) / 1e3, label='conv_custom', linestyle='-')


ax_hf.set(xlabel='elapsed time [s]', ylabel='Heat fluxes [kW/m^2]')
ax_hf.legend()


Fig_vh, axs_vh = plt.subplots(figsize = (6,5))
axs_vh.plot(airspeed[entry_epochs_cells].reshape(len(drag_acc_mod)) / 1e3, altitude[entry_epochs_cells].reshape(len(drag_acc_mod)) / 1e3)
axs_vh.set(xlabel='airspeed [km/s]', ylabel='altitude [km]')


## HEAT FLUXES COMPARISON
total_heat_flux = convective_hf+radiative_hf # W/m2
total_heat_flux_w_blockage = convective_hf_w_blockage+radiative_hf_wall # W/m2
only_rad_w_blockage = radiative_hf_wall

peak_heat_flux = max(total_heat_flux)  # W/m^2
peak_heat_flux_w_blockage = max(total_heat_flux_w_blockage)  # W/m^2
peak_heat_flux_only_rad = max(only_rad_w_blockage)  # W/m^2

integrated_heat_load = np.trapz(total_heat_flux, epochs_vector) # J/m^2
integrated_heat_load_w_blockage = np.trapz(total_heat_flux_w_blockage, epochs_vector) # J/m^2
integrated_heat_load_only_rad = np.trapz(only_rad_w_blockage, epochs_vector) # J/m^2


# integrated_heat_load = 0
# for i in range(1, len(epochs_vector)):
#     delta_t = epochs_vector[i] - epochs_vector[i-1]
#     integrated_heat_load = integrated_heat_load + (total_heat_flux[i] + total_heat_flux[i-1]) * delta_t / 2
# print(integrated_heat_load)

tps_mass_fraction = (0.091 * (integrated_heat_load/1e4) ** 0.51575)/100
tps_mass_fraction_w_blockage = (0.091 * (integrated_heat_load_w_blockage/1e4) ** 0.51575)/100
tps_mass_fraction_only_rad = (0.091 * (integrated_heat_load_only_rad/1e4) ** 0.51575)/100


if show_galileo_flight:
    tps_mass = Util.galileo_mass * tps_mass_fraction
    tps_mass_w_blockage = Util.galileo_mass * tps_mass_fraction_w_blockage
    tps_mass_only_rad = Util.galileo_mass * tps_mass_fraction_only_rad
else:
    tps_mass = Util.vehicle_mass * tps_mass_fraction
    tps_mass_w_blockage = Util.vehicle_mass * tps_mass_fraction_w_blockage
    tps_mass_only_rad = Util.vehicle_mass * tps_mass_fraction_only_rad

print('WITHOUT RADIATION BLOCKAGE')
print(f'The heat load is of {integrated_heat_load:.3f} J/m^2  or  {integrated_heat_load/1e4:.3f} J/cm^2')
print(f'TPS mass fraction is {tps_mass_fraction:.5f}, which corresponds to a mass of {tps_mass:.3f} kg')
print(f'Peak heat flux is {peak_heat_flux/1e3:.3f} kw/m^2')

print('\nWITH RADIATION BLOCKAGE AND BLOWING (WALL CONDITIONS)')
print(f'The heat load is of {integrated_heat_load_w_blockage:.3f} J/m^2  or  {integrated_heat_load_w_blockage/1e4:.3f} J/cm^2')
print(f'TPS mass fraction is {tps_mass_fraction_w_blockage:.5f}, which corresponds to a mass of {tps_mass_w_blockage:.3f} kg')
print(f'Peak heat flux is {peak_heat_flux_w_blockage/1e3:.3f} kw/m^2')

print('\nWITH JUST RADIATION WALL COND')
print(f'The heat load is of {integrated_heat_load_only_rad:.3f} J/m^2  or  {integrated_heat_load_only_rad/1e4:.3f} J/cm^2')
print(f'TPS mass fraction is {tps_mass_fraction_only_rad:.5f}, which corresponds to a mass of {tps_mass_only_rad:.3f} kg')
print(f'Peak heat flux is {peak_heat_flux_only_rad/1e3:.3f} kw/m^2')



########################################################################################################################
# MISSION DATA PRINTS ##################################################################################################
########################################################################################################################

states_to_evaluate = [0,-1]
states_names = ['initial', 'final']
states_titles = ['INITIAL ORBIT', 'FINAL ORBIT']
states_dict = dict()
for i, current_state_to_eval in enumerate(states_to_evaluate):
    print(f'\n### {states_titles[i]} ###')
    curr_state = simulation_result[current_state_to_eval, :]
    curr_position = curr_state[0:3]
    curr_velocity = curr_state[3:6]

    term1 = LA.norm(curr_velocity) ** 2 - Util.central_body_gravitational_parameter / LA.norm(curr_position)
    term2 = np.dot(curr_position, curr_velocity)
    curr_eccentricity_vector = (term1 * curr_position - term2 * curr_velocity) / Util.central_body_gravitational_parameter
    curr_eccentricity = LA.norm(curr_eccentricity_vector)

    curr_angular_momentum = LA.norm(np.cross(curr_position, curr_velocity))
    curr_orbital_energy = Util.orbital_energy(LA.norm(curr_position), LA.norm(curr_velocity))

    curr_orbit_sma = - Util.central_body_gravitational_parameter / (2 * curr_orbital_energy)

    add_string = ''
    conjug = ' and'

    if current_state_to_eval == -1:
        final_orbit_orbital_period = 2*np.pi * np.sqrt(curr_orbit_sma**3/Util.central_body_gravitational_parameter)
        add_string = f' and orbital period {final_orbit_orbital_period/constants.JULIAN_DAY:.3f} days'
        conjug = ','

    states_dict[states_names[i]] = curr_orbital_energy
    print(f'The {states_names[i]} orbit has eccentricity {curr_eccentricity:.5f}{conjug} specific energy {curr_orbital_energy / 1e3:.3f} kJ/kg' + add_string)

    if current_state_to_eval == 0:
        initial_orbit_pericenter = curr_orbit_sma * (1-curr_eccentricity)
        initial_orbit_pericenter_velocity = curr_angular_momentum / initial_orbit_pericenter
        target_sma = initial_orbit_pericenter / (1-0.98)
        target_energy = - Util.central_body_gravitational_parameter / (2 * target_sma)
        target_velocity = Util.velocity_from_energy(target_energy,initial_orbit_pericenter)
        delta_v = initial_orbit_pericenter_velocity - target_velocity
        print(f'Initial pericenter velocity: {initial_orbit_pericenter_velocity/1e3:.3f} km/s')
        print(f'Target pericenter velocity: {target_velocity / 1e3:.3f} km/s   (target orbit eccentricity: 0.98)')
        print(f'The required delta v to impart at pericenter to have eccentricity of 0.98 is of: {delta_v/1e3:.3f} km/s')

        test_specific_impulse = 318  # s
        g_0 = 9.80665  # m/s^2
        mass_fraction_mf_mi = np.exp(-delta_v/(g_0*test_specific_impulse))
        vehicle_mass = Util.vehicle_mass
        final_mass = mass_fraction_mf_mi * vehicle_mass
        propellant_mass = vehicle_mass - final_mass
        print(f'Which corresponds to a propellant mass of {propellant_mass:.3f} kg, for a propellant mass fraction of {propellant_mass/vehicle_mass:.5f}')

v_initial = Util.velocity_from_energy(states_dict['initial'], initial_orbit_pericenter)
v_final = Util.velocity_from_energy(states_dict['final'], initial_orbit_pericenter)
delta_v2 = v_initial - v_final
delta_E = states_dict['final'] - states_dict['initial']
print(f'\nThe difference in orbital specific energy is {abs(delta_E/1e3):.3f} kJ/kg')
print(f'The delta v of the aerocapture is {delta_v2/1e3:.3f} km/s')

########################################################################################################################
# 3-D TRAJECTORY PLOT ##################################################################################################
########################################################################################################################

# Plot 3-D Trajectory
fig = plt.figure()
ax = plt.axes(projection='3d')

# draw jupiter
u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
x = Util.jupiter_radius * np.cos(u) * np.sin(v)
y = Util.jupiter_radius * np.sin(u) * np.sin(v)
z = Util.jupiter_radius * np.cos(v)
ax.plot_wireframe(x, y, z, color="saddlebrown")

# label axes and figure
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('Jupiter aerocapture trajectory')

# draw galilean moons' orbits
for moon in Util.galilean_moons_data.keys():
    moon_sma = Util.galilean_moons_data[moon]['SMA']
    theta_angle = np.linspace(0, 2*np.pi, 200)
    x_m = moon_sma * np.cos(theta_angle)
    y_m = moon_sma * np.sin(theta_angle)
    z_m = np.zeros(len(theta_angle))
    ax.plot3D(x_m, y_m, z_m, 'b')

# set proper plot axis limits
xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
ax.set_xlim3d(XYZlim)
ax.set_ylim3d(XYZlim)
ax.set_zlim3d(XYZlim * 0.75)
ax.set_aspect('auto')

# plot trajectory
ax.plot3D(simulation_result[:,0], simulation_result[:,1], simulation_result[:,2], 'gray')


########################################################################################################################
# ERROR WRT BENCHMARK ##################################################################################################
########################################################################################################################

if see_error_wrt_benchmark:
    fig2, ax2 = plt.subplots(figsize=(6,5))

    benchmark_name = 'benchmark_' + str(benchmark_number_to_retrieve) + '_states.dat'
    bench1 = np.loadtxt(benchmark_output_path + benchmark_name)
    benchmark_state_difference_matrix = np.loadtxt(benchmark_output_path + 'benchmarks_state_difference.dat')
    benchmark_state_difference = dict(zip(benchmark_state_difference_matrix[:,0], benchmark_state_difference_matrix[:,1:]))
    benchmark_other_info = np.loadtxt(benchmark_output_path + 'ancillary_benchmark_info.txt', str)

    benchmark_error = np.vstack(list(benchmark_state_difference.values()))
    bench_diff_epochs = np.array(list(benchmark_state_difference.keys()))
    bench_diff_epochs_plot = (bench_diff_epochs - bench_diff_epochs[0]) / constants.JULIAN_DAY
    benchmark_error = benchmark_error[2:-2, :]
    bench_diff_epochs_plot = bench_diff_epochs_plot[2:-2]
    bench_diff_epochs = bench_diff_epochs[2:-2]  # useless for now
    position_error = LA.norm(benchmark_error[:, 0:3], axis=1)

    bench1_epochs = bench1[:,0]

    bench1_state_hist = dict()
    for e_num, epoch in enumerate(list(bench1_epochs)):
        bench1_state_hist[epoch] = bench1[e_num, 1:7]

    benchmark_interpolator_settings = interpolators.lagrange_interpolation(6,
             boundary_interpolation = interpolators.extrapolate_at_boundary)
    benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(bench1_state_hist,
                                                                                                    benchmark_interpolator_settings)

    state_difference = dict()

    for epoch in state_history.keys():
        state_difference[epoch] = state_history[epoch] - benchmark_state_interpolator.interpolate(epoch)



    ax2.plot(epochs_plot[8:-1], LA.norm(np.vstack(list(state_difference.values()))[:,0:3], axis=1)[8:-1])
    ax2.plot(bench_diff_epochs_plot, position_error, color='firebrick')
    ax2.set(xlabel='days', ylabel='error [m]')
    ax2.set_title(f'benchmark has step sizes: {list(benchmark_other_info[1:4])} [s]')
    ax2.set_yscale('log')

########################################################################################################################
# BENCHMARK STEP SIZE STUDY ############################################################################################
########################################################################################################################
plot_fit_to_benchmark_errors = False

if step_size_study_benchmark:
    ancillary_info = np.loadtxt(benchmark_output_path + 'ancillary_benchmark_info.txt')
    lower_limit = ancillary_info[0, 0]
    upper_limit = ancillary_info[-1, 0]
    no_of_entries = len(ancillary_info[:, 0])
    benchmark_step_sizes = np.rint(np.linspace(lower_limit, upper_limit, no_of_entries))

    benchmark_info = dict()
    max_position_errors = dict()
    step_size_name = ''
    for benchmark_step_size in benchmark_step_sizes:
        step_size_name = '_stepsize_' + str(benchmark_step_size)

        # first_bench = np.loadtxt(benchmark_output_path + 'benchmark_1_states' + step_size_name + '.dat')
        # first_bench_dep_var = np.loadtxt(
        #     benchmark_output_path + 'benchmark_1_dependent_variables' + step_size_name + '.dat')
        # second_bench = np.loadtxt(benchmark_output_path + 'benchmark_2_states' + step_size_name + '.dat')
        # second_bench_dep_var = np.loadtxt(
        #     benchmark_output_path + 'benchmark_2_dependent_variables' + step_size_name + '.dat')
        # benchmark_list = [dict(zip(first_bench[:, 0], first_bench[:, 1:])),
        #                   dict(zip(second_bench[:, 0], second_bench[:, 1:])),
        #                   dict(zip(first_bench_dep_var[:, 0], first_bench_dep_var[:, 1:])),
        #                   dict(zip(second_bench_dep_var[:, 0], second_bench_dep_var[:, 1:]))
        #                   ]
        benchmark_state_difference_matrix = np.loadtxt(benchmark_output_path + 'benchmarks_state_difference' + step_size_name + '.dat')
        benchmark_state_difference = dict(zip(benchmark_state_difference_matrix[:, 0], benchmark_state_difference_matrix[:, 1:]))

        benchmark_error = np.vstack(list(benchmark_state_difference.values()))
        bench_diff_epochs = np.array(list(benchmark_state_difference.keys()))
        bench_diff_epochs_plot = (bench_diff_epochs - bench_diff_epochs[0]) / constants.JULIAN_DAY
        benchmark_error = benchmark_error[2:-2, :]
        bench_diff_epochs_plot = bench_diff_epochs_plot[2:-2]
        bench_diff_epochs = bench_diff_epochs[2:-2]  # useless for now
        position_error = LA.norm(benchmark_error[:, 0:3], axis=1)
        max_position_error = np.amax(position_error)
        max_position_errors[benchmark_step_size] = max_position_error


    figg, axx = plt.subplots(figsize=(6,5))
    axx.plot(max_position_errors.keys(),max_position_errors.values(),marker='D',fillstyle='none')
    # if plot_fit_to_benchmark_errors:
    #     a = max_position_errors[list(max_position_errors.keys())[choose_ref_value_cell]] / list(max_position_errors.keys())[choose_ref_value_cell]**global_truncation_error_power
    #     fitting_function = a * np.array(list(max_position_errors.keys()))**global_truncation_error_power
    #     axx.plot(max_position_errors.keys(), fitting_function, color='firebrick')
    axx.set_yscale('log')
    axx.set_xscale('log')
    axx.set_xlabel('time step (s)')
    axx.set_ylabel('maximum position error norm (m)')
    axx.axhline(y=0.1, color='r',linestyle=':')

########################################################################################################################
# RKF STEP SIZES PLOT ##################################################################################################
########################################################################################################################

Fig3, ax3 = plt.subplots(2, 1, figsize= (6,5), sharex='col')
time_steps = np.diff(epochs_vector, n=1, axis=0)
ax3[0].plot(epochs_plot[:-1], time_steps,marker='D',fillstyle='none')
# ax3[0].scatter(epochs_plot[:-1], time_steps)

ax3[1].plot(epochs_plot, altitude)

# Fig4, ax4 = plt.subplots(figsize=(6,5))
# bench_error = np.loadtxt(current_dir + '/SimulationOutput/benchmarks/benchmarks_state_difference.dat')
#
# bench_position_e rror = LA.norm(bench_error[:,1:4], axis=1)
#
# ax4.plot(bench_error[:,0], bench_position_error)
# ax4.set_yscale('log')
# ax4.set_title('Benchmark error')

########################################################################################################################
# GALILEO TABULATED DATA ###############################################################################################
########################################################################################################################

if plot_galileo_tabulated_data:
    galileo_flight_data = np.loadtxt(current_dir +'/galileo_flight_data.txt')
    upper_atmosphere_data = np.loadtxt(current_dir + '/galileo_flight_data_2.txt')


    flight_epoch = galileo_flight_data[:,0]
    flight_altitude = galileo_flight_data[:,1] *1e3
    flight_velocity = galileo_flight_data[:,2] * 1e3
    flight_fpa = galileo_flight_data[:,3]
    flight_mach_no = galileo_flight_data[:,6]
    flight_cd = galileo_flight_data[:,9]

    entry_altitudes = altitude[entry_epochs_cells].reshape(len(drag_acc_mod))
    flight_heat_fluxes = Util.galileo_heat_fluxes_park(entry_altitudes)

    flight_density = Util.jupiter_atmosphere_density_model(flight_altitude)

    flight_drag = 0.5 * flight_cd * flight_density * Util.galileo_ref_area * (flight_velocity)**2 / Util.galileo_mass

    # epochs_ae_phase = epochs_vector[entry_epochs_cells].reshape(len(drag_acc_mod))
    # epochs_plot_ae_phase = epochs_ae_phase - epochs_ae_phase[0]

    starting_cell = 15
    time_offset = 1.137

    flight_epochs_plot = flight_epoch[starting_cell:] - flight_epoch[starting_cell] + time_offset
    linestyle_string='--'

    axs_f[0].plot(flight_epochs_plot, flight_drag[starting_cell:], color='grey', label='galileo drag', linestyle=linestyle_string)
    axs_f[0].legend()

    axs_f[1].plot(flight_epochs_plot, flight_altitude[starting_cell:] / 1e3, color='grey', linestyle=linestyle_string)
    # axs_f[1].axhline(y=atmosphere_altitude_interfaces[0] / 1e3, color='grey', linestyle='dotted')
    # axs_f[1].axhline(y=atmosphere_altitude_interfaces[1] / 1e3, color='grey', linestyle='dotted')
    # axs_f[1].set(ylabel='altitude [km]')

    axs_f[2].plot(flight_epochs_plot, flight_fpa[starting_cell:], color='grey', linestyle=linestyle_string)
    # axs_f[2].set(ylabel='f.p.a. [deg]')

    axs_f[3].plot(flight_epochs_plot, flight_mach_no[starting_cell:], color='grey', linestyle=linestyle_string)
    # axs_f[3].set(ylabel='Mach number [-]')

    axs_f[4].plot(flight_epochs_plot, flight_density[starting_cell:], color='grey', linestyle=linestyle_string)
    # axs_f[4].set(ylabel='Density [kg/m^3]', yscale='log')

    axs_f[5].plot(flight_epochs_plot, flight_velocity[starting_cell:] / 1e3, color='grey', linestyle=linestyle_string)
    # axs_f[5].set(xlabel='elapsed time [s]', ylabel='Airspeed [km/s]')

    if plot_heatfluxes_in_dep_var_plots:
        axs_f[6].plot(epochs_plot_ae_phase, flight_heat_fluxes[:,0] / 1e3, label='rad_bl_e_gal')
        axs_f[6].plot(epochs_plot_ae_phase, flight_heat_fluxes[:,1] / 1e3, label='rad_w_gal')
        axs_f[6].plot(epochs_plot_ae_phase, flight_heat_fluxes[:, 2] / 1e3, label='conv_gal')
        # axs_f[6].set(xlabel='elapsed time [s]', ylabel='Heat fluxes [kW/m^2]')
        axs_f[6].legend()

    ax_hf.plot(epochs_plot_ae_phase, flight_heat_fluxes[:, 0] / 1e3, label='rad_bl_e_gal')
    ax_hf.plot(epochs_plot_ae_phase, flight_heat_fluxes[:, 1] / 1e3, label='rad_w_gal')
    ax_hf.plot(epochs_plot_ae_phase, flight_heat_fluxes[:, 2] / 1e3, label='conv_gal')
    # axs_f[6].set(xlabel='elapsed time [s]', ylabel='Heat fluxes [kW/m^2]')
    ax_hf.legend()

    # PLOT PARK RESULTS
    fig_park, ax_park = plt.subplots()
    ax_park.plot(epochs_plot_ae_phase, flight_heat_fluxes[:, 0] / (1e3*1e4), label=r'$q_{R_e}$')
    ax_park.plot(epochs_plot_ae_phase, flight_heat_fluxes[:, 1] / (1e3*1e4), label=r'$q_{R_w}$')
    ax_park.plot(epochs_plot_ae_phase, flight_heat_fluxes[:, 2] / (1e3*1e4), label=r'$q_{C_w}$')
    ax_park.set(xlabel='Flight time [s]', ylabel=r'Heat fluxes [kW/cm$^2$]')
    ax_park.set(xlim=[40.6,58.5])
    plt.tight_layout()
    ax_park.legend()


    # park heat load:
    total_heat_flux_park = flight_heat_fluxes[:, 1] + flight_heat_fluxes[:, 2]  # W/m2
    peak_heat_flux_park = max(total_heat_flux_park)  # W/m^2
    integrated_heat_load_park = np.trapz(total_heat_flux_park, epochs_plot_ae_phase)  # J/m^2
    tps_mass_fraction_park = (0.091 * (integrated_heat_load_park / 1e4) ** 0.51575) / 100
    tps_mass_park = Util.galileo_mass * tps_mass_fraction_park

    print('\nPARK RESULTS')
    print(f'The heat load is of {integrated_heat_load_park:.3f} J/m^2  or  {integrated_heat_load_park / 1e4:.3f} J/cm^2')
    print(f'TPS mass fraction is {tps_mass_fraction_park:.5f}, which corresponds to a mass of {tps_mass_park:.3f} kg')
    print(f'Peak heat flux is {peak_heat_flux_park / 1e3:.3f} kw/m^2')



    Fig_gm, ax_gm = plt.subplots(3, 1, figsize=(6,8), sharex='col')
    # epochs_plot_ae_phase = epochs_ae_phase - epochs_ae_phase[0]

    flight_data_list = [flight_velocity[starting_cell:], flight_altitude[starting_cell:], np.deg2rad(flight_fpa[starting_cell:])]
    dep_var_data = [airspeed, altitude, flight_path_angle]
    y_labels = ['velocity error [km/s]', 'altitude error [km]', 'fpa error [deg]']
    data_error_scaling_factor = [1e-3, 1e-3, 180/np.pi]

    for k, data_to_interpolate in enumerate(dep_var_data):
        interpolator_settings = interpolators.lagrange_interpolation(
            8, )

        entries_number = len(data_to_interpolate)
        data_to_interpolate_vector = np.array(list(data_to_interpolate)).reshape((entries_number, 1))
        data_to_interpolate_dict = dict(zip(epochs_plot_ae_phase, data_to_interpolate_vector))
        data_to_interpolate_interpolator = interpolators.create_one_dimensional_vector_interpolator(data_to_interpolate_dict,
                                                                                                    interpolator_settings)

        data_to_interpolate_interpolated = np.zeros(len(flight_epochs_plot))
        for i in range(len(flight_epochs_plot)):
            data_to_interpolate_interpolated[i] = data_to_interpolate_interpolator.interpolate(flight_epochs_plot[i])
        data_error = abs(data_to_interpolate_interpolated - flight_data_list[k])

        processed_data_error = data_error * data_error_scaling_factor[k]
        ax_gm[k].plot(flight_epochs_plot, processed_data_error)
        ax_gm[k].set_ylabel(y_labels[k])
    ax_gm[-1].set_xlabel('elapsed time [s]')

plt.show()
