import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.linalg as LA
import numpy as np
import scipy as sp
import random
import os

# mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer

# Tudatpy imports
from tudatpy.io import save2txt, read_vector_history_from_file
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.astro import frame_conversion, element_conversion

# Problem-specific imports
from CapsuleEntryUtilities import compare_models, calculate_peak_hfx_and_heat_load, calculate_tps_mass_fraction
from handle_functions import eccentricity_vector_from_cartesian_state

from UncertaintyStudy_GlobalVariables import *

# SET PARAMETERS:  arc 0: from 0 to 7   arc 1: from 8 to 17   arc 12: from 18 to 27
uncertainty_to_analyze = 0
seed_number = 50  # default 50

rescale_y_axis_units = 1e-3  # allowed values are 1, 1e3 (mm/s), 1e-3 (km/s)
rescale_x_axis_units = 1e-3  # allowed values are 1, 1e3 (mm),   1e-3 (km)


# # from 0 to 3
# 'InitialPosition_RSW': 0, 'InitialPosition_R': 0, 'InitialPosition_S': 0, 'InitialPosition_W': 0,
# # from 4 to 7
# 'InitialVelocity_RSW': 0, 'InitialVelocity_R': 0, 'InitialVelocity_S': 0, 'InitialVelocity_W': 0,
#
# # from 8 to 11
# 'InitialPosition_RSW_Entry': 1, 'InitialPosition_R_Entry': 1,
# 'InitialPosition_S_Entry': 1, 'InitialPosition_W_Entry': 1,
# # from 12 to 15
# 'InitialVelocity_RSW_Entry': 1, 'InitialVelocity_R_Entry': 1,
# 'InitialVelocity_S_Entry': 1, 'InitialVelocity_W_Entry': 1,
# # 16, 17
# 'FlightPathAngle_Entry': 1, 'VelocityMag_Entry': 1,
#
# # from 18 to 21
# 'InitialPosition_RSW_Entry_FinalOrbit': 12, 'InitialPosition_R_Entry_FinalOrbit': 12,
# 'InitialPosition_S_Entry_FinalOrbit': 12, 'InitialPosition_W_Entry_FinalOrbit': 12,
# # from 22 to 25
# 'InitialVelocity_RSW_Entry_FinalOrbit': 12, 'InitialVelocity_R_Entry_FinalOrbit': 12,
# 'InitialVelocity_S_Entry_FinalOrbit': 12, 'InitialVelocity_W_Entry_FinalOrbit': 12,
# # 26, 27
# 'FlightPathAngle_Entry_FinalOrbit': 12, 'VelocityMag_Entry_FinalOrbit': 12,
#
# # from 28 to 31
# 'InitialPosition_RSW_FullOrbit': -1, 'InitialPosition_R_FullOrbit': -1,
# 'InitialPosition_S_FullOrbit': -1, 'InitialPosition_W_FullOrbit': -1,
# # from 32 to 35
# 'InitialVelocity_RSW_FullOrbit': -1, 'InitialVelocity_R_FullOrbit': -1,
# 'InitialVelocity_S_FullOrbit': -1, 'InitialVelocity_W_FullOrbit': -1


uncertainties = list(uncertainties_dictionary.keys())  # list of uncertainty names
arcs_computed = list(uncertainties_dictionary.values())  # list of corresponding arcs


current_dir = os.path.dirname(__file__)
uncertainty_analysis_folder = '/UncertaintyAnalysisData_seed' + str(seed_number) + '/'

subdirectory = uncertainty_analysis_folder + uncertainties[uncertainty_to_analyze] + '/'  # it can be 0, 1, 2
data_path = current_dir + subdirectory

perturbations = np.loadtxt(current_dir + uncertainty_analysis_folder + f'simulation_results_{uncertainties[uncertainty_to_analyze]}.dat')
number_of_runs = len(perturbations[:,0]) + 1

# trajectory_parameters = np.loadtxt(current_dir + uncertainty_analysis_folder + 'trajectory_parameters.dat')  # [0, vel, fpa]
# # evaluated_arc = trajectory_parameters[0]
# interplanetary_arrival_velocity = trajectory_parameters[1]
# atmospheric_entry_fpa = trajectory_parameters[2]

evaluated_arc = arcs_computed[uncertainty_to_analyze]

final_state_names = ['Atmospheric Entry', 'Atmospheric Exit', 'Final Orbit']
if evaluated_arc == 0:
    final_state_name = final_state_names[0]
    time_interval_names = ['(t_0)', '(t_E)']
elif evaluated_arc == 1:
    final_state_name = final_state_names[1]
    time_interval_names = ['(t_E)', '(t_F)']
elif evaluated_arc == 12 or evaluated_arc == -1:
    final_state_name = final_state_names[2]
    time_interval_names = ['(t_E)', '(t_1)'] if evaluated_arc == 12 else ['(t_0)', '(t_1)']
else:
    raise Exception('The propagated arc cannot yet be shown! Update the code.')


if evaluated_arc == 0:
    stop_before_aerocapture = True
    start_at_entry_interface = False
    stop_after_aerocapture = False
elif evaluated_arc == 1:
    stop_before_aerocapture = False
    start_at_entry_interface = True
    stop_after_aerocapture = True
elif evaluated_arc == 12:
    stop_before_aerocapture = False
    start_at_entry_interface = True
    stop_after_aerocapture = False
elif evaluated_arc == -1:
    stop_before_aerocapture = False
    start_at_entry_interface = False
    stop_after_aerocapture = False
else:
    raise Exception('Trajectory arc not yet supported. The variable evaluated_arc has an unsupported value.')

# This is made for compatibility between the various uncertainty analyses
entries = perturbations.shape[1]

# if uncertainty_to_analyze == 0:  # earth ephemeris
#     entry_names_x = [0, 'eph']
#     entry_names_y = [0, 'R \; (t_1)', 'S \; (t_1)', 'W \; (t_1)']
# elif uncertainty_to_analyze == 1:  # radiation pressure coefficient
#     entry_names_x = [0, 'C_r']
#     entry_names_y = [0, 'R \; (t_1)', 'S \; (t_1)', 'W \; (t_1)']


current_uncertainty_split = uncertainties[uncertainty_to_analyze].split('_')

entry_names_x = None
x_axis_measure_unit = None

if current_uncertainty_split[0] == 'InitialPosition':
    x_axis_measure_unit = '(m)'
    if rescale_x_axis_units == 1e3:
        x_axis_measure_unit = '(mm)'
    elif rescale_x_axis_units == 1e-3:
        x_axis_measure_unit = '(km)'
    elif rescale_x_axis_units != 1:
        raise Exception('Invalid value for x axis rescaling. Allowed values are 1, 1e3, 1e-3')

    if current_uncertainty_split[1] == 'RSW':
        entry_names_x = [0, 'R \;', 'S \;', 'W \;']
    else:
        entry_names_x = [0, current_uncertainty_split[1] + ' \;']

if current_uncertainty_split[0] == 'InitialVelocity':
    x_axis_measure_unit = '(m/s)'
    if rescale_x_axis_units == 1e3:
        x_axis_measure_unit = '(mm/s)'
    elif rescale_x_axis_units == 1e-3:
        x_axis_measure_unit = '(km/s)'
    elif rescale_x_axis_units != 1:
        raise Exception('Invalid value for x axis rescaling. Allowed values are 1, 1e3, 1e-3')

    if current_uncertainty_split[1] == 'RSW':
        entry_names_x = [0, 'V_R \;', 'V_S \;', 'V_W \;']
    else:
        entry_names_x = [0, 'V_' + current_uncertainty_split[1] + ' \;']

if current_uncertainty_split[0] == 'FlightPathAngle':
    entry_names_x = [0, '\gamma']
    x_axis_measure_unit = '(°)'
    if rescale_x_axis_units != 1:
        raise Exception('Unable to rescale x axis units for the flight path angle')


if current_uncertainty_split[0] == 'VelocityMag':
    entry_names_x = [0, 'V']
    x_axis_measure_unit = '(m/s)'
    if rescale_x_axis_units == 1e3:
        x_axis_measure_unit = '(mm/s)'
    elif rescale_x_axis_units == 1e-3:
        x_axis_measure_unit = '(km/s)'
    elif rescale_x_axis_units != 1:
        raise Exception('Invalid value for x axis rescaling. Allowed values are 1, 1e3, 1e-3')

if entry_names_x is None or x_axis_measure_unit is None:
    raise Exception('entry_names_x and/or x_axis_measure_unit are not defined')


entry_names_y = [0, 'R \;', 'S \;', 'W \;', 'V_R \;', 'V_S \;', 'V_W \;']
y_axis_measure_units = ['(m)', '(m/s)']

if rescale_y_axis_units == 1e-3:
    y_axis_measure_units = ['(km)', '(km/s)']
elif rescale_y_axis_units == 1e3:
    y_axis_measure_units = ['(mm)', '(mm/s)']
elif rescale_y_axis_units != 1:
    raise Exception('Invalid value for y axis rescaling. Allowed values are 1, 1e3, 1e-3')


print(f'Analyzing uncertainty: {uncertainties[uncertainty_to_analyze]}')

# PLOT HISTOGRAMS REGARDING THE APPLIED UNCERTAINTY
fig_hist, ax_hist = plt.subplots(entries-1, figsize=(7,6), sharex=True)
for entry in range(1,entries):
    if type(ax_hist) == np.ndarray:
        ax_hist[entry-1].hist(perturbations[:, entry]*rescale_x_axis_units, 20, edgecolor='black', linewidth=1.2)
        ax_hist[entry-1].set_xlabel(fr'$\Delta {entry_names_x[entry]} {time_interval_names[0]}$ ' + x_axis_measure_unit, fontsize=x_label_size)
        ax_hist[entry-1].tick_params(axis='both', which='major', labelsize=ticks_size)
        ax_hist[entry-1].tick_params(axis='x', labelbottom=True)
    else:
        ax_hist.hist(perturbations[:, entry]*rescale_x_axis_units, 20, edgecolor='black', linewidth=1.2)
        ax_hist.set_xlabel(fr'$\Delta {entry_names_x[entry]} {time_interval_names[0]}$ ' + x_axis_measure_unit, fontsize=x_label_size)
        ax_hist.tick_params(axis='both', which='major', labelsize=ticks_size)
fig_hist.suptitle('Random variable distribution at initial time', fontsize=suptitle_size)
fig_hist.supylabel('Occurrences', fontsize=common_y_label_size)
fig_hist.tight_layout()


state_history_end_values = dict()
rsw_end_values = dict()

# Create interpolator settings
interpolator_settings = interpolators.lagrange_interpolation(
    8, boundary_interpolation=interpolators.use_boundary_value)

# Extract nominal state history dictionary
nominal_state_history_np = np.loadtxt(data_path + 'state_history_0.dat')
nominal_state_history = dict(zip(nominal_state_history_np[:, 0],nominal_state_history_np[:, 1:]))
nominal_sh_interpolator = interpolators.create_one_dimensional_vector_interpolator(
    nominal_state_history, interpolator_settings)

nominal_final_eccentricity = LA.norm(eccentricity_vector_from_cartesian_state(nominal_state_history_np[-1, 1:]))

# PLOT THE POSITION ERROR RESULTING FROM THE APPLIED UNCERTAINTY
fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex='col')

for run_number in range(1, number_of_runs):
    state_history_np = np.loadtxt(data_path + 'state_history_' + str(run_number) + '.dat')
    state_difference_wrt_nominal_case_np = np.loadtxt(data_path + 'state_difference_wrt_nominal_case_' + str(run_number)
                                                      + '.dat')
    spacecraft_final_state = state_history_np[-1, 1:]

    epochs_comparison_sh = state_difference_wrt_nominal_case_np[:, 0]
    epochs_diff = (epochs_comparison_sh - epochs_comparison_sh[0]) / constants.JULIAN_DAY

    state_history = dict(zip(state_history_np[:, 0], state_history_np[:, 1:]))
    sh_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        state_history, interpolator_settings)

    state_difference_wrt_nominal_case = dict(zip(state_difference_wrt_nominal_case_np[:, 0], state_difference_wrt_nominal_case_np[:, 1:]))
    cartesian_state_difference = state_difference_wrt_nominal_case_np[:, 1:]

    position_difference = state_difference_wrt_nominal_case_np[:, 1:4]
    velocity_difference = state_difference_wrt_nominal_case_np[:, 4:7]
    position_difference_norm = LA.norm(position_difference, axis=1)
    velocity_difference_norm = LA.norm(velocity_difference, axis=1)

    altitude_difference, speed_difference = np.zeros(len(epochs_comparison_sh)), np.zeros(len(epochs_comparison_sh))
    # altitude_difference_debug = np.zeros(len(epochs_comparison_sh))
    for i, epoch in enumerate(epochs_comparison_sh):
        nominal_state = nominal_sh_interpolator.interpolate(epoch)
        current_state = sh_interpolator.interpolate(epoch)

        altitude_difference[i] = abs(LA.norm(nominal_state[0:3]) - LA.norm(current_state[0:3]))
        # altitude_difference_debug[i] = LA.norm(np.dot(cartesian_state_difference[i,0:3], Util.unit_vector(current_state[0:3])))
        speed_difference[i] = abs(LA.norm(nominal_state[3:6]) - LA.norm(current_state[3:6]))

    rsw_state_difference = np.zeros(np.shape(cartesian_state_difference))
    for i, epoch in enumerate(state_difference_wrt_nominal_case_np[:,0]):
        rsw_matrix = frame_conversion.inertial_to_rsw_rotation_matrix(sh_interpolator.interpolate(epoch))
        rsw_state_difference[i, 0:3] = rsw_matrix @ cartesian_state_difference[i, 0:3]
        rsw_state_difference[i, 3:6]= rsw_matrix @ cartesian_state_difference[i, 3:6]

    rsw_position_difference = rsw_state_difference[:,0:3]
    rsw_velocity_difference = rsw_state_difference[:,3:6]

    final_eccentricity = LA.norm(eccentricity_vector_from_cartesian_state(spacecraft_final_state))
    final_eccentricity_difference = final_eccentricity - nominal_final_eccentricity

    rsw_end_values[run_number] = rsw_state_difference[-1, :]
    state_history_end_values[run_number] = np.array([position_difference_norm[-1], velocity_difference_norm[-1], final_eccentricity_difference])  # np.amax(position_difference_norm)

    axs[0, 0].plot(epochs_diff, position_difference_norm*rescale_y_axis_units, color='blue')
    axs[1, 0].plot(epochs_diff, velocity_difference_norm*rescale_y_axis_units, color='blue')

    axs[0, 1].plot(epochs_diff, altitude_difference*rescale_y_axis_units, color='blue')
    axs[1, 1].plot(epochs_diff, speed_difference*rescale_y_axis_units, color='blue')

fig.suptitle(f'Performed Propagations (End state: {final_state_name})', fontsize=suptitle_size)
fig.supxlabel('Elapsed time (days)', fontsize=x_label_size)
axs[0, 0].set_ylabel('Position Error Magnitude ' + y_axis_measure_units[0], fontsize=y_label_size)
axs[1, 0].set_ylabel('Velocity Error Magnitude ' + y_axis_measure_units[1], fontsize=y_label_size)
axs[0, 1].set_ylabel('Altitude Error ' + y_axis_measure_units[0], fontsize=y_label_size)
axs[1, 1].set_ylabel('Speed Error ' + y_axis_measure_units[1], fontsize=y_label_size)
for i in [0,1]:
    for j in [0,1]:
        axs[i, j].set_xlabel('Elapsed time (days)', fontsize=x_label_size)
        axs[i, j].set_yscale('log')
        axs[i, j].tick_params(axis='both', which='major', labelsize=ticks_size)
        axs[i, j].tick_params(axis='x', labelbottom=True)
        # Show the major grid lines with dark grey lines
        axs[i ,j].grid(visible=True, which='major', color='#666666', linestyle='-')
        # Show the minor grid lines with very faint and almost transparent grey lines
        axs[i, j].minorticks_on()
        axs[i, j].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
fig.tight_layout()

# Reprocess values to plot them later
max_vals_in_rsw = np.vstack(list(rsw_end_values.values()))
state_history_end_values_array = np.vstack(list(state_history_end_values.values()))

# Extract the nominal dependent variable history
nominal_dependent_variable_history_np = np.loadtxt(data_path + 'dependent_variable_history_0.dat')
nominal_dependent_variable_history = dict(zip(nominal_dependent_variable_history_np[:, 0], nominal_dependent_variable_history_np[:, 1:]))
nominal_depvarh_interpolator = interpolators.create_one_dimensional_vector_interpolator(
    nominal_dependent_variable_history, interpolator_settings)

nominal_altitude_based_depvar_dictionary = dict(zip(nominal_dependent_variable_history_np[:,5], nominal_dependent_variable_history_np[:,6:8]))
nominal_altitude_keys = list(nominal_altitude_based_depvar_dictionary.keys())

# PLOT THE DIFFERENCE IN ENTRY PARAMETERS: f.p.a., entry airspeed, and so on + heat fluxes
dependent_variable_end_values = dict()
# heat_fluxes_values = dict()
for run_number in range(1, number_of_runs):
    dependent_variable_history_np = np.loadtxt(data_path + 'dependent_variable_history_' + str(run_number) + '.dat')
    dependent_variable_difference_wrt_nominal_case_np = np.loadtxt(data_path + 'dependent_variable_difference_wrt_nominal_case_' + str(run_number)
                                                      + '.dat')
    epochs_comparison_depvar = dependent_variable_difference_wrt_nominal_case_np[:, 0]

    dependent_variable_history = dict(zip(dependent_variable_history_np[:, 0], dependent_variable_history_np[:, 1:]))
    depvarh_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        dependent_variable_history, interpolator_settings)

    if np.any(dependent_variable_history_np[:, 5] < 450e3):
        there_is_aerocapture = 1
    else:
        there_is_aerocapture = 0

    # heat_fluxes_history = read_vector_history_from_file(2, data_path + 'heat_fluxes_history_' + str(run_number) + '.dat')

    # dependent_variable_difference_values = dependent_variable_difference_wrt_nominal_case_np[:, 1:]
    # epochs_comparison_depvar = dependent_variable_difference_wrt_nominal_case_np[:, 0]
    # dependent_variable_difference_wrt_nominal_case = dict()
    # for i, epoch in enumerate(epochs_comparison_depvar):
    #     dependent_variable_difference_wrt_nominal_case[epoch] = dependent_variable_difference_values[i, :]

    # fpa_difference, airspeed_difference = np.zeros(len(epochs_comparison_depvar)), np.zeros(len(epochs_comparison_depvar))
    # for i, epoch in enumerate(epochs_comparison_depvar):
    #     nominal_dependent_variables = nominal_depvarh_interpolator.interpolate(epoch)
    #     current_dependent_variables = depvarh_interpolator.interpolate(epoch)
    #
    #     fpa_difference[i] = abs(nominal_dependent_variables[:,5] - current_dependent_variables[:,5])
    #     airspeed_difference[i] = abs(nominal_dependent_variables[:,6] - current_dependent_variables[:,6])

    # if stop_before_aerocapture:
    if False:
        # altitude_velocity_dictionary = dict(zip())
        altitude_based_depvar_dictionary = dict(zip(dependent_variable_history_np[:, 5], dependent_variable_history_np[:, 6:8]))

        altitude_keys = list(altitude_based_depvar_dictionary.keys())
        # output_interpolation_step=-1000e3

        limit_value = 2
        interpolation_upper_limit = max(nominal_altitude_keys[limit_value], altitude_keys[limit_value])
        interpolation_lower_limit = min(nominal_altitude_keys[-limit_value], altitude_keys[-limit_value])

        # unfiltered_interpolation_altitudes = np.arange(altitude_keys[0], altitude_keys[-1], output_interpolation_step)
        unfiltered_interpolation_altitudes = np.geomspace(altitude_keys[0], altitude_keys[-1], int(1e4))
        unfiltered_interpolation_altitudes = [n for n in unfiltered_interpolation_altitudes if n <= interpolation_upper_limit]
        interpolation_altitudes = [n for n in unfiltered_interpolation_altitudes if n >= interpolation_lower_limit]

        print(f'Comparing dependent variables for run: {run_number}')
        depvar_difference_wrt_altitude = compare_models(altitude_based_depvar_dictionary,
                                                        nominal_altitude_based_depvar_dictionary,
                                                        interpolation_altitudes, None, None)
        print('Comparison done.\n')

        # fpa, velocity
        depvar_difference_wrt_altitude_values = np.vstack(list(depvar_difference_wrt_altitude.values()))
        final_fpa_difference = depvar_difference_wrt_altitude_values[-1,0]
        final_airspeed_difference = depvar_difference_wrt_altitude_values[-1,1]
    else:

        # inertial_fpa_based_depvar_dictionary = dict(zip(dependent_variable_history_np[:, 10], dependent_variable_history_np[:, [5,7,9]]))

        nominal_end_dependent_variables = nominal_depvarh_interpolator.interpolate(epochs_comparison_depvar[-1])
        current_end_dependent_variables = depvarh_interpolator.interpolate(epochs_comparison_depvar[-1])

        final_fpa_difference = -(nominal_end_dependent_variables[5] - current_end_dependent_variables[5])
        final_airspeed_difference = -(nominal_end_dependent_variables[6] - current_end_dependent_variables[6])

    # dependent_variable_end_values[run_number] = np.array([fpa_difference[-1], airspeed_difference[-1]])
    dependent_variable_end_values[run_number] = np.array([final_fpa_difference, final_airspeed_difference, there_is_aerocapture])

dependent_variable_end_values_array = np.vstack(list(dependent_variable_end_values.values()))

# Create x label for the following two plots
xlabel = ''
for entry in range(1, entries):
    x_component_name = entry_names_x[entry].split()[0]
    add_x_comma = ',' if entry < entries-1 else ''
    xlabel = xlabel + '\Delta ' + x_component_name + add_x_comma if entry < entries else xlabel

if stop_after_aerocapture or stop_before_aerocapture:
    fig_depvar, ax_depvar = plt.subplots(2,2, figsize=(6, 8))
    for entry in range(1, entries):
        marker = marker_styles[0][entry-1]
        facecolor = marker_styles[1][entry-1]
        edgecolor = marker_styles[2][entry-1]
        ax_depvar[0, 0].scatter(perturbations[:, entry]*rescale_x_axis_units, dependent_variable_end_values_array[:,0],
                     label=fr'$\Delta {entry_names_x[entry]} {time_interval_names[0]}$', marker=marker, facecolor=facecolor, edgecolor=edgecolor)
        ax_depvar[0, 1].scatter(perturbations[:, entry]*rescale_x_axis_units, dependent_variable_end_values_array[:, 1]*rescale_y_axis_units,
                     label=fr'$\Delta {entry_names_x[entry]} {time_interval_names[0]}$', marker=marker, facecolor=facecolor,edgecolor=edgecolor)

        aerocapture_performed_cells = list(np.where(dependent_variable_end_values_array[:, 2] == 1)[0])
        yes_ae__perturbations = perturbations[aerocapture_performed_cells, :]
        yes_ae__depvars_endvalues = dependent_variable_end_values_array[aerocapture_performed_cells, :]
        ax_depvar[1, 0].scatter(yes_ae__perturbations[:, entry]*rescale_x_axis_units, yes_ae__depvars_endvalues[:, 0],
                                label=fr'$\Delta {entry_names_x[entry]} {time_interval_names[0]}$', marker=marker, facecolor=facecolor,
                                edgecolor=edgecolor)
        ax_depvar[1, 1].scatter(yes_ae__perturbations[:, entry]*rescale_x_axis_units, yes_ae__depvars_endvalues[:, 1]*rescale_y_axis_units,
                                label=fr'$\Delta {entry_names_x[entry]} {time_interval_names[0]}$', marker=marker, facecolor=facecolor,
                                edgecolor=edgecolor)

    ax_depvar[0, 0].set_ylabel(fr'$\gamma \, {time_interval_names[1]}$ (°)', fontsize=y_label_size)
    ax_depvar[0, 1].set_ylabel(fr'$Airspeed \, {time_interval_names[1]}$ ' + y_axis_measure_units[1], fontsize=y_label_size)
    ax_depvar[1, 0].set_ylabel(fr'$\gamma \, {time_interval_names[1]}$ (°)', fontsize=y_label_size)
    ax_depvar[1, 1].set_ylabel(fr'$Airspeed \, {time_interval_names[1]}$ ' + y_axis_measure_units[1], fontsize=y_label_size)
    fig_depvar.supxlabel(fr'${xlabel} \, {time_interval_names[0]}$ ' + x_axis_measure_unit, fontsize=x_label_size)

    x_axis_bounds = np.amax(abs(perturbations[:,1:]*rescale_x_axis_units))
    for i in [0,1]:
        for j in [0,1]:
            ax_depvar[i,j].set_xlabel(fr'${xlabel} \, {time_interval_names[0]}$ ' + x_axis_measure_unit, fontsize=x_label_size)
            ax_depvar[i,j].set_xlim(-x_axis_bounds,x_axis_bounds)

            ax_depvar[i,j].tick_params(axis='both', which='major', labelsize=ticks_size)
            # Show the major grid lines with dark grey lines
            ax_depvar[i,j].grid(visible=True, which='major', color='#666666', linestyle='-')
            # Show the minor grid lines with very faint and almost transparent grey lines
            ax_depvar[i,j].minorticks_on()
            ax_depvar[i,j].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            if entries-1 > 1:
                ax_depvar[i, j].legend()
    fig_depvar.suptitle(fr'Impact of ${xlabel} \, {time_interval_names[0]} $ on the final $\gamma$ and Airspeed (End state: {final_state_name})', fontsize=suptitle_size)
    fig_depvar.tight_layout()


# PLOT THE IMPACT OF THE APPLIED UNCERTAINTY ON THE FINAL NORM OF POSITION DIFFERENCE
fig2, axs2 = plt.subplots(2, figsize=(6, 8))

for entry in range(1, entries):
    axs2[0].scatter(perturbations[:, entry]*rescale_x_axis_units, state_history_end_values_array[:, 0]*rescale_y_axis_units, label=fr'$\Delta {entry_names_x[entry]} {time_interval_names[0]}$',
                    marker=marker_styles[0][entry-1], facecolor=marker_styles[1][entry-1], edgecolor=marker_styles[2][entry-1])
    axs2[1].scatter(perturbations[:, entry]*rescale_x_axis_units, state_history_end_values_array[:, 1]*rescale_y_axis_units, label=fr'$\Delta {entry_names_x[entry]} {time_interval_names[0]}$',
                    marker=marker_styles[0][entry - 1], facecolor=marker_styles[1][entry - 1],
                    edgecolor=marker_styles[2][entry - 1])
axs2[0].set_ylabel(r'$|\Delta \mathbf{r}|' + fr' \, {time_interval_names[1]}$ {y_axis_measure_units[0]}', fontsize=y_label_size)
axs2[1].set_ylabel(r'$|\Delta \mathbf{v}|' + fr' \, {time_interval_names[1]}$ {y_axis_measure_units[1]}', fontsize=y_label_size)
fig2.supxlabel(fr'${xlabel} \, {time_interval_names[0]}$ ' + x_axis_measure_unit, fontsize=x_label_size)
for i in [0,1]:
    axs2[i].set_xlabel(fr'${xlabel} \, {time_interval_names[0]}$ ' + x_axis_measure_unit, fontsize=x_label_size)
    # Show the major grid lines with dark grey lines
    axs2[i].grid(visible=True, which='major', color='#666666', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    axs2[i].minorticks_on()
    axs2[i].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axs2[i].tick_params(axis='both', which='major', labelsize=ticks_size)
    if entries-1 > 1:
        axs2[i].legend()
fig2.suptitle(fr'Impact of ${xlabel} \, {time_interval_names[0]}$ on the final state (End state: {final_state_name})', fontsize=suptitle_size)
fig2.tight_layout()

# Plot of final eccentricity
if not stop_before_aerocapture:
    fig_ecc, axs_ecc = plt.subplots()

    for entry in range(1, entries):
        axs_ecc.scatter(perturbations[:, entry]*rescale_x_axis_units, state_history_end_values_array[:, 2],
                        label=fr'$\Delta {entry_names_x[entry]} {time_interval_names[0]}$',
                        marker=marker_styles[0][entry - 1], facecolor=marker_styles[1][entry - 1],
                        edgecolor=marker_styles[2][entry - 1])
    # axs_ecc.axhline(y=nominal_final_eccentricity, color='grey', linestyle='--')

    for i in [0]:  # change if you add other subplots
        # Show the major grid lines with dark grey lines
        axs_ecc.grid(visible=True, which='major', color='#666666', linestyle='-')
        # Show the minor grid lines with very faint and almost transparent grey lines
        axs_ecc.minorticks_on()
        axs_ecc.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        axs_ecc.tick_params(axis='both', which='major', labelsize=ticks_size)
        if entries-1 > 1:
            axs_ecc.legend()
    axs_ecc.set_ylabel(fr'Final eccentricity error  ${time_interval_names[1]}$', fontsize=y_label_size)
    axs_ecc.set_xlabel(fr'${xlabel} \, {time_interval_names[0]}$ ' + x_axis_measure_unit, fontsize=x_label_size)

    fig_ecc.suptitle(fr'Impact of ${xlabel} \, {time_interval_names[0]}$ on the final eccentricity (End state: {final_state_name})', fontsize=suptitle_size)
    fig_ecc.tight_layout()


# SHOW THE IMPACT OF THE APPLIED UNCERTAINTY ON THE FINAL NORM OF POSITION DIFFERENCE
# and
# SHOW UNCERTAINTY HISTOGRAMS AND RESULTING POSITION ERROR
# plt.show()


max_entries_pert = perturbations.shape[1] - 1

# PLOT THE IMPACT OF THE APPLIED UNCERTAINTY ON THE FINAL RSW COMPONENTS AND
# PLOT THE HISTOGRAM OF THE FINAL RSW COMPONENTS DISTRIBUTION
fig3_hist, ax3_hist = plt.subplots(3,2, figsize=(9, 6), sharex='col', constrained_layout=True)
fig3, axs3 = plt.subplots(2, figsize=(6, 10))

xlabel = ['', '']
ylabel = ['', '']
for entry in range(1,7):
    plot_nr = int((entry-1)/3)  # 0, 0, 0, 1, 1, 1
    coord_nr = (entry-1) % 3  # 0, 1, 2, 0, 1, 2

    # nice_c = entry%3-1
    # entry%3 : 1, 2, 0, 1, 2, 0

    plot_label = '\Delta ' + entry_names_x[min(coord_nr+1,max_entries_pert)] + ' ' + time_interval_names[0] + ' \\rightarrow ' + '\Delta ' + entry_names_y[entry] + ' ' + time_interval_names[1]

    x_component_name = entry_names_x[min(coord_nr+1,max_entries_pert)].split()[0]
    y_component_name = entry_names_y[entry].split()[0]

    # It leads to a label like: x1, x2, x3 (so it's like: comma, comma, no comma; comma, comma, no comma)
    add_y_comma = ',' if entry%3 != 0 else ''
    add_x_comma = ',' if entry%3 != 0 and max_entries_pert > 1 else ''

    xlabel[plot_nr] = xlabel[plot_nr] + '\Delta ' + x_component_name + add_x_comma if entry%3+1 <= max_entries_pert else xlabel[plot_nr]
    ylabel[plot_nr] = ylabel[plot_nr] + '\Delta ' + y_component_name + add_y_comma

    axs3[plot_nr].scatter(perturbations[:,min(coord_nr+1,max_entries_pert)]*rescale_x_axis_units,max_vals_in_rsw[:,entry-1]*rescale_y_axis_units, label=fr'${plot_label}$',
                 marker=marker_styles[0][coord_nr], facecolor=marker_styles[1][coord_nr], edgecolor=marker_styles[2][coord_nr])

    ax3_hist[coord_nr, plot_nr].hist(max_vals_in_rsw[:, entry - 1]*rescale_y_axis_units, 30, edgecolor='black', linewidth=1.2)
    ax3_hist[coord_nr, plot_nr].tick_params(axis='both', which='major', labelsize=ticks_size)
    ax3_hist[coord_nr, plot_nr].set_xlabel(fr'$\Delta {entry_names_y[entry]} {time_interval_names[1]}$ ' + y_axis_measure_units[plot_nr], fontsize=x_label_size)
    ax3_hist[coord_nr, plot_nr].tick_params(axis='x', labelbottom=True)

# axs3[0].set_aspect('equal', 'box')
for i in [0,1]:
    axs3[i].tick_params(axis='both', which='major', labelsize=ticks_size)
    # Show the major grid lines with dark grey lines
    axs3[i].grid(visible=True, which='major', color='#666666', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    axs3[i].minorticks_on()
    axs3[i].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axs3[i].set_xlabel(fr'${xlabel[i]} \; {time_interval_names[0]}$ ' + x_axis_measure_unit, fontsize=x_label_size)
    axs3[i].set_ylabel(fr'${ylabel[i]} \; {time_interval_names[1]}$ ' + y_axis_measure_units[i], fontsize=y_label_size)
    axs3[i].legend()

fig3.suptitle(fr'Impact of ${xlabel[0]} \; {time_interval_names[0]}$ on final RSW position and velocity {time_interval_names[0]}', fontsize=suptitle_size)

fig3_hist.suptitle(f'End Values Distribution (End state: {final_state_name})', fontsize=suptitle_size)
fig3_hist.supylabel('Occurrences', fontsize=common_y_label_size)

fig3_hist.tight_layout()
fig3.tight_layout()

# Plot the heat flux accuracy and the subsequent
if evaluated_arc == 1 or evaluated_arc == 12:
    nominal_heat_fluxes_np = np.loadtxt(data_path + 'heat_fluxes_history_' + str(0) + '.dat')
    nominal_heat_fluxes_epochs = nominal_heat_fluxes_np[:, 0]
    nominal_heat_fluxes_values = nominal_heat_fluxes_np[:, 1:]

    nominal_convective_heat_flux = nominal_heat_fluxes_values[:, 0]
    nominal_radiative_heat_flux = nominal_heat_fluxes_values[:, 1]

    nominal_total_wall_heat_flux = nominal_convective_heat_flux + nominal_radiative_heat_flux
    nominal_peak_heat_flux, nominal_total_heat_load = calculate_peak_hfx_and_heat_load(nominal_heat_fluxes_epochs, nominal_total_wall_heat_flux)
    nominal_tps_mass_fraction = calculate_tps_mass_fraction(nominal_total_heat_load)

    nominal_heat_flux_values = np.array([0., nominal_peak_heat_flux,
                                         nominal_total_heat_load, nominal_tps_mass_fraction])

    heat_fluxes_values_dictionary = dict()
    for run_number in range(1, number_of_runs):
        heat_fluxes_np = np.loadtxt(data_path + 'heat_fluxes_history_' + str(run_number) + '.dat')
        heat_fluxes_difference_wrt_nominal_case_np = np.loadtxt(
            data_path + 'heat_fluxes_difference_wrt_nominal_case_' + str(run_number) + '.dat')
        heat_fluxes_difference_values = heat_fluxes_difference_wrt_nominal_case_np[:, 1:]
        conv_hfx_diff = heat_fluxes_difference_values[:,0]
        rad_hfx_diff = heat_fluxes_difference_values[:,1]
        total_hfx_diff = conv_hfx_diff + rad_hfx_diff
        epochs_comparison_hfxes = heat_fluxes_difference_wrt_nominal_case_np[:, 0]

        heat_fluxes_epochs = heat_fluxes_np[:, 0]
        heat_fluxes_values = heat_fluxes_np[:, 1:]
        convective_heat_flux = heat_fluxes_values[:,0]
        radiative_heat_flux = heat_fluxes_values[:,1]

        heat_fluxes_history = dict(zip(heat_fluxes_epochs, heat_fluxes_values))
        hfxes_interpolator = interpolators.create_one_dimensional_vector_interpolator(
            heat_fluxes_history, interpolator_settings)

        total_wall_heat_flux = convective_heat_flux + radiative_heat_flux
        peak_heat_flux, total_heat_load = calculate_peak_hfx_and_heat_load(heat_fluxes_epochs, total_wall_heat_flux)
        tps_mass_fraction = calculate_tps_mass_fraction(total_heat_load)

        max_total_heat_flux_difference = max(abs(total_hfx_diff)) if max(abs(total_hfx_diff)) == max(total_hfx_diff) else -max(abs(total_hfx_diff))
        peak_heat_flux_difference = peak_heat_flux - nominal_peak_heat_flux
        heat_load_difference = total_heat_load - nominal_total_heat_load
        tps_mass_fraction_difference = tps_mass_fraction - nominal_tps_mass_fraction

        heat_fluxes_values_dictionary[run_number] = np.array([max_total_heat_flux_difference, tps_mass_fraction_difference, peak_heat_flux_difference, heat_load_difference])

    heat_fluxes_values_array = np.vstack(list(heat_fluxes_values_dictionary.values()))

    fig_hfx, ax_hfx = plt.subplots(2,2, figsize=(7,7))
    heat_fluxes_y_labels = np.array([['Max $q_w$ difference (kW/m$^2$)', 'TPS mass fraction difference'],
                                     ['Peak $q_w$ difference (kW/m$^2$)', 'Heat load difference (kJ/m$^2$)']])
    heat_fluxes_rescaling = np.array([1e-3, 1, 1e-3, 1e-3])
    for entry in range(1, entries):
        for i in range(2):
            for j in range(2):

                ax_hfx[i,j].scatter(perturbations[:, entry]*rescale_x_axis_units, heat_fluxes_values_array[:,int(2*i+1*j)] * heat_fluxes_rescaling[int(2*i+1*j)],
                                label=fr'$\Delta {entry_names_x[entry]} {time_interval_names[0]}$',
                                marker=marker_styles[0][entry - 1], facecolor=marker_styles[1][entry - 1],
                                edgecolor=marker_styles[2][entry - 1])
                if entries - 1 > 1:
                    ax_hfx[i, j].legend()
                if entry == 1:
                    # axs_ecc[i,j].axhline(y=nominal_heat_flux_values[int(2*i+1*j)], color='grey', linestyle='--')

                    # Show the major grid lines with dark grey lines
                    ax_hfx[i,j].grid(visible=True, which='major', color='#666666', linestyle='-')
                    # Show the minor grid lines with very faint and almost transparent grey lines
                    ax_hfx[i,j].minorticks_on()
                    ax_hfx[i,j].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
                    ax_hfx[i,j].tick_params(axis='both', which='major', labelsize=ticks_size)
                    ax_hfx[i,j].set_ylabel(fr'{heat_fluxes_y_labels[i,j]}', fontsize=y_label_size)
                    ax_hfx[i,j].set_xlabel(fr'${xlabel[0]} \, {time_interval_names[0]}$ ' + x_axis_measure_unit, fontsize=x_label_size)

    fig_hfx.suptitle(fr'Impact of ${xlabel[0]} \, $ on heat fluxes, heat load, and TPS mass fraction (End state: {final_state_name} {time_interval_names[0]})',
                     fontsize=suptitle_size)
    fig_hfx.tight_layout()


# SHOW THEM
plt.show()