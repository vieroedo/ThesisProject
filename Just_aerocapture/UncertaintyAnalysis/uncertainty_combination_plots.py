import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.linalg as LA
import numpy as np
import scipy as sp
import random
import os

from tudatpy.kernel import constants
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.astro import frame_conversion

from handle_functions import eccentricity_vector_from_cartesian_state
from UncertaintyStudy_GlobalVariables import *

choose_combination_case = 2
rescale_y_axis_units = 1  # keep to 1
rescale_x_axis_units = 1  # keep to 1

combination_dictionary = {
    'Initial_RSW_Position':[1,2,3], 'Initial_RSW_Velocity':[5,6,7],
    'Entry_RSW_Position':[9,10,11], 'Entry_RSW_Velocity':[13,14,15],
    'FinalOrbit_RSW_Position':[19,20,21], 'FinalOrbit_RSW_velocity':[23,24,25]
    }

combination_cases = list(combination_dictionary.keys())

if choose_combination_case > len(combination_cases):
    raise Exception('Case selected is unavailable. Select another case.')


uncertainties = list(uncertainties_dictionary.keys())  # list of uncertainty names
arcs_computed = list(uncertainties_dictionary.values())  # list of corresponding arcs

current_dir = os.path.dirname(__file__)

trajectory_parameters = np.loadtxt(current_dir + '/UncertaintyAnalysisData/trajectory_parameters.dat')  # [0, vel, fpa]
# evaluated_arc = trajectory_parameters[0]
interplanetary_arrival_velocity = trajectory_parameters[1]
atmospheric_entry_fpa = trajectory_parameters[2]

entry_names_y = ['R \;', 'S \;', 'W \;', 'V_R \;', 'V_S \;', 'V_W \;']
y_axis_measure_units = ['(m)', '(m/s)']

if choose_combination_case%2 == 0:
    position_combination = True
    velocity_combination = False
else:
    position_combination = False
    velocity_combination = True

combination_to_plot = combination_dictionary[combination_cases[choose_combination_case]]

# Set labels for plotting
if position_combination:
    x_axis_measure_unit = '(m)'
    if rescale_x_axis_units == 1e3:
        x_axis_measure_unit = '(mm)'
    elif rescale_x_axis_units == 1e-3:
        x_axis_measure_unit = '(km)'
    elif rescale_x_axis_units != 1:
        raise Exception('Invalid value for x axis rescaling. Allowed values are 1, 1e3, 1e-3')

    entry_names_x = ['R \;', 'S \;', 'W \;']
elif velocity_combination:
    x_axis_measure_unit = '(m/s)'
    if rescale_x_axis_units == 1e3:
        x_axis_measure_unit = '(mm/s)'
    elif rescale_x_axis_units == 1e-3:
        x_axis_measure_unit = '(km/s)'
    elif rescale_x_axis_units != 1:
        raise Exception('Invalid value for x axis rescaling. Allowed values are 1, 1e3, 1e-3')

    entry_names_x = ['V_R \;', 'V_S \;', 'V_W \;']
else:
    raise Exception('You selected a case that\'s not even a velocity or position combination. Check the code ;)')

# Set initial and final time
if choose_combination_case in [0,1]:
    arc_to_analyze = 0
    time_interval_names = ['(t_0)', '(t_E)']
elif choose_combination_case in [2,3]:
    arc_to_analyze = 1
    time_interval_names = ['(t_E)', '(t_F)']
elif choose_combination_case in [4,5]:
    arc_to_analyze = 12
    time_interval_names = ['(t_E)', '(t_1)']
else:
    raise Exception('You forgot to set names for time interval boundaries hehe (get your shit together ffs)')

if rescale_y_axis_units == 1e-3:
    y_axis_measure_units = ['(km)', '(km/s)']
elif rescale_y_axis_units == 1e3:
    y_axis_measure_units = ['(mm)', '(mm/s)']
elif rescale_y_axis_units != 1:
    raise Exception('Invalid value for y axis rescaling. Allowed values are 1, 1e3, 1e-3')


# PLOT THE IMPACT OF THE APPLIED UNCERTAINTY ON THE FINAL NORM OF POSITION DIFFERENCE
# Combine delta r and delta v plots of initial position error R, S, W: cases 1, 2, 3
fig_rv_norms, axs_rv_norms = plt.subplots(2, figsize=(6, 8))

fig_rsw, axs_rsw = plt.subplots(2, 3, figsize=(8, 8), sharey='row')
fig_ecc, axs_ecc = plt.subplots()

total_entries = len(combination_to_plot)
for entry_nr, uncertainty_to_analyze in enumerate(combination_to_plot):
    uncert_plot = entry_nr
    subdirectory = '/UncertaintyAnalysisData/' + uncertainties[uncertainty_to_analyze] + '/'  # it can be 0, 1, 2
    data_path = current_dir + subdirectory

    perturbations = np.loadtxt(current_dir + f'/UncertaintyAnalysisData/simulation_results_{uncertainties[uncertainty_to_analyze]}.dat')
    number_of_runs = len(perturbations[:,0]) + 1

    evaluated_arc = arcs_computed[uncertainty_to_analyze]


    # Extract nominal state history dictionary
    nmnl_state_history_filedata = np.loadtxt(data_path + 'state_history_0.dat')
    nmnl_state_history = dict(zip(nmnl_state_history_filedata[:, 0], nmnl_state_history_filedata[:, 1:]))
    nmnl_sh_interpolator = interpolators.create_one_dimensional_vector_interpolator(nmnl_state_history, interpolator_settings)

    nmnl_final_eccentricity = LA.norm(eccentricity_vector_from_cartesian_state(nmnl_state_history_filedata[-1, 1:]))

    state_history_end_values = dict()
    rsw_end_values = dict()

    for run_number in range(1, number_of_runs):
        state_history_np = np.loadtxt(data_path + 'state_history_' + str(run_number) + '.dat')
        state_difference_wrt_nominal_case_np = np.loadtxt(
            data_path + 'state_difference_wrt_nominal_case_' + str(run_number)
            + '.dat')
        spacecraft_final_state = state_history_np[-1, 1:]

        epochs_comparison_sh = state_difference_wrt_nominal_case_np[:, 0]
        # epochs_diff = (epochs_comparison_sh - epochs_comparison_sh[0]) / constants.JULIAN_DAY

        state_history = dict(zip(state_history_np[:, 0], state_history_np[:, 1:]))
        sh_interpolator = interpolators.create_one_dimensional_vector_interpolator(
            state_history, interpolator_settings)

        state_difference_wrt_nominal_case = dict(
            zip(state_difference_wrt_nominal_case_np[:, 0], state_difference_wrt_nominal_case_np[:, 1:]))
        cartesian_state_difference = state_difference_wrt_nominal_case_np[:, 1:]

        position_difference = state_difference_wrt_nominal_case_np[:, 1:4]
        velocity_difference = state_difference_wrt_nominal_case_np[:, 4:7]
        position_difference_norm = LA.norm(position_difference, axis=1)
        velocity_difference_norm = LA.norm(velocity_difference, axis=1)

        # Calculate altitude and speed difference
        # altitude_difference, speed_difference = np.zeros(len(epochs_comparison_sh)), np.zeros(len(epochs_comparison_sh))
        # for i, epoch in enumerate(epochs_comparison_sh):
            # nominal_state = nmnl_sh_interpolator.interpolate(epoch)
            # current_state = sh_interpolator.interpolate(epoch)
            # altitude_difference[i] = abs(LA.norm(nominal_state[0:3]) - LA.norm(current_state[0:3]))
            # speed_difference[i] = abs(LA.norm(nominal_state[3:6]) - LA.norm(current_state[3:6]))

        # Calculate state difference in the RSW frame
        rsw_state_difference = np.zeros(np.shape(cartesian_state_difference))
        for i, epoch in enumerate(state_difference_wrt_nominal_case_np[:, 0]):
            rsw_matrix = frame_conversion.inertial_to_rsw_rotation_matrix(sh_interpolator.interpolate(epoch))
            rsw_state_difference[i, 0:3] = rsw_matrix @ cartesian_state_difference[i, 0:3]
            rsw_state_difference[i, 3:6] = rsw_matrix @ cartesian_state_difference[i, 3:6]

        # Calculate eccentricity of the final state (usually t_E, t_F, t_1)
        final_eccentricity = LA.norm(eccentricity_vector_from_cartesian_state(spacecraft_final_state))

        rsw_end_values[run_number] = rsw_state_difference[-1, :]
        state_history_end_values[run_number] = np.array([position_difference_norm[-1], velocity_difference_norm[-1], final_eccentricity])

    # Reprocess values to plot them
    rsw_end_values_array = np.vstack(list(rsw_end_values.values()))
    state_history_end_values_array = np.vstack(list(state_history_end_values.values()))

    # Plot effect of the perturbation on norm of delta_r and delta_v
    axs_rv_norms[0].scatter(perturbations[:, 1], state_history_end_values_array[:, 0],
                            label=fr'$\Delta {entry_names_x[entry_nr]}$',
                            marker=marker_styles[0][entry_nr], facecolor=marker_styles[1][entry_nr],
                            edgecolor=marker_styles[2][entry_nr])
    axs_rv_norms[1].scatter(perturbations[:, 1], state_history_end_values_array[:, 1],
                            label=fr'$\Delta {entry_names_x[entry_nr]}$',
                            marker=marker_styles[0][entry_nr], facecolor=marker_styles[1][entry_nr],
                            edgecolor=marker_styles[2][entry_nr])

    xlabel_rsw = ['']
    ylabel_rsw = ['','']
    # Plot the effect on final RSW perturbations: plots the two vertical plots per uncertainty
    for entry in range(1, 7):
        plot_nr = int((entry - 1) / 3)  # 0, 0, 0, 1, 1, 1
        coord_nr = (entry - 1) % 3  # 0, 1, 2, 0, 1, 2

        # nice_c = entry%3-1
        # entry%3 : 1, 2, 0, 1, 2, 0

        plot_label = '\Delta ' + entry_names_x[uncert_plot] + ' ' + time_interval_names[0] + ' \\rightarrow ' + '\Delta ' + entry_names_y[entry-1] + ' ' + time_interval_names[1]

        x_component_name = entry_names_x[uncert_plot].split()[0]
        y_component_name = entry_names_y[entry-1].split()[0]

        # It leads to a label like: x1, x2, x3 (so it's like: comma, comma, no comma; comma, comma, no comma)
        add_y_comma = ',' if entry % 3 != 0 else ''

        xlabel_rsw = '\Delta ' + x_component_name
        ylabel_rsw[plot_nr] = ylabel_rsw[plot_nr] + '\Delta ' + y_component_name + add_y_comma

        axs_rsw[plot_nr, uncert_plot].scatter(perturbations[:, 1] * rescale_x_axis_units, rsw_end_values_array[:, entry - 1] * rescale_y_axis_units,
                                              label=fr'${plot_label}$',
                                              marker=marker_styles[0][coord_nr], facecolor=marker_styles[1][coord_nr],
                                              edgecolor=marker_styles[2][coord_nr])
    for i in [0, 1]:
        axs_rsw[i, uncert_plot].set_xlabel(fr'${xlabel_rsw} \; {time_interval_names[0]}$ ' + x_axis_measure_unit,
                                           fontsize=x_label_size)
        if entry_nr==0:
            axs_rsw[i, 0].set_ylabel(fr'${ylabel_rsw[i]} \; {time_interval_names[1]}$ ' + y_axis_measure_units[i],
                                     fontsize=y_label_size)
            fig_rsw.suptitle(fr'Impact of ${xlabel_rsw} \; {time_interval_names[0]}$ on final RSW position and velocity {time_interval_names[0]}',
                fontsize=suptitle_size)


    # Plot the effect on the eccentricity
    # for entry in range(1, total_entries):
    #     entry_for_labels = (uncertainty_to_analyze+1)%3+1 if uncertainty_to_analyze in [5,6,7] else 0
    #     entry_for_labels = (uncertainty_to_analyze - 1) % 3 + 1 if uncertainty_to_analyze in [13, 14, 15] else 0
    #     entry_for_labels = uncertainty_to_analyze % 3 + 1 if uncertainty_to_analyze in [9,10,11] else 0
    axs_ecc.scatter(perturbations[:, 1]*rescale_x_axis_units, state_history_end_values_array[:, 2],
                    label=fr'$\Delta {entry_names_x[entry_nr]} {time_interval_names[0]}$',
                    marker=marker_styles[0][entry_nr], facecolor=marker_styles[1][entry_nr],
                    edgecolor=marker_styles[2][entry_nr])
    if entry_nr==0:
        axs_ecc.axhline(y=nmnl_final_eccentricity, color='grey', linestyle='--')

xlabel = ''
for entry in range(total_entries):
    x_component_name = entry_names_x[entry].split()[0]
    add_x_comma = ',' if entry < len(combination_to_plot)-1 else ''
    xlabel = xlabel + '\Delta ' + x_component_name + add_x_comma  # if entry < total_entries else xlabel

axs_rv_norms[0].set_ylabel(r'$|\Delta \mathbf{r}| \,' + rf'{time_interval_names[1]}$ (m)', fontsize=y_label_size)
axs_rv_norms[1].set_ylabel(r'$|\Delta \mathbf{v}| \,' + rf'{time_interval_names[1]}$ (m/s)', fontsize=y_label_size)
fig_rv_norms.supxlabel(fr'${xlabel} \, ' + rf'{time_interval_names[0]}$ ' + x_axis_measure_unit, fontsize=x_label_size)
for i in [0,1]:
    axs_rv_norms[i].set_xlabel(fr'${xlabel} \, (t_0)$ ' + x_axis_measure_unit, fontsize=x_label_size)
    # Show the major grid lines with dark grey lines
    axs_rv_norms[i].grid(visible=True, which='major', color='#666666', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    axs_rv_norms[i].minorticks_on()
    axs_rv_norms[i].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axs_rv_norms[i].tick_params(axis='both', which='major', labelsize=ticks_size)
    axs_rv_norms[i].legend()
fig_rv_norms.suptitle(fr'Impact of ${xlabel} \, {time_interval_names[0]} $ on the final state ${time_interval_names[1]}$', fontsize=suptitle_size)
fig_rv_norms.tight_layout()


for i in[0,1]:
    for j in range(axs_rsw.shape[1]):
        axs_rsw[i, j].tick_params(axis='both', which='major', labelsize=ticks_size)
        # Show the major grid lines with dark grey lines
        axs_rsw[i, j].grid(visible=True, which='major', color='#666666', linestyle='-')
        # Show the minor grid lines with very faint and almost transparent grey lines
        axs_rsw[i, j].minorticks_on()
        axs_rsw[i, j].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        if arc_to_analyze == 1 and position_combination:
            if j == 0:
                axs_rsw[i, j].set_xlim(-20000 * rescale_x_axis_units, 20000 * rescale_x_axis_units)
            else:
                axs_rsw[i, j].set_xlim(-10000 * rescale_x_axis_units, 10000 * rescale_x_axis_units)
        axs_rsw[i, j].legend()
fig_rsw.tight_layout()


# xlabel = ''
# for entry in range(1, 4):
#     x_component_name = entry_names_x[entry].split()[0]
#     add_x_comma = ',' if entry < 4-1 else ''
#     xlabel = xlabel + '\Delta ' + x_component_name + add_x_comma if entry < 4 else xlabel

# axs_ecc.axhline(y=nmnl_final_eccentricity, color='grey', linestyle='--')

for i in [0]:  # change if you add other subplots
    # Show the major grid lines with dark grey lines
    axs_ecc.grid(visible=True, which='major', color='#666666', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    axs_ecc.minorticks_on()
    axs_ecc.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axs_ecc.tick_params(axis='both', which='major', labelsize=ticks_size)
    axs_ecc.legend()
axs_ecc.set_ylabel(fr'Final eccentricity  ${time_interval_names[1]}$', fontsize=y_label_size)
axs_ecc.set_xlabel(fr'${xlabel} \, {time_interval_names[0]}$ ' + x_axis_measure_unit, fontsize=x_label_size)

fig_ecc.suptitle(fr'Impact of ${xlabel} \, {time_interval_names[0]}$ on the final eccentricity (End state: lol)', fontsize=suptitle_size)
fig_ecc.tight_layout()


fig_depvar, ax_depvar = plt.subplots(1, 2, figsize=(8, 6))
# time_interval_names = ['(t_E)', '(t_F)']
# uncertainties_to_analyze = [9,10,11]

# time_interval_names = ['(t_0)', '(t_E)']
# uncertainties_to_analyze = [1,2,3]

for entry_nr, uncertainty_to_analyze in enumerate(combination_to_plot):
    uncert_plot = entry_nr

    subdirectory = '/UncertaintyAnalysisData/' + uncertainties[uncertainty_to_analyze] + '/'  # it can be 0, 1, 2
    data_path = current_dir + subdirectory

    perturbations = np.loadtxt(
        current_dir + f'/UncertaintyAnalysisData/simulation_results_{uncertainties[uncertainty_to_analyze]}.dat')
    number_of_runs = len(perturbations[:, 0]) + 1

    # evaluated_arc = arcs_computed[uncertainty_to_analyze]


    nominal_dependent_variable_history_np = np.loadtxt(data_path + 'dependent_variable_history_0.dat')
    nominal_dependent_variable_history = dict(
        zip(nominal_dependent_variable_history_np[:, 0], nominal_dependent_variable_history_np[:, 1:]))
    nominal_depvarh_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        nominal_dependent_variable_history, interpolator_settings)

    dependent_variable_end_values = dict()
    # heat_fluxes_values = dict()
    for run_number in range(1, number_of_runs):
        dependent_variable_history_np = np.loadtxt(data_path + 'dependent_variable_history_' + str(run_number) + '.dat')
        dependent_variable_difference_wrt_nominal_case_np = np.loadtxt(
            data_path + 'dependent_variable_difference_wrt_nominal_case_' + str(run_number)
            + '.dat')
        epochs_comparison_depvar = dependent_variable_difference_wrt_nominal_case_np[:, 0]

        dependent_variable_history = dict(
            zip(dependent_variable_history_np[:, 0], dependent_variable_history_np[:, 1:]))
        depvarh_interpolator = interpolators.create_one_dimensional_vector_interpolator(
            dependent_variable_history, interpolator_settings)

        nominal_end_dependent_variables = nominal_depvarh_interpolator.interpolate(epochs_comparison_depvar[-1])
        current_end_dependent_variables = depvarh_interpolator.interpolate(epochs_comparison_depvar[-1])

        final_fpa_difference = (nominal_end_dependent_variables[5] - current_end_dependent_variables[5])
        final_airspeed_difference = (nominal_end_dependent_variables[6] - current_end_dependent_variables[6])

        # dependent_variable_end_values[run_number] = np.array([fpa_difference[-1], airspeed_difference[-1]])
        dependent_variable_end_values[run_number] = np.array(
            [final_fpa_difference, final_airspeed_difference])

    dependent_variable_end_values_array = np.vstack(list(dependent_variable_end_values.values()))


    marker = marker_styles[0][entry_nr]
    facecolor = marker_styles[1][entry_nr]
    edgecolor = marker_styles[2][entry_nr]
    ax_depvar[0].scatter(perturbations[:, 1]*rescale_x_axis_units, dependent_variable_end_values_array[:,0],
                 label=fr'$\Delta {entry_names_x[entry_nr]} {time_interval_names[0]}$', marker=marker, facecolor=facecolor, edgecolor=edgecolor)
    ax_depvar[1].scatter(perturbations[:, 1]*rescale_x_axis_units, dependent_variable_end_values_array[:, 1],
                 label=fr'$\Delta {entry_names_x[entry_nr]} {time_interval_names[0]}$', marker=marker, facecolor=facecolor,edgecolor=edgecolor)


xlabel = ''
for entry in range(total_entries):
    x_component_name = entry_names_x[entry].split()[0]
    add_x_comma = ',' if entry < total_entries - 1 else ''
    xlabel = xlabel + '\Delta ' + x_component_name + add_x_comma


ax_depvar[0].set_ylabel(fr'$\gamma \, {time_interval_names[1]}$ (°)', fontsize=y_label_size)
ax_depvar[1].set_ylabel(fr'$Airspeed \, {time_interval_names[1]}$ (m/s)', fontsize=y_label_size)
fig_depvar.supxlabel(fr'${xlabel} \, {time_interval_names[0]}$ ' + x_axis_measure_unit, fontsize=x_label_size)
for i in [0,1]:
    ax_depvar[i].set_xlabel(fr'${xlabel} \, {time_interval_names[0]}$ ' + x_axis_measure_unit, fontsize=x_label_size)
    ax_depvar[i].tick_params(axis='both', which='major', labelsize=ticks_size)
    # Show the major grid lines with dark grey lines
    ax_depvar[i].grid(visible=True, which='major', color='#666666', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    ax_depvar[i].minorticks_on()
    ax_depvar[i].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    if position_combination:
        ax_depvar[i].set_xlim(-10000*rescale_x_axis_units,10000*rescale_x_axis_units)
    ax_depvar[i].legend()
fig_depvar.suptitle(fr'Impact of ${xlabel} \, {time_interval_names[0]} $ on the final $\gamma$ and Airspeed ${time_interval_names[1]}$)', fontsize=suptitle_size)
fig_depvar.tight_layout()


plt.show()