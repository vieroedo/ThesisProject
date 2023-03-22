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

# Create interpolator settings
interpolator_settings = interpolators.lagrange_interpolation(
    8, boundary_interpolation=interpolators.use_boundary_value)

# Marker styles and cmap
cmap = plt.get_cmap('tab10')
marker_styles = (['D',     'o',     'o'],
                 [cmap(5), cmap(0), 'none'],
                 [cmap(5), cmap(0), cmap(1)])

# Font sizes
ticks_size = 12
x_label_size, y_label_size = 14, 14
common_y_label_size = 16
suptitle_size = 18

uncertainties_dictionary = {
    'InitialPosition': 0, 'InitialPosition_R': 0, 'InitialPosition_S': 0, 'InitialPosition_W': 0,
    'InitialVelocity': 0, 'InitialVelocity_R': 0, 'InitialVelocity_S': 0, 'InitialVelocity_W': 0,

    'InitialPosition_Entry': 1, 'InitialPosition_R_Entry': 1, 'InitialPosition_S_Entry': 1, 'InitialPosition_W_Entry': 1,
    'InitialVelocity_Entry': 1, 'InitialVelocity_R_Entry': 1, 'InitialVelocity_S_Entry': 1, 'InitialVelocity_W_Entry': 1,
    'EntryFlightPathAngle': 1, 'EntryVelocity': 1,

    'FinalOrbit_InitialPosition_Entry': 12, 'FinalOrbit_InitialPosition_R_Entry': 12, 'FinalOrbit_InitialPosition_S_Entry': 12, 'FinalOrbit_InitialPosition_W_Entry': 12,
    'FinalOrbit_InitialVelocity_Entry': 12, 'FinalOrbit_InitialVelocity_R_Entry': 12, 'FinalOrbit_InitialVelocity_S_Entry': 12, 'FinalOrbit_InitialVelocity_W_Entry': 12,
    'FinalOrbit_EntryFlightPathAngle': 12, 'FinalOrbit_EntryVelocity': 12
}

uncertainties = list(uncertainties_dictionary.keys())  # list of uncertainty names
arcs_computed = list(uncertainties_dictionary.values())  # list of corresponding arcs

current_dir = os.path.dirname(__file__)

trajectory_parameters = np.loadtxt(current_dir + '/UncertaintyAnalysis/trajectory_parameters.dat')  # [0, vel, fpa]
# evaluated_arc = trajectory_parameters[0]
interplanetary_arrival_velocity = trajectory_parameters[1]
atmospheric_entry_fpa = trajectory_parameters[2]

entry_names_y = [0, 'R \;', 'S \;', 'W \;', 'V_R \;', 'V_S \;', 'V_W \;']
y_axis_measure_units = ['(m)', '(m/s)']

# Combine delta r and delta v plots of initial position error R, S, W: cases 1, 2, 3

# PLOT THE IMPACT OF THE APPLIED UNCERTAINTY ON THE FINAL NORM OF POSITION DIFFERENCE
fig2, axs2 = plt.subplots(2, figsize=(6, 8))

# Set labels for plotting
temp_base = [0, 'R \; (t_0)', 'S \; (t_0)', 'W \; (t_0)']
x_axis_measure_unit = '(m)'

for uncertainty_to_analyze in [1, 2, 3]:
    subdirectory = '/UncertaintyAnalysis/' + uncertainties[uncertainty_to_analyze] + '/'  # it can be 0, 1, 2
    data_path = current_dir + subdirectory

    perturbations = np.loadtxt(current_dir + f'/UncertaintyAnalysis/simulation_results_{uncertainties[uncertainty_to_analyze]}.dat')
    number_of_runs = len(perturbations[:,0]) + 1

    evaluated_arc = arcs_computed[uncertainty_to_analyze]

    entries = perturbations.shape[1]

    # Extract nominal state history dictionary
    nominal_state_history_filedata = np.loadtxt(data_path + 'state_history_0.dat')
    nominal_state_history = dict(zip(nominal_state_history_filedata[:, 0], nominal_state_history_filedata[:, 1:]))
    nominal_sh_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        nominal_state_history, interpolator_settings)

    nominal_final_eccentricity = LA.norm(eccentricity_vector_from_cartesian_state(nominal_state_history_filedata[-1, 1:]))

    state_history_end_values = dict()
    rsw_end_values = dict()

    for run_number in range(1, number_of_runs):
        state_history_np = np.loadtxt(data_path + 'state_history_' + str(run_number) + '.dat')
        state_difference_wrt_nominal_case_np = np.loadtxt(
            data_path + 'state_difference_wrt_nominal_case_' + str(run_number)
            + '.dat')
        spacecraft_final_state = state_history_np[-1, 1:]

        epochs_comparison_sh = state_difference_wrt_nominal_case_np[:, 0]
        epochs_diff = (epochs_comparison_sh - epochs_comparison_sh[0]) / constants.JULIAN_DAY

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

        altitude_difference, speed_difference = np.zeros(len(epochs_comparison_sh)), np.zeros(len(epochs_comparison_sh))
        # altitude_difference_debug = np.zeros(len(epochs_comparison_sh))
        for i, epoch in enumerate(epochs_comparison_sh):
            nominal_state = nominal_sh_interpolator.interpolate(epoch)
            current_state = sh_interpolator.interpolate(epoch)

            altitude_difference[i] = abs(LA.norm(nominal_state[0:3]) - LA.norm(current_state[0:3]))
            # altitude_difference_debug[i] = LA.norm(np.dot(cartesian_state_difference[i,0:3], Util.unit_vector(current_state[0:3])))
            speed_difference[i] = abs(LA.norm(nominal_state[3:6]) - LA.norm(current_state[3:6]))

        rsw_state_difference = np.zeros(np.shape(cartesian_state_difference))
        for i, epoch in enumerate(state_difference_wrt_nominal_case_np[:, 0]):
            rsw_matrix = frame_conversion.inertial_to_rsw_rotation_matrix(sh_interpolator.interpolate(epoch))
            rsw_state_difference[i, 0:3] = rsw_matrix @ cartesian_state_difference[i, 0:3]
            rsw_state_difference[i, 3:6] = rsw_matrix @ cartesian_state_difference[i, 3:6]

        rsw_position_difference = rsw_state_difference[:, 0:3]
        rsw_velocity_difference = rsw_state_difference[:, 3:6]

        final_eccentricity = LA.norm(eccentricity_vector_from_cartesian_state(spacecraft_final_state))

        rsw_end_values[run_number] = rsw_state_difference[-1, :]
        state_history_end_values[run_number] = np.array(
            [position_difference_norm[-1], velocity_difference_norm[-1], final_eccentricity])

    # Reprocess values to plot them
    max_vals_in_rsw = np.vstack(list(rsw_end_values.values()))
    state_history_end_values_array = np.vstack(list(state_history_end_values.values()))

    entry_names_x = [0, temp_base[uncertainty_to_analyze]]

    entry = uncertainty_to_analyze
    axs2[0].scatter(perturbations[:, 1], state_history_end_values_array[:, 0],
                    label=fr'$\Delta {entry_names_x[1]}$',
                    marker=marker_styles[0][entry - 1], facecolor=marker_styles[1][entry - 1],
                    edgecolor=marker_styles[2][entry - 1])
    axs2[1].scatter(perturbations[:, 1], state_history_end_values_array[:, 1],
                    label=fr'$\Delta {entry_names_x[1]}$',
                    marker=marker_styles[0][entry - 1], facecolor=marker_styles[1][entry - 1],
                    edgecolor=marker_styles[2][entry - 1])

xlabel = ''
for entry in range(1, 4):
    x_component_name = temp_base[entry].split()[0]
    add_x_comma = ',' if entry < 4-1 else ''
    xlabel = xlabel + '\Delta ' + x_component_name + add_x_comma if entry < 4 else xlabel

axs2[0].set_ylabel(r'$|\Delta \mathbf{r}| \,(t_E)$ (m)', fontsize=y_label_size)
axs2[1].set_ylabel(r'$|\Delta \mathbf{v}| \,(t_E)$ (m/s)', fontsize=y_label_size)
fig2.supxlabel(fr'${xlabel} \, (t_0)$ ' + x_axis_measure_unit, fontsize=x_label_size)
for i in [0,1]:
    axs2[i].set_xlabel(fr'${xlabel} \, (t_0)$ ' + x_axis_measure_unit, fontsize=x_label_size)
    # Show the major grid lines with dark grey lines
    axs2[i].grid(visible=True, which='major', color='#666666', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    axs2[i].minorticks_on()
    axs2[i].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axs2[i].tick_params(axis='both', which='major', labelsize=ticks_size)
    axs2[i].legend()
fig2.suptitle(fr'Impact of ${xlabel} \, (t_0) $ on the final state (End state: Atmospheric Entry)', fontsize=suptitle_size)
fig2.tight_layout()


fig3, axs3 = plt.subplots(2,3, figsize=(8, 8), sharey='row')
fig_ecc, axs_ecc = plt.subplots()
# rescale_units = 1e-3
rescale_y_units = 1
rescale_x_units = 1
time_interval_names = ['(t_E)', '(t_F)']
# uncertainties_to_analyze = [9,10,11]
uncertainties_to_analyze = [13,14,15]


# time_interval_names = ['(t_0)', '(t_E)']
# uncertainties_to_analyze = [1,2,3]
# uncertainties_to_analyze = [5,6,7]


for uncertainty_to_analyze in uncertainties_to_analyze:
    uncert_plot = uncertainty_to_analyze - uncertainties_to_analyze[0]

    if uncertainty_to_analyze in [0,1,2,3,8,9,10,11]:
        entry_names_x = [0, 'R \;', 'S \;', 'W \;']
        if rescale_x_units != 1:
            x_axis_measure_unit = 'km'
    if rescale_y_units != 1:
        y_axis_measure_units = ['km', 'km/s']

    if uncertainty_to_analyze in [4,5,6,7,12,13,14,15]:
        entry_names_x = [0, 'V_R \;', 'V_S \;', 'V_W \;']
        if rescale_x_units != 1:
            x_axis_measure_unit = 'mm/s'

    xlabel = ['', '']
    ylabel = ['', '']

    subdirectory = '/UncertaintyAnalysis/' + uncertainties[uncertainty_to_analyze] + '/'
    data_path = current_dir + subdirectory

    perturbations = np.loadtxt(
        current_dir + f'/UncertaintyAnalysis/simulation_results_{uncertainties[uncertainty_to_analyze]}.dat')
    number_of_runs = len(perturbations[:, 0]) + 1

    evaluated_arc = arcs_computed[uncertainty_to_analyze]

    entries = perturbations.shape[1]

    # Extract nominal state history dictionary
    nominal_state_history_filedata = np.loadtxt(data_path + 'state_history_0.dat')
    nominal_state_history = dict(zip(nominal_state_history_filedata[:, 0], nominal_state_history_filedata[:, 1:]))
    nominal_sh_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        nominal_state_history, interpolator_settings)

    nominal_final_eccentricity = LA.norm(
        eccentricity_vector_from_cartesian_state(nominal_state_history_filedata[-1, 1:]))

    state_history_end_values = dict()
    rsw_end_values = dict()

    for run_number in range(1, number_of_runs):
        state_history_np = np.loadtxt(data_path + 'state_history_' + str(run_number) + '.dat')
        state_difference_wrt_nominal_case_np = np.loadtxt(
            data_path + 'state_difference_wrt_nominal_case_' + str(run_number)
            + '.dat')
        spacecraft_final_state = state_history_np[-1, 1:]

        epochs_comparison_sh = state_difference_wrt_nominal_case_np[:, 0]
        epochs_diff = (epochs_comparison_sh - epochs_comparison_sh[0]) / constants.JULIAN_DAY

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

        altitude_difference, speed_difference = np.zeros(len(epochs_comparison_sh)), np.zeros(len(epochs_comparison_sh))
        # altitude_difference_debug = np.zeros(len(epochs_comparison_sh))
        for i, epoch in enumerate(epochs_comparison_sh):
            nominal_state = nominal_sh_interpolator.interpolate(epoch)
            current_state = sh_interpolator.interpolate(epoch)

            altitude_difference[i] = abs(LA.norm(nominal_state[0:3]) - LA.norm(current_state[0:3]))
            # altitude_difference_debug[i] = LA.norm(np.dot(cartesian_state_difference[i,0:3], Util.unit_vector(current_state[0:3])))
            speed_difference[i] = abs(LA.norm(nominal_state[3:6]) - LA.norm(current_state[3:6]))

        rsw_state_difference = np.zeros(np.shape(cartesian_state_difference))
        for i, epoch in enumerate(state_difference_wrt_nominal_case_np[:, 0]):
            rsw_matrix = frame_conversion.inertial_to_rsw_rotation_matrix(sh_interpolator.interpolate(epoch))
            rsw_state_difference[i, 0:3] = rsw_matrix @ cartesian_state_difference[i, 0:3]
            rsw_state_difference[i, 3:6] = rsw_matrix @ cartesian_state_difference[i, 3:6]

        rsw_position_difference = rsw_state_difference[:, 0:3]
        rsw_velocity_difference = rsw_state_difference[:, 3:6]

        final_eccentricity = LA.norm(eccentricity_vector_from_cartesian_state(spacecraft_final_state))

        rsw_end_values[run_number] = rsw_state_difference[-1, :]
        state_history_end_values[run_number] = np.array(
            [position_difference_norm[-1], velocity_difference_norm[-1], final_eccentricity])

    # Reprocess values to plot them
    max_vals_in_rsw = np.vstack(list(rsw_end_values.values()))
    state_history_end_values_array = np.vstack(list(state_history_end_values.values()))
    for entry in range(1, 7):
        plot_nr = int((entry - 1) / 3)  # 0, 0, 0, 1, 1, 1
        coord_nr = (entry - 1) % 3  # 0, 1, 2, 0, 1, 2

        # nice_c = entry%3-1
        # entry%3 : 1, 2, 0, 1, 2, 0

        plot_label = '\Delta ' + entry_names_x[uncert_plot+1] + ' ' + time_interval_names[
            0] + ' \\rightarrow ' + '\Delta ' + entry_names_y[entry] + ' ' + time_interval_names[1]

        x_component_name = entry_names_x[uncert_plot+1].split()[0]
        y_component_name = entry_names_y[entry].split()[0]

        # It leads to a label like: x1, x2, x3 (so it's like: comma, comma, no comma; comma, comma, no comma)
        add_y_comma = ',' if entry % 3 != 0 else ''
        add_x_comma = ',' if entry % 3 != 0 and 1 > 1 else ''

        xlabel[plot_nr] = xlabel[
                              plot_nr] + '\Delta ' + x_component_name + add_x_comma if entry % 3 + 1 <= 1 else \
        xlabel[plot_nr]
        ylabel[plot_nr] = ylabel[plot_nr] + '\Delta ' + y_component_name + add_y_comma

        axs3[plot_nr, uncert_plot].scatter(perturbations[:, min(coord_nr + 1, 1)]*rescale_x_units, max_vals_in_rsw[:, entry - 1]*rescale_y_units,
                              label=fr'${plot_label}$',
                              marker=marker_styles[0][coord_nr], facecolor=marker_styles[1][coord_nr],
                              edgecolor=marker_styles[2][coord_nr])
    for i in [0, 1]:
        axs3[i, uncert_plot].set_xlabel(fr'${xlabel[i]} \; {time_interval_names[0]}$ ' + x_axis_measure_unit,
                              fontsize=x_label_size)

    for entry in range(1, entries):
        entry_for_labels = (uncertainty_to_analyze+1)%3+1 if uncertainty_to_analyze in [5,6,7] else 0
        entry_for_labels = (uncertainty_to_analyze - 1) % 3 + 1 if uncertainty_to_analyze in [13, 14, 15] else 0
        axs_ecc.scatter(perturbations[:, entry]*rescale_x_units, state_history_end_values_array[:, 2]*rescale_y_units,
                        label=fr'$\Delta {entry_names_x[entry_for_labels]} {time_interval_names[0]}$',
                        marker=marker_styles[0][entry_for_labels - 1], facecolor=marker_styles[1][entry_for_labels - 1],
                        edgecolor=marker_styles[2][entry_for_labels - 1])
for i in[0,1]:
    axs3[i, 0].set_ylabel(fr'${ylabel[i]} \; {time_interval_names[1]}$ ' + y_axis_measure_units[i],
                          fontsize=y_label_size)
    for j in range(axs3.shape[1]):
        axs3[i,j].tick_params(axis='both', which='major', labelsize=ticks_size)
        # Show the major grid lines with dark grey lines
        axs3[i,j].grid(visible=True, which='major', color='#666666', linestyle='-')
        # Show the minor grid lines with very faint and almost transparent grey lines
        axs3[i,j].minorticks_on()
        axs3[i,j].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        if uncertainties_to_analyze == [9, 10, 11]:
            if j == 0:
                axs3[i, j].set_xlim(-20000 * rescale_x_units, 20000 * rescale_x_units)
            else:
                axs3[i,j].set_xlim(-10000*rescale_x_units,10000*rescale_x_units)
        # axs3[i, j].set_xlim(-10000 * rescale_units, 10000 * rescale_units)

        axs3[i,j].legend()

fig3.suptitle(
    fr'Impact of ${xlabel[0]} \; {time_interval_names[0]}$ on final RSW position and velocity {time_interval_names[0]}',
    fontsize=suptitle_size)
fig3.tight_layout()

xlabel = ''
for entry in range(1, 4):
    x_component_name = entry_names_x[entry].split()[0]
    add_x_comma = ',' if entry < 4-1 else ''
    xlabel = xlabel + '\Delta ' + x_component_name + add_x_comma if entry < 4 else xlabel

axs_ecc.axhline(y=nominal_final_eccentricity, color='grey', linestyle='--')

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

time_interval_names = ['(t_0)', '(t_E)']
uncertainties_to_analyze = [1,2,3]

for uncertainty_to_analyze in uncertainties_to_analyze:
    uncert_plot = uncertainty_to_analyze - uncertainties_to_analyze[0]
    entry_names_x = [0, 'R \;', 'S \;', 'W \;']
    # xlabel = ['', '']
    ylabel = ['', '']

    xlabel = ''
    for entry in range(1, 4):
        x_component_name = entry_names_x[entry].split()[0]
        add_x_comma = ',' if entry < 4 - 1 else ''
        xlabel = xlabel + '\Delta ' + x_component_name + add_x_comma if entry < 4 else xlabel

    subdirectory = '/UncertaintyAnalysis/' + uncertainties[uncertainty_to_analyze] + '/'  # it can be 0, 1, 2
    data_path = current_dir + subdirectory

    perturbations = np.loadtxt(
        current_dir + f'/UncertaintyAnalysis/simulation_results_{uncertainties[uncertainty_to_analyze]}.dat')
    number_of_runs = len(perturbations[:, 0]) + 1

    evaluated_arc = arcs_computed[uncertainty_to_analyze]

    entries = perturbations.shape[1]

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

    # for entry in range(1, entries):
    entry = uncert_plot+1
    marker = marker_styles[0][entry-1]
    facecolor = marker_styles[1][entry-1]
    edgecolor = marker_styles[2][entry-1]
    ax_depvar[0].scatter(perturbations[:, 1], dependent_variable_end_values_array[:,0],
                 label=fr'$\Delta {entry_names_x[entry]} {time_interval_names[0]}$', marker=marker, facecolor=facecolor, edgecolor=edgecolor)
    ax_depvar[1].scatter(perturbations[:, 1], dependent_variable_end_values_array[:, 1],
                 label=fr'$\Delta {entry_names_x[entry]} {time_interval_names[0]}$', marker=marker, facecolor=facecolor,edgecolor=edgecolor)

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
    ax_depvar[i].set_xlim(-10000,10000)
    ax_depvar[i].legend()
fig_depvar.suptitle(fr'Impact of ${xlabel} \, {time_interval_names[0]} $ on the final $\gamma$ and Airspeed {time_interval_names[1]})', fontsize=suptitle_size)
fig_depvar.tight_layout()

plt.show()