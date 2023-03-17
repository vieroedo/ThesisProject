import matplotlib.pyplot as plt
import numpy.linalg as LA
import numpy as np
import scipy as sp
import random
import os

# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.astro import frame_conversion, element_conversion

# Problem-specific imports
from CapsuleEntryUtilities import compare_models
from handle_functions import eccentricity_vector_from_cartesian_state

# SET PARAMETERS
uncertainty_to_analyze = 5  # From 0 to 17

# uncertainties = ['EarthEph', 'SRP', 'InitialState', 'InitialState_1', 'InitialState_2', 'InitialState_3']
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

# Font sizes
ticks_size = 12
x_label_size, y_label_size = 14, 14
common_y_label_size = 16
suptitle_size = 18

# Marker styles and cmap
cmap = plt.get_cmap('tab10')
# marker_styles = (['o', 'o', 'v', 'v', 'D', 'D'],
#                  [cmap(0), 'none', 'none', cmap(3), 'none', cmap(5)],
#                  [cmap(0), cmap(1), cmap(2), cmap(3), cmap(4), cmap(5)])
marker_styles = (['D', 'o', 'o'],
                 [cmap(5), cmap(0), 'none'],
                 [cmap(5), cmap(0), cmap(1)])


current_dir = os.path.dirname(__file__)
subdirectory = '/UncertaintyAnalysis/' + uncertainties[uncertainty_to_analyze] + '/'  # it can be 0, 1, 2
data_path = current_dir + subdirectory

perturbations = np.loadtxt(current_dir + f'/UncertaintyAnalysis/simulation_results_{uncertainties[uncertainty_to_analyze]}.dat')
number_of_runs = len(perturbations[:,0]) + 1

trajectory_parameters = np.loadtxt(current_dir + '/UncertaintyAnalysis/trajectory_parameters.dat')  # [0, vel, fpa]
# evaluated_arc = trajectory_parameters[0]
interplanetary_arrival_velocity = trajectory_parameters[1]
atmospheric_entry_fpa = trajectory_parameters[2]

evaluated_arc = arcs_computed[uncertainty_to_analyze]

final_state_names = ['Atmospheric Entry', 'Atmospheric Exit', 'Final Orbit']
if evaluated_arc == 0:
    final_state_name = final_state_names[0]
elif evaluated_arc == 1:
    final_state_name = final_state_names[1]
elif evaluated_arc == 12 or evaluated_arc == -1:
    final_state_name = final_state_names[2]
else:
    raise Exception('The propagated arc cannot yet be shown! Update the code.')

# if uncertainties[uncertainty_to_analyze] == uncertainties[8] or uncertainties[uncertainty_to_analyze] == uncertainties[9]:
#     evaluated_arc = 1

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
if uncertainty_to_analyze == 0:
    entry_names_x = [0, 'R \; (t_0)', 'S \; (t_0)', 'W \; (t_0)']
    x_axis_measure_unit = '(m)'
elif 0<uncertainty_to_analyze<4:
    temp_base = [0, 'R \; (t_0)', 'S \; (t_0)', 'W \; (t_0)']
    entry_names_x = [0, temp_base[uncertainty_to_analyze]]
    x_axis_measure_unit = '(m)'
elif uncertainty_to_analyze==4:
    entry_names_x = [0, 'V_R \; (t_0)', 'V_S \; (t_0)', 'V_W \; (t_0)']
    x_axis_measure_unit = '(m/s)'
elif 4<uncertainty_to_analyze<8:
    temp_base = [0, 'V_R \; (t_0)', 'V_S \; (t_0)', 'V_W \; (t_0)']
    entry_names_x = [0, temp_base[uncertainty_to_analyze-4]]
    x_axis_measure_unit = '(m/s)'
elif uncertainty_to_analyze == 8 or uncertainty_to_analyze==18:
    entry_names_x = [0, 'R \; (t_E)', 'S \; (t_E)', 'W \; (t_E)']
    x_axis_measure_unit = '(m)'
elif 8<uncertainty_to_analyze<12 or 18<uncertainty_to_analyze<22:
    temp_base = [0, 'R \; (t_E)', 'S \; (t_E)', 'W \; (t_E)']
    entry_names_x = [0, temp_base[uncertainty_to_analyze]]
    x_axis_measure_unit = '(m)'
elif uncertainty_to_analyze==12or uncertainty_to_analyze==22:
    entry_names_x = [0, 'V_R \; (t_E)', 'V_S \; (t_E)', 'V_W \; (t_E)']
    x_axis_measure_unit = '(m/s)'
elif 12<uncertainty_to_analyze<16 or 22<uncertainty_to_analyze<26:
    temp_base = [0, 'V_R \; (t_E)', 'V_S \; (t_E)', 'V_W \; (t_E)']
    entry_names_x = [0, temp_base[uncertainty_to_analyze-4]]
    x_axis_measure_unit = '(m/s)'
elif uncertainty_to_analyze == 16 or uncertainty_to_analyze==26:
    entry_names_x = [0, '\gamma (t_E)']
    x_axis_measure_unit = '(°)'
elif uncertainty_to_analyze == 17 or uncertainty_to_analyze ==27:
    entry_names_x = [0, 'V (t_E)']
    x_axis_measure_unit = '(m/s)'
else:
    raise Exception('No such uncertainty exists to be analyzed, or you just forgot to update the elif\'s.'
                    'That\'s probably the case...')

entry_names_y = [0, 'R \; (t_1)', 'S \; (t_1)', 'W \; (t_1)', 'V_R \; (t_1)', 'V_S \; (t_1)', 'V_W \; (t_1)']
y_axis_measure_units = ['(m)', '(m/s)']

print(f'Analyzing uncertainty: {uncertainties[uncertainty_to_analyze]}')

# PLOT HISTOGRAMS REGARDING THE APPLIED UNCERTAINTY
fig_hist, ax_hist = plt.subplots(entries-1, figsize=(7,6), sharex=True)
for entry in range(1,entries):
    if type(ax_hist) == np.ndarray:
        ax_hist[entry-1].hist(perturbations[:, entry], 20, edgecolor='black', linewidth=1.2)
        ax_hist[entry-1].set_xlabel(fr'$\Delta {entry_names_x[entry]}$ ' + x_axis_measure_unit, fontsize=x_label_size)
        ax_hist[entry-1].tick_params(axis='both', which='major', labelsize=ticks_size)
        ax_hist[entry-1].tick_params(axis='x', labelbottom=True)
    else:
        ax_hist.hist(perturbations[:, entry], 20, edgecolor='black', linewidth=1.2)
        ax_hist.set_xlabel(fr'$\Delta {entry_names_x[entry]}$ ' + x_axis_measure_unit, fontsize=x_label_size)
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

    rsw_end_values[run_number] = rsw_state_difference[-1, :]
    state_history_end_values[run_number] = np.array([position_difference_norm[-1], velocity_difference_norm[-1], final_eccentricity])  # np.amax(position_difference_norm)

    axs[0, 0].plot(epochs_diff, position_difference_norm, color='blue')
    axs[1, 0].plot(epochs_diff, velocity_difference_norm, color='blue')

    axs[0, 1].plot(epochs_diff, altitude_difference, color='blue')
    axs[1, 1].plot(epochs_diff, speed_difference, color='blue')

fig.suptitle(f'Performed Propagations (End state: {final_state_name})', fontsize=suptitle_size)
fig.supxlabel('Elapsed time (days)', fontsize=x_label_size)
axs[0, 0].set_ylabel('Position Error Magnitude (m)', fontsize=y_label_size)
axs[1, 0].set_ylabel('Velocity Error Magnitude (m/s)', fontsize=y_label_size)
axs[0, 1].set_ylabel('Altitude Error (m)', fontsize=y_label_size)
axs[1, 1].set_ylabel('Speed Error (m/s)', fontsize=y_label_size)
for i in [0,1]:
    for j in [0,1]:
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

# PLOT THE DIFFERENCE IN ENTRY PARAMETERS: f.p.a., entry airspeed, and so on
dependent_variable_end_values = dict()
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

    if stop_before_aerocapture:
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

        nominal_end_dependent_variables = nominal_depvarh_interpolator.interpolate(epochs_comparison_depvar[-1])
        current_end_dependent_variables = depvarh_interpolator.interpolate(epochs_comparison_depvar[-1])

        final_fpa_difference = abs(nominal_end_dependent_variables[5] - current_end_dependent_variables[5])
        final_airspeed_difference = abs(nominal_end_dependent_variables[6] - current_end_dependent_variables[6])

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
    fig_depvar, ax_depvar = plt.subplots(2,2, figsize=(6, 8), sharex='col')
    for entry in range(1, entries):
        marker = marker_styles[0][entry-1]
        facecolor = marker_styles[1][entry-1]
        edgecolor = marker_styles[2][entry-1]
        ax_depvar[0, 0].scatter(perturbations[:, entry], dependent_variable_end_values_array[:,0],
                     label=fr'$\Delta {entry_names_x[entry]}$', marker=marker, facecolor=facecolor, edgecolor=edgecolor)
        ax_depvar[1, 0].scatter(perturbations[:, entry], dependent_variable_end_values_array[:, 1],
                     label=fr'$\Delta {entry_names_x[entry]}$', marker=marker, facecolor=facecolor,edgecolor=edgecolor)

        aerocapture_performed_cells = list(np.where(dependent_variable_end_values_array[:, 2] == 1)[0])
        yes_ae__perturbations = perturbations[aerocapture_performed_cells, :]
        yes_ae__depvars_endvalues = dependent_variable_end_values_array[aerocapture_performed_cells, :]
        ax_depvar[0, 1].scatter(yes_ae__perturbations[:, entry], yes_ae__depvars_endvalues[:, 0],
                                label=fr'$\Delta {entry_names_x[entry]}$', marker=marker, facecolor=facecolor,
                                edgecolor=edgecolor)
        ax_depvar[1, 1].scatter(yes_ae__perturbations[:, entry], yes_ae__depvars_endvalues[:, 1],
                                label=fr'$\Delta {entry_names_x[entry]}$', marker=marker, facecolor=facecolor,
                                edgecolor=edgecolor)

    ax_depvar[0, 0].set_ylabel(r'$\gamma \,(t_1)$ (\textdegree)', fontsize=y_label_size)
    ax_depvar[1, 0].set_ylabel(r'$Airspeed \,(t_1)$ (m/s)', fontsize=y_label_size)
    fig_depvar.supxlabel(fr'${xlabel} \, (t_0)$ ' + x_axis_measure_unit, fontsize=x_label_size)
    for i in [0,1]:
        for j in [0,1]:
            if entries-1 > 1:
                ax_depvar[i, j].legend()
    fig_depvar.suptitle(fr'Impact of ${xlabel} \, (t_0) $ on the final $\gamma$ and Airspeed (End state: {final_state_name})', fontsize=suptitle_size)
    fig_depvar.tight_layout()


# PLOT THE IMPACT OF THE APPLIED UNCERTAINTY ON THE FINAL NORM OF POSITION DIFFERENCE
fig2, axs2 = plt.subplots(2, figsize=(6, 8))

for entry in range(1, entries):
    axs2[0].scatter(perturbations[:, entry], state_history_end_values_array[:, 0], label=fr'$\Delta {entry_names_x[entry]}$',
                    marker=marker_styles[0][entry-1], facecolor=marker_styles[1][entry-1], edgecolor=marker_styles[2][entry-1])
    axs2[1].scatter(perturbations[:, entry], state_history_end_values_array[:, 1], label=fr'$\Delta {entry_names_x[entry]}$',
                    marker=marker_styles[0][entry - 1], facecolor=marker_styles[1][entry - 1],
                    edgecolor=marker_styles[2][entry - 1])
axs2[0].set_ylabel(r'$|\Delta \mathbf{r}| \,(t_1)$ (m)', fontsize=y_label_size)
axs2[1].set_ylabel(r'$|\Delta \mathbf{v}| \,(t_1)$ (m/s)', fontsize=y_label_size)
fig2.supxlabel(fr'${xlabel} \, (t_0)$ ' + x_axis_measure_unit, fontsize=x_label_size)
for i in [0,1]:
    # Show the major grid lines with dark grey lines
    axs2[i].grid(visible=True, which='major', color='#666666', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    axs2[i].minorticks_on()
    axs2[i].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axs2[i].tick_params(axis='both', which='major', labelsize=ticks_size)
    if entries-1 > 1:
        axs2[i].legend()
fig2.suptitle(fr'Impact of ${xlabel} \, (t_0) $ on the final state (End state: {final_state_name})', fontsize=suptitle_size)
fig2.tight_layout()

# Plot of final eccentricity
fig_ecc, axs_ecc = plt.subplots()

for entry in range(1, entries):
    axs_ecc.scatter(perturbations[:, entry], state_history_end_values_array[:, 2],
                    label=fr'$\Delta {entry_names_x[entry]}$',
                    marker=marker_styles[0][entry - 1], facecolor=marker_styles[1][entry - 1],
                    edgecolor=marker_styles[2][entry - 1])
axs_ecc.axhline(y=nominal_final_eccentricity, color='grey', linestyle='--')

for i in [0]:  # change if you add other subplots
    # Show the major grid lines with dark grey lines
    axs_ecc.grid(visible=True, which='major', color='#666666', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    axs_ecc.minorticks_on()
    axs_ecc.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axs_ecc.tick_params(axis='both', which='major', labelsize=ticks_size)
    if entries-1 > 1:
        axs_ecc.legend()
axs_ecc.set_ylabel(r'Final eccentricity $(t_1)$')
axs_ecc.set_xlabel(fr'${xlabel} \, (t_0)$ ' + x_axis_measure_unit, fontsize=x_label_size)

fig_ecc.suptitle(fr'Impact of ${xlabel} \, (t_0) $ on the final eccentricity (End state: {final_state_name})', fontsize=suptitle_size)
fig_ecc.tight_layout()


# SHOW THE IMPACT OF THE APPLIED UNCERTAINTY ON THE FINAL NORM OF POSITION DIFFERENCE
# and
# SHOW UNCERTAINTY HISTOGRAMS AND RESULTING POSITION ERROR
# plt.show()


max_entries_pert = perturbations.shape[1] - 1

# PLOT THE IMPACT OF THE APPLIED UNCERTAINTY ON THE FINAL RSW COMPONENTS AND
# PLOT THE HISTOGRAM OF THE FINAL RSW COMPONENTS DISTRIBUTION
fig3_hist, ax3_hist = plt.subplots(3,2, figsize=(9, 6), sharex='col', constrained_layout=True)
fig3, axs3 = plt.subplots(2, figsize=(6, 8))

xlabel = ['', '']
ylabel = ['', '']
for entry in range(1,7):
    plot_nr = int((entry-1)/3)  # 0, 0, 0, 1, 1, 1
    coord_nr = (entry-1) % 3  # 0, 1, 2, 0, 1, 2

    # nice_c = entry%3-1
    # entry%3 : 1, 2, 0, 1, 2, 0

    plot_label = '\\Delta ' + entry_names_x[min(coord_nr+1,max_entries_pert)] + ' \\rightarrow ' + '\\Delta ' + entry_names_y[entry]

    x_component_name = entry_names_x[min(coord_nr+1,max_entries_pert)].split()[0]
    y_component_name = entry_names_y[entry].split()[0]

    # It leads to a label like: x1, x2, x3 (so it's like: comma, comma, no comma; comma, comma, no comma)
    add_y_comma = ',' if entry%3 != 0 else ''
    add_x_comma = ',' if entry%3 != 0 and max_entries_pert > 1 else ''

    xlabel[plot_nr] = xlabel[plot_nr] + '\\Delta ' + x_component_name + add_x_comma if entry%3+1 <= max_entries_pert else xlabel[plot_nr]
    ylabel[plot_nr] = ylabel[plot_nr] + '\\Delta ' + y_component_name + add_y_comma

    axs3[plot_nr].scatter(perturbations[:,min(coord_nr+1,max_entries_pert)],max_vals_in_rsw[:,entry-1], label=fr'${plot_label}$',
                 marker=marker_styles[0][coord_nr], facecolor=marker_styles[1][coord_nr], edgecolor=marker_styles[2][coord_nr])

    ax3_hist[coord_nr, plot_nr].hist(max_vals_in_rsw[:, entry - 1], 30, edgecolor='black', linewidth=1.2)
    ax3_hist[coord_nr, plot_nr].tick_params(axis='both', which='major', labelsize=ticks_size)
    ax3_hist[coord_nr, plot_nr].set_xlabel(fr'$\Delta {entry_names_y[entry]} $ ' + y_axis_measure_units[plot_nr], fontsize=x_label_size)
    ax3_hist[coord_nr, plot_nr].tick_params(axis='x', labelbottom=True)


for i in [0,1]:
    axs3[i].tick_params(axis='both', which='major', labelsize=ticks_size)
    # Show the major grid lines with dark grey lines
    axs3[i].grid(visible=True, which='major', color='#666666', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    axs3[i].minorticks_on()
    axs3[i].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axs3[i].set_xlabel(fr'${xlabel[i]} \; (t_0)$ ' + x_axis_measure_unit, fontsize=x_label_size)
    axs3[i].set_ylabel(fr'${ylabel[i]} \; (t_1)$ ' + y_axis_measure_units[i], fontsize=y_label_size)
    axs3[i].legend()

fig3.suptitle(fr'Impact of ${xlabel[0]} \; (t_0)$ on final RSW position and velocity', fontsize=suptitle_size)

fig3_hist.suptitle(f'End Values Distribution (End state: {final_state_name})', fontsize=suptitle_size)
fig3_hist.supylabel('Occurrences', fontsize=common_y_label_size)

fig3_hist.tight_layout()
fig3.tight_layout()

# Plot the heat flux accuracy and the subsequent
if evaluated_arc == 1 or evaluated_arc==12:
    ...

# SHOW THEM
plt.show()