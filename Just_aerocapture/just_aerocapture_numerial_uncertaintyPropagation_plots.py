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
# import CapsuleEntryUtilities as Util

current_dir = os.path.dirname(__file__)

# uncertainties = ['EarthEph', 'SRP', 'InitialState', 'InitialState_1', 'InitialState_2', 'InitialState_3']
uncertainties = ['InitialPosition', 'InitialPosition_R', 'InitialPosition_S', 'InitialPosition_W',
                 'InitialVelocity', 'InitialVelocity_R', 'InitialVelocity_S', 'InitialVelocity_W']

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

# SET PARAMETERS
uncertainty_to_analyze = 0  # From 0 to 5
# number_of_runs = 100 (automatically calculated)

subdirectory = '/UncertaintyAnalysis/' + uncertainties[uncertainty_to_analyze] + '/'  # it can be 0, 1, 2
data_path = current_dir + subdirectory

perturbations = np.loadtxt(current_dir + f'/UncertaintyAnalysis/simulation_results_{uncertainties[uncertainty_to_analyze]}.dat')
# perturbations = perturbations[1:, :]
number_of_runs = len(perturbations[:,0]) + 1

# This is made for compatibility between the various uncertainty analyses
entries = perturbations.shape[1]

# if uncertainty_to_analyze == 0:
#     entry_names_x = [0, 'eph']
#     entry_names_y = [0, 'R \; (t_1)', 'S \; (t_1)', 'W \; (t_1)']
# elif uncertainty_to_analyze == 1:
#     entry_names_x = [0, 'C_r']
#     entry_names_y = [0, 'R \; (t_1)', 'S \; (t_1)', 'W \; (t_1)']
if uncertainty_to_analyze == 0:
    entry_names_x = [0, 'R \; (t_0)', 'S \; (t_0)', 'W \; (t_0)']
    # entry_names_y = [0, 'R \; (t_1)', 'S \; (t_1)', 'W \; (t_1)']
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
else:
    raise Exception('No such uncertainty exists to be analyzed, or you just forgot to update the elif\'s you cunt')

entry_names_y = [0, 'R \; (t_1)', 'S \; (t_1)', 'W \; (t_1)', 'V_R \; (t_1)', 'V_S \; (t_1)', 'V_W \; (t_1)']
y_axis_measure_units = ['(m)', '(m/s)']

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
fig_hist.suptitle('Random variable distribution', fontsize=suptitle_size)
fig_hist.supylabel('Occurrences', fontsize=common_y_label_size)
fig_hist.tight_layout()

# PLOT THE POSITION ERROR RESULTING FROM THE APPLIED UNCERTAINTY
fig, axs = plt.subplots(2, figsize=(6, 6), sharex=True)
end_values = dict()
# rsw_state_histories = dict()
rsw_end_values = dict()

for run_number in range(1,number_of_runs):
    state_history_np = np.loadtxt(data_path + 'state_history_' + str(run_number) + '.dat')
    state_difference_wrt_nominal_case_np = np.loadtxt(data_path + 'state_difference_wrt_nominal_case_' + str(run_number)
                                                      + '.dat')

    cartesian_state_history_np = state_history_np[:, 1:7]
    state_history = dict()
    for i, epoch in enumerate(state_history_np[:, 0]):
        state_history[epoch] = cartesian_state_history_np[i, :]

    cartesian_state_difference_wrt_nominal_case_np = state_difference_wrt_nominal_case_np[:, 1:7]
    state_difference_wrt_nominal_case = dict()
    for i, epoch in enumerate(state_difference_wrt_nominal_case_np[:, 0]):
        state_difference_wrt_nominal_case[epoch] = cartesian_state_difference_wrt_nominal_case_np[i, :]

    position_difference = state_difference_wrt_nominal_case_np[:, 1:4]
    velocity_difference = state_difference_wrt_nominal_case_np[:, 4:7]
    distance_difference = LA.norm(position_difference, axis=1)
    speed_difference = LA.norm(velocity_difference, axis=1)

    # Create interpolator settings
    interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation=interpolators.use_boundary_value)
    # Create 8th-order Lagrange interpolator for both cases
    sh_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        state_history, interpolator_settings)

    rsw_state_difference = np.zeros(np.shape(cartesian_state_difference_wrt_nominal_case_np))
    for i, epoch in enumerate(state_difference_wrt_nominal_case_np[:,0]):
        rsw_matrix = frame_conversion.inertial_to_rsw_rotation_matrix(sh_interpolator.interpolate(epoch))
        rsw_state_difference[i, 0:3] = rsw_matrix @ cartesian_state_difference_wrt_nominal_case_np[i, 0:3]
        rsw_state_difference[i, 3:6]= rsw_matrix @ cartesian_state_difference_wrt_nominal_case_np[i, 3:6]

    rsw_position_difference = rsw_state_difference[:,0:3]
    rsw_velocity_difference = rsw_state_difference[:,3:6]

    epochs_diff = state_difference_wrt_nominal_case_np[:, 0]
    epochs_diff = (epochs_diff - epochs_diff[0]) / constants.JULIAN_DAY

    # rsw_state_histories[run_number] = [epochs_diff, rsw_position_difference]
    # rsw_end_values[run_number] = rsw_position_difference[-1,:] # np.amax(rsw_position_difference,axis=0)
    rsw_end_values[run_number] = rsw_state_difference[-1, :]
    end_values[run_number] = np.array([distance_difference[-1], speed_difference[-1]])  # np.amax(distance_difference)

    # rsw_end_values[run_number] = np.amax(rsw_position_difference,axis=0)
    # end_values[run_number] = np.amax(distance_difference)
    axs[0].plot(epochs_diff, distance_difference, color='blue')
    axs[1].plot(epochs_diff, speed_difference, color='blue')

fig.suptitle('Performed Propagations', fontsize=suptitle_size)
[axs[i].set_yscale('log') for i in [0,1]]
fig.supxlabel('Elapsed time (days)', fontsize=x_label_size)
axs[0].set_ylabel('Position Error (m)', fontsize=y_label_size)
axs[1].set_ylabel('Speed Error (m/s)', fontsize=y_label_size)
[axs[i].tick_params(axis='both', which='major', labelsize=ticks_size) for i in [0,1]]
[axs[i].tick_params(axis='x', labelbottom=True) for i in [0,1]]
# Show the major grid lines with dark grey lines
[axs[i].grid(visible=True, which='major', color='#666666', linestyle='-') for i in [0,1]]
# Show the minor grid lines with very faint and almost transparent grey lines
[axs[i].minorticks_on() for i in [0,1]]
[axs[i].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2) for i in [0,1]]


# print(rsw_end_values)
max_vals_in_rsw = np.vstack(list(rsw_end_values.values()))

end_values_array = np.vstack(list(end_values.values()))

# PLOT THE IMPACT OF THE APPLIED UNCERTAINTY ON THE FINAL NORM OF POSITION DIFFERENCE
fig2, axs2 = plt.subplots(2, figsize=(6, 8))

xlabel = ''
for entry in range(1, entries):
    x_component_name = entry_names_x[entry].split()[0]
    add_x_comma = ',' if entry < entries-1 else ''
    xlabel = xlabel + '\Delta ' + x_component_name + add_x_comma if entry < entries else xlabel
    axs2[0].scatter(perturbations[:, entry], end_values_array[:,0], label=fr'$\Delta {entry_names_x[entry]}$',
                 marker=marker_styles[0][entry-1], facecolor=marker_styles[1][entry-1], edgecolor=marker_styles[2][entry-1])
    axs2[1].scatter(perturbations[:, entry], end_values_array[:, 1], label=fr'$\Delta {entry_names_x[entry]}$',
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
fig2.suptitle(fr'Impact of ${xlabel} \, (t_0) $ on the final state', fontsize=suptitle_size)
fig2.tight_layout()

# SHOW THE IMPACT OF THE APPLIED UNCERTAINTY ON THE FINAL NORM OF POSITION DIFFERENCE
# and
# SHOW UNCERTAINTY HISTOGRAMS AND RESULTING POSITION ERROR
plt.show()


max_entries_pert = perturbations.shape[1] -1

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

    plot_label = '\Delta ' + entry_names_x[min(coord_nr+1,max_entries_pert)] + ' \\rightarrow ' + '\Delta ' + entry_names_y[entry]

    x_component_name = entry_names_x[min(coord_nr+1,max_entries_pert)].split()[0]
    y_component_name = entry_names_y[entry].split()[0]

    # It leads to a label like: x1, x2, x3 (so it's like: comma, comma, no comma; comma, comma, no comma)
    add_y_comma = ',' if entry%3 != 0 else ''
    add_x_comma = ',' if entry%3 != 0 and max_entries_pert > 1 else ''

    xlabel[plot_nr] = xlabel[plot_nr] + '\Delta ' + x_component_name + add_x_comma if entry%3+1 <= max_entries_pert else xlabel[plot_nr]
    ylabel[plot_nr] = ylabel[plot_nr] + '\Delta ' + y_component_name + add_y_comma
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

# ax3_hist[1].set_ylabel('Occurrences', fontsize=14)
# ax3_hist.set_xticklabels(fontsize=12)
# ax3_hist.set_xticklabels(fontsize=12)

fig3.suptitle(fr'Impact of ${xlabel[0]} \; (t_0)$ on final RSW position and velocity', fontsize=suptitle_size)

fig3_hist.suptitle('End Values Distribution', fontsize=suptitle_size)
fig3_hist.supylabel('Occurrences', fontsize=common_y_label_size)

fig3_hist.tight_layout()
fig3.tight_layout()

# SHOW THEM
plt.show()