import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import os

from tudatpy.kernel import constants
import CapsuleEntryUtilities as Util
plt.rc('font', size=15)
current_dir = os.path.dirname(__file__)

output_folder = current_dir + '/SimulationOutput/IntegratorTuning/'
benchmark_output_path = current_dir + '/SimulationOutput/benchmarks/full_traj/'

benchmark_error_matrix = np.loadtxt(benchmark_output_path + 'benchmarks_state_difference.dat')
bm_epochs = benchmark_error_matrix[:,0]
bm_epochs_plot = (bm_epochs - bm_epochs[0]) / constants.JULIAN_DAY
bm_position_error = benchmark_error_matrix[:,1:4]
bm_distance_error = LA.norm(bm_position_error, axis=1)

cut_beginning_boundary_values_by = 8
cut_end_boundary_values_by = 2

# from -5 to +2
number_of_settings = range(-5, 2)

fig, ax = plt.subplots(figsize=(6,5), constrained_layout=True)
# fig3d = plt.figure()
# ax3d = plt.axes(projection='3d')
for settings_index in number_of_settings:

    output_path = output_folder + '/tol_1e' + str(-10 + settings_index) + '/'

    state_difference_matrix = np.loadtxt(output_path + 'state_difference_wrt_benchmark.dat')
    state_history_matrix = np.loadtxt(output_path + 'state_history.dat')
    # state_difference = dict(zip(state_difference_matrix[:, 0], state_difference_matrix[:, 1:]))

    epochs = state_difference_matrix[:, 0]
    epochs_plot = (epochs - epochs[0])/constants.JULIAN_DAY
    position_error = state_difference_matrix[:, 1:4]
    position_history = state_history_matrix[:,1:4]

    distance_error = LA.norm(position_error, axis=1)

    cbv_end = cut_end_boundary_values_by
    cbv_beg = cut_beginning_boundary_values_by
    ax.plot(epochs_plot[cbv_beg:-cbv_end], distance_error[cbv_beg:-cbv_end], label='1e' + str(-10 + settings_index))
    # Plot 3-D Trajectory

    # ax3d.plot(position_history[:,0], position_history[:,1], position_history[:,2])
    # fig3d = plt.figure()
    # ax3d = plt.axes(projection='3d')
    # fig3d, ax3d = Util.plot_base_trajectory(state_history_matrix, fig3d, ax3d)
    # fig3d.show()

ax.plot(bm_epochs_plot, bm_distance_error, label='bm')
ax.set(xlabel='Elapsed time [days]', ylabel='Position error [m]')
ax.set_yscale('log')
ax.legend()
plt.show()


