from JupiterTrajectory_GlobalParameters import *
from optimization_functions import *
from PlotsUtilities import *
from tudatpy.util import pareto_optimums
from tudatpy import plotting

import pandas as pd
import numpy as np
import shutil

current_dir = os.path.dirname(__file__)
# select_objective = 'benefit_over_insertion_burn'

use_old_results = True
use_refined_results = True  # overwrites previous boolean

apply_constraints = True
filter_results_further = True

show_plots_singularly = False
save_figures = False
delete_figures_folder_if_existing = False
fontsize = 20


save_dir = current_dir + '/Figures/'

if delete_figures_folder_if_existing:
    does_folder_exists = os.path.exists(save_dir)
    shutil.rmtree(save_dir) if does_folder_exists else None

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

decision_variable_list = ['InterplanetaryVelocity', 'EntryFpa']
objectives_list = ['payload_mass_fraction', 'total_radiation_dose_krad', 'benefit_over_insertion_burn']
constraints_list = ['maximum_aerodynamic_load', 'peak_heat_flux', 'minimum_jupiter_distance', 'maximum_jupiter_distance', 'final_eccentricity']

total_columns = decision_variable_list + objectives_list + constraints_list

# decision_variable_nice_labels = [r'$V_{J\,\infty}$ [m/s]', r'$\gamma_E$ [deg]']
# objectives_nice_labels = [r'$f_{payload}$ [-]', 'Radiation Dose [rad]', 'Aerocapture Benefit [-]']
#
# labels_list = decision_variable_nice_labels + objectives_nice_labels

filename_addition = '_old' if use_old_results else ''
filename_addition = '_refined' if use_refined_results else ''
grid_search_data = np.loadtxt(current_dir + '/GridSearch/' + 'grid_search_results' + filename_addition + '.dat')

grid_search_df = pd.DataFrame(grid_search_data, columns=total_columns)

if apply_constraints:
    filtered_grid_search_df = constraints_filter(grid_search_df, constraints_list)
else:
    filtered_grid_search_df = grid_search_df

if filter_results_further:
    convex_filtered_grid_search_df = filtered_grid_search_df[(filtered_grid_search_df['EntryFpa'] > -3.4)]
else:
    convex_filtered_grid_search_df = filtered_grid_search_df

max_rad_dose = filtered_grid_search_df['total_radiation_dose_krad'].max()
min_rad_dose = filtered_grid_search_df['total_radiation_dose_krad'].min()
print(max_rad_dose)
print(min_rad_dose)

# # Plot decision variable space with color assigned based on the selected objective value
# fig1, ax1 = plot_with_colorbar(filtered_grid_search_df, 'InterplanetaryVelocity', 'EntryFpa', select_objective, directly_show=show_plots_singularly)
#
# # Plot the selected ojective as function of the EntryFpa, coloring based on value of InterplanetaryVelocity
# fig2, ax2 = plot_with_colorbar(filtered_grid_search_df, 'EntryFpa', select_objective, 'InterplanetaryVelocity', directly_show=show_plots_singularly)
# # Plot the selected ojective as function of the InterplanetaryVelocity, coloring based on value of EntryFpa
# fig3, ax3 = plot_with_colorbar(filtered_grid_search_df, 'InterplanetaryVelocity', select_objective, 'EntryFpa', directly_show=show_plots_singularly)
#
# # 3D plot with projections
# fig10, ax10 = plot_decision_variables_3D_ver2(filtered_grid_search_df, decision_variable_list, select_objective, objectives_list, directly_show=show_plots_singularly)
# fig11, ax11 = plot_decision_variables_3D(filtered_grid_search_df, decision_variable_list, select_objective, objectives_list, directly_show=show_plots_singularly)

# Double objective plots
fig50, ax50 = plot_with_colorbar(filtered_grid_search_df, 'total_radiation_dose_krad', 'benefit_over_insertion_burn',
                                 'EntryFpa', directly_show=show_plots_singularly, save_fig=save_figures,
                                 save_dir=save_dir, fontsize=fontsize)
fig51, ax51 = plot_with_colorbar(filtered_grid_search_df, 'total_radiation_dose_krad', 'benefit_over_insertion_burn',
                                 'InterplanetaryVelocity', directly_show=show_plots_singularly, save_fig=save_figures,
                                 save_dir=save_dir, fontsize=fontsize)

coloring = 'final_eccentricity'  # 'EntryFpa'
fig50_2, ax50_2 = pareto_front_plot(convex_filtered_grid_search_df, 'total_radiation_dose_krad',
                                    'benefit_over_insertion_burn', coloring, [min, max], save_fig=save_figures,
                                    save_dir=save_dir, fontsize=fontsize)  #, best_candidate_index=7358)
fig50_2_zoom, ax50_2_zoom = pareto_front_plot(convex_filtered_grid_search_df, 'total_radiation_dose_krad',
                                              'benefit_over_insertion_burn', 'EntryFpa', [min, max],
                                              save_fig=save_figures, pareto_zoom=True, save_dir=save_dir, fontsize=fontsize)

if use_refined_results:
    for i, sol in enumerate(best_solutions.keys()):
        rad_dose_sol = best_solutions[sol]['total_radiation_dose_krad']
        benefit_ae_sol = best_solutions[sol]['benefit_over_insertion_burn']

        ax50_2.scatter(rad_dose_sol, benefit_ae_sol, color='r', facecolor='none')
        ax50_2.annotate(str(sol), xy=(rad_dose_sol, benefit_ae_sol), xytext=(-60*(-1)**i, -20*(-1)**i), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->'), fontsize=fontsize * 5 / 6)

    fig50_2_zoom2, ax50_2_zoom2 = pareto_front_plot(convex_filtered_grid_search_df, 'total_radiation_dose_krad',
                                                    'benefit_over_insertion_burn', 'EntryFpa', [min, max],
                                                    save_fig=save_figures, pareto_zoom=True, save_dir=save_dir,
                                                    fontsize=fontsize,
                                                    best_candidate_index=[9867])
else:
    fig50_2_zoom2, ax50_2_zoom2 = pareto_front_plot(convex_filtered_grid_search_df, 'total_radiation_dose_krad',
                                                  'benefit_over_insertion_burn', 'EntryFpa', [min, max],
                                                  save_fig=save_figures, pareto_zoom=True, save_dir=save_dir, fontsize=fontsize,
                                                  best_candidate_index=[7358, 9857])


fig55, ax55 = plot_with_colorbar(filtered_grid_search_df, 'total_radiation_dose_krad', 'payload_mass_fraction',
                                 'EntryFpa', directly_show=show_plots_singularly, save_fig=save_figures,
                                 save_dir=save_dir, fontsize=fontsize)
fig56, ax56 = plot_with_colorbar(filtered_grid_search_df, 'total_radiation_dose_krad', 'payload_mass_fraction',
                                 'InterplanetaryVelocity', directly_show=show_plots_singularly, save_fig=save_figures,
                                 save_dir=save_dir, fontsize=fontsize)

fig55_2, ax55_2 = pareto_front_plot(convex_filtered_grid_search_df, 'total_radiation_dose_krad',
                                    'payload_mass_fraction', 'EntryFpa', [min, max], save_fig=save_figures,
                                    save_dir=save_dir, fontsize=fontsize)
fig55_2_zoom, ax55_2_zoom = pareto_front_plot(convex_filtered_grid_search_df, 'total_radiation_dose_krad',
                                              'payload_mass_fraction', 'EntryFpa', [min, max],
                                              save_fig=save_figures, pareto_zoom=True, save_dir=save_dir, fontsize=fontsize)


fig60, ax60 = plot_with_colorbar(filtered_grid_search_df, 'benefit_over_insertion_burn', 'payload_mass_fraction',
                                 'EntryFpa', directly_show=show_plots_singularly, save_fig=save_figures,
                                 save_dir=save_dir, fontsize=fontsize)
fig61, ax61 = plot_with_colorbar(filtered_grid_search_df, 'benefit_over_insertion_burn', 'payload_mass_fraction',
                                 'InterplanetaryVelocity', directly_show=show_plots_singularly, save_fig=save_figures,
                                 save_dir=save_dir, fontsize=fontsize)

fig60_2, ax60_2 = pareto_front_plot(convex_filtered_grid_search_df, 'benefit_over_insertion_burn',
                                    'payload_mass_fraction', 'EntryFpa', [max, max], save_fig=save_figures,
                                    save_dir=save_dir, fontsize=fontsize)
fig60_2_zoom, ax60_2_zoom = pareto_front_plot(convex_filtered_grid_search_df, 'benefit_over_insertion_burn',
                                              'payload_mass_fraction', 'EntryFpa', [max, max],
                                              save_fig=save_figures, pareto_zoom=True, save_dir=save_dir, fontsize=fontsize)


plt.show()
