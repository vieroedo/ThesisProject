from JupiterTrajectory_GlobalParameters import *
from optimization_functions import *
from PlotsUtilities import *

import pandas as pd
import numpy as np

current_dir = os.path.dirname(__file__)

select_objective = 'benefit_over_insertion_burn'
use_old_results = True
apply_constraints = True


decision_variable_list = ['InterplanetaryVelocity', 'EntryFpa']
objectives_list = ['payload_mass_fraction', 'total_radiation_dose_krad', 'benefit_over_insertion_burn']
constraints_list = ['maximum_aerodynamic_load', 'peak_heat_flux', 'minimum_jupiter_distance', 'maximum_jupiter_distance', 'final_eccentricity']

total_columns = decision_variable_list + objectives_list + constraints_list

# decision_variable_nice_labels = [r'$V_{J\,\infty}$ [m/s]', r'$\gamma_E$ [deg]']
# objectives_nice_labels = [r'$f_{payload}$ [-]', 'Radiation Dose [rad]', 'Aerocapture Benefit [-]']
#
# labels_list = decision_variable_nice_labels + objectives_nice_labels

filename_addition_old = '_old' if use_old_results else ''
grid_search_data = np.loadtxt(current_dir + '/GridSearch/' + 'grid_search_results' + filename_addition_old + '.dat')

grid_search_df = pd.DataFrame(grid_search_data, columns=total_columns)

if apply_constraints:
    filtered_grid_search_df = constraints_filter(grid_search_df, constraints_list)
else:
    filtered_grid_search_df = grid_search_df

max_rad_dose = filtered_grid_search_df['total_radiation_dose_krad'].max()
min_rad_dose = filtered_grid_search_df['total_radiation_dose_krad'].min()
print(max_rad_dose)
print(min_rad_dose)

# Plot decision variable space with color assigned based on the selected objective value
plot_with_colorbar(filtered_grid_search_df, 'InterplanetaryVelocity', 'EntryFpa', select_objective, directly_show=True)

# Plot the selected ojective as function of the EntryFpa, coloring based on value of InterplanetaryVelocity
plot_with_colorbar(filtered_grid_search_df, 'EntryFpa', select_objective, 'InterplanetaryVelocity', directly_show=True)
# Plot the selected ojective as function of the InterplanetaryVelocity, coloring based on value of EntryFpa
plot_with_colorbar(filtered_grid_search_df, 'InterplanetaryVelocity', select_objective, 'EntryFpa', directly_show=True)

# 3D plot with projections
plot_decision_variables_3D_ver2(filtered_grid_search_df, decision_variable_list, select_objective, objectives_list)
plot_decision_variables_3D(filtered_grid_search_df, decision_variable_list, select_objective, objectives_list)

# Double objective plots
plot_with_colorbar(filtered_grid_search_df, 'total_radiation_dose_krad', 'benefit_over_insertion_burn', 'EntryFpa', directly_show=True)
plot_with_colorbar(filtered_grid_search_df, 'total_radiation_dose_krad', 'payload_mass_fraction', 'EntryFpa', directly_show=True)
plot_with_colorbar(filtered_grid_search_df, 'benefit_over_insertion_burn', 'payload_mass_fraction', 'EntryFpa', directly_show=True)

