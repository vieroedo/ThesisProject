# import matplotlib as mpl
import matplotlib.pyplot as plt
# import numpy.linalg as LA
# import numpy as np
# import scipy as sp
# import random
# import os
#
# from tudatpy.kernel import constants
from tudatpy.kernel.math import interpolators

peak_heat_flux_max_error = 250  # kW/m2
heat_load_max_error = 40  # MJ/m2
eccentricity_max_error = 0.001



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
    # from 0 to 3
    'InitialPosition': 0, 'InitialPosition_R': 0, 'InitialPosition_S': 0, 'InitialPosition_W': 0,
    # from 4 to 7
    'InitialVelocity': 0, 'InitialVelocity_R': 0, 'InitialVelocity_S': 0, 'InitialVelocity_W': 0,

    # from 8 to 11
    'InitialPosition_Entry': 1, 'InitialPosition_R_Entry': 1,
    'InitialPosition_S_Entry': 1, 'InitialPosition_W_Entry': 1,
    # from 12 to 15
    'InitialVelocity_Entry': 1, 'InitialVelocity_R_Entry': 1,
    'InitialVelocity_S_Entry': 1, 'InitialVelocity_W_Entry': 1,
    # 16, 17
    'EntryFlightPathAngle': 1, 'EntryVelocity': 1,

    # from 18 to 21
    'FinalOrbit_InitialPosition_Entry': 12, 'FinalOrbit_InitialPosition_R_Entry': 12,
    'FinalOrbit_InitialPosition_S_Entry': 12, 'FinalOrbit_InitialPosition_W_Entry': 12,
    # from 22 to 25
    'FinalOrbit_InitialVelocity_Entry': 12, 'FinalOrbit_InitialVelocity_R_Entry': 12,
    'FinalOrbit_InitialVelocity_S_Entry': 12, 'FinalOrbit_InitialVelocity_W_Entry': 12,
    # 26, 27
    'FinalOrbit_EntryFlightPathAngle': 12, 'FinalOrbit_EntryVelocity': 12
}


multiple_initial_position_deviation_uncertainty = 2700  # m (1 sigma -> 3 sigma = 9000 m)
initial_position_deviation_uncertainty = 6300  # m (1 sigma -> 3 sigma = 9000 m)
multiple_initial_velocity_deviation_uncertainty = 7e-4  # m/s (1 sigma -> 3 sigma = 30 m/s)
initial_velocity_deviation_uncertainty = 1e-3  # m/s (1 sigma -> 3 sigma = 30 m/s)
entry_fpa_deviation_uncertainty = 0.01  # degrees (1 sigma -> 3 sigma = 0.03 degrees)
entry_velocity_magnitude_deviation_uncertainty = 33  # m/s (1 sigma -> 3 sigma = 99 m/s)
