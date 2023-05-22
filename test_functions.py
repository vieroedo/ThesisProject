import numpy as np
from tudatpy.kernel import constants
from JupiterTrajectory_GlobalParameters import *
from handle_functions import *


time_vector = np.array([-5 * constants.JULIAN_DAY, -1 * constants.JULIAN_DAY])
eccentricity = 1.011
semimajor_axis = -6104236998.987195
mu_parameter = jupiter_gravitational_parameter

delta_theta = true_anomaly_from_delta_t(time_vector, eccentricity, semimajor_axis, mu_parameter)
print(delta_theta)
