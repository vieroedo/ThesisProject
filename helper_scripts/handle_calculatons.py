import numpy as np
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from handle_functions import *


#### r_SOI calculation ####################################################
r_SOI = dict()
for moon in list(galilean_moons_data.keys()):
    moon_sma = galilean_moons_data[moon]['SMA']
    moon_mass = galilean_moons_data[moon]['Mass']
    r_SOI[moon] = moon_sma * (moon_mass / jupiter_mass)**(2/5)
for moon in r_SOI.keys():
    print(f'{moon}: {r_SOI[moon]/1e3} km')
###########################################################################

# test

ecc = 1.5
sma = -50e6
theta_max = np.arccos(-1/ecc)
theta = np.linspace(-theta_max,theta_max,200)
radii = radius_from_true_anomaly(theta, ecc, sma, planet_SoI=1e9)
x, y = cartesian_2d_from_polar(radii, theta)
plt.plot(x, y)
plt.show()

### Jupiter atmosphere ################
ref_density = 0.16  # kg/m^3
scale_height = 27e3
molecular_weight = 2.22  # kg/kmol
gas_constant = 8314.32 / molecular_weight
surface_gravity = central_body_gravitational_parameter / jupiter_radius**2
average_temperature = scale_height * surface_gravity / gas_constant

print(f'Jupiter atmosphere avg temp: {average_temperature} K')