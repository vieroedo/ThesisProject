import numpy as np
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from handle_functions import *
import os

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
surface_gravity = jupiter_gravitational_parameter / jupiter_radius ** 2
average_temperature = scale_height * surface_gravity / gas_constant

print(f'Jupiter atmosphere avg temp: {average_temperature} K')

current_dir = os.path.dirname(__file__)


lol = np.loadtxt(current_dir + '/galileo_flight_data.txt')

print(jupiter_atmosphere_density_model(460e3))
print('lol')

altitude_lol = np.linspace(0,1e4*1e3,10000)
density_lol = jupiter_atmosphere_density_model(altitude_lol)
plt.plot(altitude_lol/1e3,density_lol)
plt.yscale('log')
plt.xscale('log')
plt.show()

convective_hf, radiative_hf, radiative_hf_w_blockage = atmospheric_entry_heat_loads_correlations(density_lol,
                                                                                                 galileo_velocity_from_altitude(
                                                                                                     altitude_lol),
                                                                                                 nose_radius=galileo_radius)

plt.plot(altitude_lol/1e3,convective_hf, label='conv')
plt.plot(altitude_lol/1e3,radiative_hf, label='rad')
plt.plot(altitude_lol/1e3,radiative_hf_w_blockage, label='rad_w_blockage')
plt.yscale('log')
plt.legend()
plt.show()


x_data = np.linspace(-2,8,1000)

y_data2 = 1 - 0.72 * x_data + 0.13 * x_data**2
y_data = ( 2.344/x_data * (np.sqrt(x_data+1)-1) ) **1.063

plt.plot(x_data, y_data, label='other one')
plt.plot(x_data, y_data2, label='quadratic')
plt.title('B equations')
plt.legend()
plt.show()