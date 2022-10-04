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
theta = np.linspace(0,2,200)
radius