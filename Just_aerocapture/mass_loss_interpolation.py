import scipy as sp
import numpy as np
from handle_functions import *

current_dir = os.path.dirname(__file__)

def mass_rate(input_list, k):
    # input_list = [m_0, alt, vel, delta_t]
    m_0 = input_list[0]
    alt = input_list[1]
    vel = input_list[2]
    delta_t = input_list[3]

    rho = jupiter_atmosphere_density_model(alt)
    dm_dt = k * rho * vel**6.9
    mass_change = dm_dt * delta_t
    final_mass = m_0 - mass_change
    return final_mass


galileo_flight_data = np.loadtxt(current_dir + '/GalileoMission/galileo_flight_data.txt')

flight_epoch = galileo_flight_data[:,0]*1
flight_altitude = galileo_flight_data[:,1] *1e3
flight_velocity = galileo_flight_data[:,2] * 1e3
flight_mass = galileo_flight_data[:,4]*1


delta_t = abs(flight_epoch[1:] - flight_epoch[:-1])
m_0 = flight_mass[:-1]

mass_solutions = flight_mass[1:]

x_values = np.vstack((m_0, flight_altitude[:-1], flight_velocity[1:], delta_t))
y_values = mass_solutions

solution = sp.optimize.curve_fit(mass_rate, x_values, y_values)

k_par = solution[0][0]


calc_mass = np.zeros(len(mass_solutions))
for i in range(len(mass_solutions)):
    calc_mass[i] = mass_rate(x_values[:,i], k_par)

plt.plot(flight_altitude[:-1]/1e3, mass_solutions, label='nominal')
plt.plot(flight_altitude[:-1]/1e3, calc_mass, label='calculated')
plt.legend()
plt.show()

