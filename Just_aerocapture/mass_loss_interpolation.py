import scipy as sp
import numpy as np
from handle_functions import *

current_dir = os.path.dirname(__file__)

# From https://doi.org/10.2514/6.1996-2451

# Make everything big
plt.rc('font', size=15)

def mass_change_function(input_list, k):
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


def mass_rate(input_list, k):
    density = input_list[0]
    velocity = input_list[1]
    #k = 2.7659985259098415e-28
    dm_dt = k * density * velocity ** 6.9
    return dm_dt


galileo_flight_data = np.loadtxt(current_dir + '/GalileoMission/galileo_flight_data.txt')

flight_epoch = galileo_flight_data[:,0]*1
flight_altitude = galileo_flight_data[:,1] *1e3  # m
flight_velocity = galileo_flight_data[:,2] * 1e3  # m/s
flight_mass = galileo_flight_data[:,4]*1  # kg

flight_density = jupiter_atmosphere_density_model(flight_altitude)


delta_t = abs(flight_epoch[1:] - flight_epoch[:-1])
m_0 = flight_mass[:-1]

mass_solutions = flight_mass[1:]
dm_dt_solutions = (flight_mass[1:] - flight_mass[:-1]) / delta_t

x_values = np.vstack((m_0, flight_altitude[:-1], flight_velocity[:-1], delta_t))
y_values = mass_solutions

scaling_values_2 = np.array([1/flight_density[:-1].max(), 1/flight_velocity[:-1].max(), 1/abs(dm_dt_solutions).max()])


x_values_2 = np.vstack((flight_density[:-1], flight_velocity[:-1]))
x_values_2 = (x_values_2.T * scaling_values_2[0:2]).T
y_values_2 = dm_dt_solutions * scaling_values_2[2]



solution = sp.optimize.curve_fit(mass_change_function, x_values, y_values)
solution_2 = sp.optimize.curve_fit(mass_rate, x_values_2, y_values_2)

k_par = solution[0][0]
k_par_2 = solution_2[0][0]

print(k_par)
print(k_par_2)


calc_mass = np.zeros(len(mass_solutions))
calc_dmdt = np.zeros(len(dm_dt_solutions))

for i in range(len(mass_solutions)):
    calc_mass[i] = mass_change_function(x_values[:, i], k_par)
for i in range(len(dm_dt_solutions)):
    calc_dmdt[i] = mass_rate(x_values_2[:, i], k_par_2) / scaling_values_2[2]

fig, ax = plt.subplots()
ax.plot(flight_altitude[:-1]/1e3, mass_solutions, label='nominal')
ax.plot(flight_altitude[:-1]/1e3, calc_mass, label='calculated')
ax.set_title('mass change')
plt.legend()


altitudes = np.linspace(20e3,1000e3,10000)
plot_dmdt = np.zeros(len(altitudes))

input_values = np.vstack((jupiter_atmosphere_density_model(altitudes), galileo_velocity_from_altitude(altitudes)))
input_values = (input_values.T * scaling_values_2[0:2]).T

for i in range(len(altitudes)):
    plot_dmdt[i] = mass_rate(input_values[:,i], k_par_2) / scaling_values_2[2]


fig2, ax2 = plt.subplots()
ax2.plot(flight_altitude[:-1]/1e3, -dm_dt_solutions, label='nominal')
# ax2.plot(flight_altitude[:-1]/1e3, calc_dmdt, label='calculated')
ax2.plot(altitudes/1e3, -plot_dmdt, label=r'$k\, \rho \, V^{6.9}$')
# ax2.set_title('dmdt')
ax2.set(xlabel='Altitude [km]', ylabel=r'Mass rate  $\dot{m}$ [kg/s]', xlim=[0,400])
plt.tight_layout()
plt.legend()


plt.show()

