import numpy as np
from matplotlib import pyplot as plt
from second_order_equations_aerocapture import *

tau_enter = 0.
tau_exit = 3.5399430153600635

x_0 = 0.7887502523678501

a1 = -0.0004902428451031968
a2 = -1.0064424565127459
a3 = 0.7941367348593324

tau_linspace = np.linspace(tau_enter, tau_exit, 1000)
x_tau = x_tau_function(tau_linspace,a1,a2,a3)
x_tau_plot = x_tau - 0

plt.plot(tau_linspace,x_tau_plot)
plt.plot(tau_linspace, np.zeros(len(tau_linspace)))
plt.show()


c_const = 0.17823730061208057
alpha_const = 0.49707999107837586
k1 = 0.00032538430781842343

phi_values = phi_function_zero_first_order(np.linspace(0,10,1000),c_const,k1, alpha_const)

plt.plot(np.linspace(0,10,1000), phi_values)
plt.show()