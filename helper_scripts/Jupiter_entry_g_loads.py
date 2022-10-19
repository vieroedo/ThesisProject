import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6,5))
verify = False  # verifies the formulas by recreating plots in book re-entry systems by Mooij
textsize= 12

jupiter_scale_height = 27e3  # m      https://web.archive.org/web/20111013042045/http://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html
g_earth = 9.81  # m/s^2
entry_velocity = 59.5e3  # m/s

if verify:
    entry_velocity = 7e3  # m/s
    jupiter_scale_height = 7200  # m (earth scale height)

lift_over_drag = np.linspace(0.1, 5, 50)
fpa_entry_angles = np.array([-1.5, -3., -4.5, -6.]) * np.pi/180


for fpa_entry in fpa_entry_angles:

    term_1 = lift_over_drag / (2*(1 + np.cos(fpa_entry)))
    term_2 = 1 - np.sqrt(1 + (4 * np.sin(fpa_entry)**2)/(lift_over_drag**2))
    fpa_max_acceleration = 2 * np.arctan(term_1 * term_2)

    beta_param = 1 / jupiter_scale_height

    av_max_over_g = beta_param * entry_velocity ** 2 / (g_earth * lift_over_drag) \
                  * (np.cos(fpa_max_acceleration) - np.cos(fpa_entry)) \
                  * np.exp(- 2 * (fpa_max_acceleration - fpa_entry) / lift_over_drag)

    a_total_max_over_g = av_max_over_g * np.sqrt(1 + lift_over_drag**2)

    ax.plot(lift_over_drag, a_total_max_over_g, label=fr'$\gamma$ = {fpa_entry * 180 / np.pi}°')
    ax.annotate(fr'$\gamma$ = {fpa_entry * 180 / np.pi}°', (lift_over_drag[6],a_total_max_over_g[6]), (0,3),textcoords='offset points', size= textsize)

ax.set_xlabel('$L/D$ [-]', size=textsize)
ax.set_ylabel(r'$a_{res}/g$ [-]', size = textsize)
ax.axhline(y=75, linestyle='--', color='gray')
ax.annotate('75 g', (4.7,75), (0,4), textcoords='offset points', color='gray', size = textsize)
# ax.legend()
plt.show()