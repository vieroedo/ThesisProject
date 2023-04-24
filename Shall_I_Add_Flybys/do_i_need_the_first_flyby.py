from JupiterTrajectory_GlobalParameters import *
# import CapsuleEntryUtilities as Util
from handle_functions import *
from tudatpy.kernel.interface import spice_interface

# Load spice kernels
spice_interface.load_standard_kernels()

choose_first_moon = 'Callisto'
MJD200_date = 62503.9325  # 01/01/2030 62502

flight_path_angle_at_atmospheric_entry = np.deg2rad(-3)  # input value in: deg
interplanetary_arrival_velocity = 5600  # m/s
delta_angle_from_hohmann_trajectory = np.deg2rad(0.98)  # input value in: deg



J2000_date = MJD200_date - 51544
first_flyby_initial_epoch = J2000_date * constants.JULIAN_DAY

moons_and_energies = dict()

for choose_first_moon in galilean_moons_data.keys():
    # Retrieve flyby moon data: Radius, mu, velocity vector
    moon_radius = galilean_moons_data[choose_first_moon]['Radius']
    moon_SOI_radius = galilean_moons_data[choose_first_moon]['SOI_Radius']
    mu_moon = galilean_moons_data[choose_first_moon]['mu']

    if choose_first_moon == 'Callisto':
        delta_angle_from_hohmann_trajectory = np.deg2rad(0.98)  # input value in: deg
    elif choose_first_moon == 'Ganymede':
        delta_angle_from_hohmann_trajectory = np.deg2rad(0.95)
    elif choose_first_moon == 'Europa':
        delta_angle_from_hohmann_trajectory = np.deg2rad(0.92)
    elif choose_first_moon == 'Io':
        delta_angle_from_hohmann_trajectory = np.deg2rad(0.92)


    first_moon_first_flyby_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name=choose_first_moon,
        observer_body_name="Jupiter",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=first_flyby_initial_epoch)
    moon_position = first_moon_first_flyby_state[0:3]
    moon_velocity = first_moon_first_flyby_state[3:6]

    # first_arc_orbital_energy = orbital_energy(jupiter_SOI_radius, interplanetary_arrival_velocity,
    #                                           jupiter_gravitational_parameter)
    # frac_numerator = np.sqrt(2 * LA.norm(moon_position) * (
    #         first_arc_orbital_energy * LA.norm(moon_position) + jupiter_gravitational_parameter))
    # max_allowed_delta_hoh = np.arcsin(frac_numerator / (jupiter_SOI_radius * interplanetary_arrival_velocity))


    # Finding flyby pericenter
    tolerance = 1e-12
    c_point, f_c, i = regula_falsi_illinois((0., np.pi), calculate_fpa_from_flyby_geometry,
                                            flight_path_angle_at_atmospheric_entry, tolerance,
                                            illinois_addition=True,
                                            arc_1_initial_velocity=interplanetary_arrival_velocity,
                                            arc_1_initial_radius=jupiter_SOI_radius,
                                            delta_hoh=delta_angle_from_hohmann_trajectory,
                                            arc_2_final_radius=jupiter_radius + atmospheric_entry_altitude,
                                            flyby_moon=choose_first_moon,
                                            flyby_epoch=first_flyby_initial_epoch,
                                            equatorial_approximation=False)
    sigma_angle = c_point
    calculated_fpa = f_c + flight_path_angle_at_atmospheric_entry

    orbit_axis = unit_vector(np.cross(moon_position, moon_velocity))

    first_arc_angular_momentum = orbit_axis * jupiter_SOI_radius * interplanetary_arrival_velocity * np.sin(delta_angle_from_hohmann_trajectory)
    first_arc_orbital_energy = orbital_energy(jupiter_SOI_radius, interplanetary_arrival_velocity, jupiter_gravitational_parameter)


    # first_arc_semilatus_rectum = first_arc_angular_momentum ** 2 / jupiter_gravitational_parameter
    first_arc_semimajor_axis = - jupiter_gravitational_parameter / (2 * first_arc_orbital_energy)
    # first_arc_eccentricity = np.sqrt(1 - first_arc_semilatus_rectum / first_arc_semimajor_axis)

    flyby_initial_position = rotate_vectors_by_given_matrix(rotation_matrix(orbit_axis, -sigma_angle),
                                                                    unit_vector(moon_velocity)) * moon_SOI_radius

    # Energy and angular momentum for first arc are calculated above

    first_arc_arrival_position = moon_position + flyby_initial_position
    first_arc_arrival_radius = LA.norm(first_arc_arrival_position)
    first_arc_arrival_velocity = np.sqrt(
        2 * (first_arc_orbital_energy + jupiter_gravitational_parameter / first_arc_arrival_radius))

    first_arc_arrival_fpa = - np.arccos(
        LA.norm(first_arc_angular_momentum) / (first_arc_arrival_radius * first_arc_arrival_velocity))
    first_arc_arrival_velocity_vector = rotate_vectors_by_given_matrix(
        rotation_matrix(orbit_axis, np.pi / 2 - first_arc_arrival_fpa),
        unit_vector(first_arc_arrival_position)) * first_arc_arrival_velocity

    flyby_initial_velocity_vector = first_arc_arrival_velocity_vector - moon_velocity
    flyby_v_inf_t = LA.norm(flyby_initial_velocity_vector)

    v_c_surf = np.sqrt(mu_moon/moon_radius)

    delta_energy_max = 2 * LA.norm(moon_velocity) * flyby_v_inf_t / (1 + flyby_v_inf_t**2/v_c_surf**2)
    moons_and_energies[choose_first_moon] = delta_energy_max

    second_arc_minimum_orbital_energy = first_arc_orbital_energy - delta_energy_max


    print(f'\nPre flyby energy: {first_arc_orbital_energy/1e3:.3f} kJ/kg')
    print(f'Post flyby energy: {second_arc_minimum_orbital_energy/1e3:.3f} kJ/kg')
    print(f'Maximum delta energy achievable with {choose_first_moon}: {delta_energy_max/1e3:.3f} kJ/kg')

print('\n\n')
print(moons_and_energies)
print('\n')
print(first_arc_orbital_energy)

# No ofc oyu dont need it you cunt you knew it since september almost
# why did you decide to remove it just now that the damage is done