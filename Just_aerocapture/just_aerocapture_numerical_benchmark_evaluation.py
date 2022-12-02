# General imports
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from time import process_time as pt


# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators

# Problem-specific imports
import CapsuleEntryUtilities as Util

# check_benchmarks = False
# generate_benchmarks_bool = False

write_results_to_file = True  # when in doubt leave true (idk anymore what setting it to false does hehe)


use_benchmark = True
# if use_benchmark is True ####################################################
generate_benchmarks = True

playwith_benchmark = False
silence_benchmark_related_plots = True

plot_error_wrt_benchmark = True

benchmark_portion_to_evaluate = 3  # 0, 1, 2, 3 (3 means all three together)

# If you play with benchmarks
lower_limit = 10e3
upper_limit = 160e3
no_of_entries = 31

# If you set a single step_size
choose_step_size = 40e3

# If you set dedicated step sizes for case 3
set_dedicated_step_sizes = True
dedicated_step_sizes = [8e3, 4, 8e3]

# For both cases
reduce_step_size = 1

change_coefficient_set = False
choose_coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_45

plot_fit_to_benchmark_errors = False
choose_ref_value_cell = 11
global_truncation_error_power = 8  # 7
#####################################################################################


current_dir = os.path.dirname(__file__)

# Load spice kernels
spice_interface.load_standard_kernels()

# shape_parameters = [8.148730872315355,
#                     2.720324489288032,
#                     0.2270385167794302,
#                     -0.4037530896422072,
#                     0.2781438040896319,
#                     0.4559143679738996]

# Atmospheric entry conditions
atmospheric_entry_interface_altitude = 400e3  # m (DO NOT CHANGE - consider changing only with valid and sound reasons)
flight_path_angle_at_atmosphere_entry = -2.1  # degrees


###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Set simulation start epoch
simulation_start_epoch = 11293 * constants.JULIAN_DAY  # s
# Set termination conditions
maximum_duration = 85 * constants.JULIAN_DAY  # s
# termination_altitude = 270.0E3  # m
# Set vehicle properties
# capsule_density = 250.0  # kg m-3


###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Define settings for celestial bodies
bodies_to_create = ['Jupiter']
# Define coordinate system
global_frame_origin = 'Jupiter'
global_frame_orientation = 'ECLIPJ2000'

# Create body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

# Add Jupiter exponential atmosphere
jupiter_scale_height = 27e3  # m      https://web.archive.org/web/20111013042045/http://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html
jupiter_1bar_density = 0.16  # kg/m^3
density_scale_height = jupiter_scale_height
density_at_zero_altitude = jupiter_1bar_density
body_settings.get('Jupiter').atmosphere_settings = environment_setup.atmosphere.exponential(
        density_scale_height, density_at_zero_altitude)

# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Create and add capsule to body system
# NOTE TO STUDENTS: When making any modifications to the capsule vehicle, do NOT make them in this code, but in the
# add_capsule_to_body_system function
# Util.add_capsule_to_body_system(bodies,
#                                 shape_parameters,
#                                 capsule_density)


# Create vehicle object
bodies.create_empty_body('Capsule')

# Set mass of vehicle
bodies.get_body('Capsule').mass = 2000  # kg

# Create aerodynamic coefficients interface (drag and lift only)
reference_area = 5.  # m^2
drag_coefficient = 1.2
lift_coefficient = 0.6
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area, [drag_coefficient, 0.0, lift_coefficient])  # [Drag, Side-force, Lift]
environment_setup.add_aerodynamic_coefficient_interface(
                bodies, 'Capsule', aero_coefficient_settings )


###########################################################################
# CREATE (CONSTANT) PROPAGATION SETTINGS ##################################
###########################################################################

# Retrieve termination settings
termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                     maximum_duration,
                                                     )
# Retrieve dependent variables to save
dependent_variables_to_save = Util.get_dependent_variable_save_settings()
# Check whether there is any
are_dependent_variables_to_save = False if not dependent_variables_to_save else True


###########################################################################
# RUN SIMULATION #####################################
###########################################################################

# Get current propagator, and define propagation settings
current_propagator = propagation_setup.propagator.unified_state_model_quaternions
current_propagator_settings = Util.get_propagator_settings(flight_path_angle_at_atmosphere_entry,
                                                           atmospheric_entry_interface_altitude,
                                                           bodies,
                                                           termination_settings,
                                                           dependent_variables_to_save,
                                                           current_propagator)


settings_index = -4
# Create integrator settings
current_integrator_settings = Util.get_integrator_settings(settings_index,
                                                           simulation_start_epoch)

# if check_benchmarks:
    # step_sizes = np.linspace(100, 100000, 10)
    # max_error =
    # for step_size in step_sizes:
    #     benchmark_integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
    #         simulation_start_epoch,
    #         step_size,
    #         propagation_setup.integrator.CoefficientSets.rkdp_87,
    #         step_size,
    #         step_size,
    #         np.inf,
    #         np.inf)
    #     dynamics_simulator = numerical_simulation.SingleArcSimulator(
    #         bodies, current_integrator_settings, current_propagator_settings, print_dependent_variable_data=False)
    #     state_history = dynamics_simulator.state_history
    #     simulation_result = np.vstack(list(state_history.values()))
    #     position_states = simulation_result[:,0:3]
    #     norm_of_position_states = LA.norm(position_states, axis=1)
    #
    # subfolder = '/benchmarks/'
    # output_path = current_dir + subfolder
    # if generate_benchmarks_bool:
    #     benchmarks = Util.generate_benchmarks(50, simulation_start_epoch, bodies, current_propagator_settings, are_dependent_variables_present=True, output_path=output_path)
    #     bmk1 = benchmarks[0]
    #     bmk2 = benchmarks[1]
    #     bench_diff = Util.compare_benchmarks(bmk1, bmk2, output_path, 'bench_diff.dat')
    #
    #     bench_diff_epochs = bench_diff.keys()
    #     bench_diff_error = LA.norm(np.vstack(list(bench_diff.values()))[:,0:3], axis=1)
    # else:
    #     bench_diff = np.loadtxt(output_path + 'bench_diff.dat')
    #     bench_diff_epochs = bench_diff[:,0]
    #     bench_diff_error = LA.norm(bench_diff[:,1:4], axis=1)
    #
    #
    # plt.plot(bench_diff_epochs, bench_diff_error)
    # plt.yscale('log')
    # plt.show()


###########################################################################
# IF DESIRED, GENERATE AND COMPARE BENCHMARKS #############################
###########################################################################

# NOTE TO STUDENTS: MODIFY THE CODE INSIDE THIS "IF STATEMENT" (AND CALLED FUNCTIONS, IF NEEDED)
# TO ASSESS THE QUALITY OF VARIOUS BENCHMARK SETTINGS
check_folder_existence = False
if use_benchmark:
    # Define benchmark interpolator settings to make a comparison between the two benchmarks
    benchmark_interpolator_settings = interpolators.lagrange_interpolation(
        8,boundary_interpolation = interpolators.extrapolate_at_boundary)

    # Create propagator settings for benchmark (Cowell)
    propagator_settings = current_propagator_settings

    benchmark_output_path = current_dir + '/SimulationOutput/benchmarks/' if write_results_to_file else None

    if playwith_benchmark:
        benchmark_output_path = current_dir + '/SimulationOutput/benchmarks/step_size_study/' if write_results_to_file else None

    if benchmark_portion_to_evaluate == 0:
        benchmark_output_path = benchmark_output_path + 'portion_0/' if write_results_to_file else None
        check_folder_existence = True
        divide_step_size_of = 1
    elif benchmark_portion_to_evaluate == 1:
        benchmark_output_path = benchmark_output_path + 'portion_1/' if write_results_to_file else None
        check_folder_existence = True
        divide_step_size_of = reduce_step_size
    elif benchmark_portion_to_evaluate == 2:
        benchmark_output_path = benchmark_output_path + 'portion_2/' if write_results_to_file else None
        check_folder_existence = True
        divide_step_size_of = 1
    elif benchmark_portion_to_evaluate == 3:
        benchmark_output_path = benchmark_output_path + 'full_traj/' if write_results_to_file else None
        check_folder_existence = True
        if set_dedicated_step_sizes:
            divide_step_size_of = 1
        else:
            divide_step_size_of = reduce_step_size

    # Generate benchmarks
    if playwith_benchmark:
        if generate_benchmarks:
            benchmark_step_sizes = np.linspace(lower_limit, upper_limit, no_of_entries)
            if benchmark_portion_to_evaluate != 1:
                benchmark_step_sizes = np.rint(np.linspace(lower_limit, upper_limit, no_of_entries))
        else:
            ancillary_info = np.loadtxt(benchmark_output_path + 'ancillary_benchmark_info.txt')
            lower_limit = ancillary_info[0,0]
            upper_limit = ancillary_info[-1, 0]
            no_of_entries = len(ancillary_info[:,0])
            benchmark_step_sizes = np.rint(np.linspace(lower_limit, upper_limit, no_of_entries))
            # benchmark_step_sizes = np.rint(np.linspace(36e3, 41e3, 40))
    else:
        if set_dedicated_step_sizes:
            benchmark_step_sizes = [0]
        else:
            benchmark_step_sizes = [choose_step_size]



    if check_folder_existence and generate_benchmarks:
        does_folder_exists = os.path.exists(benchmark_output_path)
        shutil.rmtree(benchmark_output_path) if does_folder_exists else None
        check_folder_existence = False

    if benchmark_portion_to_evaluate == 1 or benchmark_portion_to_evaluate == 2:
        subfolder = 'step_size_study/' if playwith_benchmark else ''
        subfolder = subfolder + 'portion_' + str(benchmark_portion_to_evaluate-1) + '/'
        pathpath = current_dir + '/SimulationOutput/benchmarks/' + subfolder
        bench_info = np.loadtxt(pathpath + 'ancillary_benchmark_info.txt')
        benchmark_initial_state = bench_info[0,2:8]
        simulation_start_epoch = bench_info[0,1]
    else:
        benchmark_initial_state = np.zeros(6)

    benchmark_info = dict()
    max_position_errors = dict()
    step_size_name = ''
    # bench_portion_name = ''
    for benchmark_step_size in benchmark_step_sizes:
        t0 = pt()
        propagator_settings = Util.get_propagator_settings(flight_path_angle_at_atmosphere_entry,
                                                           atmospheric_entry_interface_altitude,
                                                           bodies,
                                                           Util.get_termination_settings(simulation_start_epoch),
                                                           dependent_variables_to_save,
                                                           current_propagator)
        if playwith_benchmark:
            step_size_name = '_stepsize_' + str(benchmark_step_size)

        if generate_benchmarks:
            if benchmark_portion_to_evaluate == 3 and set_dedicated_step_sizes:
                benchmark_step_size = dedicated_step_sizes
                termination_epoch = 0.
            else:
                if benchmark_step_sizes[0] in benchmark_info:
                    termination_epoch = benchmark_info[benchmark_step_sizes[0]][0]
                else:
                    termination_epoch = 0.

            if change_coefficient_set:
                benchmark_coefficient_set = choose_coefficient_set
            else:
                benchmark_coefficient_set = propagation_setup.integrator.CoefficientSets.rkdp_87

            benchmark_list, final_epoch, final_state = Util.generate_benchmarks(benchmark_step_size,
                                                                                simulation_start_epoch,
                                                                                bodies,
                                                                                propagator_settings,
                                                                                are_dependent_variables_to_save,
                                                                                benchmark_output_path,
                                                                                step_size_name,
                                                                                benchmark_portion_to_evaluate,
                                                                                benchmark_initial_state,
                                                                                termination_epoch,
                                                                                divide_step_size_of,
                                                                                benchmark_coefficient_set)
            # if dedicated_step_sizes:
            #     benchmark_step_size_to_save = benchmark_step_size[0]
            # else:
            #     benchmark_step_size_to_save = benchmark_step_size
            if playwith_benchmark:
                benchmark_step_size_to_save = benchmark_step_size
                benchmark_info[benchmark_step_size_to_save] = np.array([final_epoch] + list(final_state))
            elif set_dedicated_step_sizes:
                benchmark_info['step_sizes'] = benchmark_step_size
        else:
            first_bench = np.loadtxt(benchmark_output_path + 'benchmark_1_states' + step_size_name +'.dat')
            first_bench_dep_var = np.loadtxt(benchmark_output_path + 'benchmark_1_dependent_variables' + step_size_name + '.dat')
            second_bench = np.loadtxt(benchmark_output_path + 'benchmark_2_states' + step_size_name + '.dat')
            second_bench_dep_var = np.loadtxt(benchmark_output_path + 'benchmark_2_dependent_variables' + step_size_name + '.dat')
            benchmark_list = [dict(zip(first_bench[:,0], first_bench[:,1:])),
                              dict(zip(second_bench[:,0], second_bench[:,1:])),
                              dict(zip(first_bench_dep_var[:, 0], first_bench_dep_var[:, 1:])),
                              dict(zip(second_bench_dep_var[:, 0], second_bench_dep_var[:, 1:]))
                              ]
            # benchmark_info = ... !!

        benchmark_cpu_time = pt() - t0
        # Extract benchmark states
        first_benchmark_state_history = benchmark_list[0]
        second_benchmark_state_history = benchmark_list[1]
        # Create state interpolator for first benchmark
        benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(first_benchmark_state_history,
                                                                                                benchmark_interpolator_settings)


        # Compare benchmark states, returning interpolator of the first benchmark
        benchmark_state_difference = Util.compare_benchmarks(first_benchmark_state_history,
                                                             second_benchmark_state_history,
                                                             benchmark_output_path,
                                                             'benchmarks_state_difference'+ step_size_name +'.dat')

        # Extract benchmark dependent variables, if present
        if are_dependent_variables_to_save:
            first_benchmark_dependent_variable_history = benchmark_list[2]
            second_benchmark_dependent_variable_history = benchmark_list[3]
            # Create dependent variable interpolator for first benchmark
            benchmark_dependent_variable_interpolator = interpolators.create_one_dimensional_vector_interpolator(
                first_benchmark_dependent_variable_history,
                benchmark_interpolator_settings)

            # Compare benchmark dependent variables, returning interpolator of the first benchmark, if present
            benchmark_dependent_difference = Util.compare_benchmarks(first_benchmark_dependent_variable_history,
                                                                     second_benchmark_dependent_variable_history,
                                                                     benchmark_output_path,
                                                                     'benchmarks_dependent_variable_difference' + step_size_name + '.dat')

        if not silence_benchmark_related_plots:
            fig1, ax1 = Util.plot_base_trajectory(first_benchmark_state_history)
            fig2, ax2 = Util.plot_time_step(first_benchmark_state_history)

            fig3, ax3 = Util.plot_base_trajectory(second_benchmark_state_history)
            fig4, ax4 = Util.plot_time_step(second_benchmark_state_history)


        benchmark_error = np.vstack(list(benchmark_state_difference.values()))
        bench_diff_epochs = np.array(list(benchmark_state_difference.keys()))
        bench_diff_epochs_plot = (bench_diff_epochs - bench_diff_epochs[0]) / constants.JULIAN_DAY
        benchmark_error = benchmark_error[2:-2,:]
        bench_diff_epochs_plot = bench_diff_epochs_plot[2:-2]
        bench_diff_epochs = bench_diff_epochs[2:-2]  # useless for now
        position_error = LA.norm(benchmark_error[:, 0:3], axis=1)
        max_position_error = np.amax(position_error)
        if playwith_benchmark:
            # max_position_errors[benchmark_step_size/divide_step_size_of] = max_position_error
            max_position_errors[benchmark_step_size] = max_position_error

        if not silence_benchmark_related_plots:
            fig_bm, ax_bm = plt.subplots(figsize=(6, 5))
            ax_bm.plot(bench_diff_epochs_plot, position_error)
            ax_bm.set_yscale('log')

        if not silence_benchmark_related_plots:
            plt.show()

    if write_results_to_file and generate_benchmarks:
        save2txt(benchmark_info, 'ancillary_benchmark_info.txt', benchmark_output_path)


    if playwith_benchmark:
        figg, axx = plt.subplots(figsize=(6,5))
        axx.plot(max_position_errors.keys(),max_position_errors.values(),marker='D',fillstyle='none')
        if plot_fit_to_benchmark_errors:
            a = max_position_errors[list(max_position_errors.keys())[choose_ref_value_cell]] / list(max_position_errors.keys())[choose_ref_value_cell]**global_truncation_error_power
            fitting_function = a * np.array(list(max_position_errors.keys()))**global_truncation_error_power
            axx.plot(max_position_errors.keys(), fitting_function, color='firebrick')
        axx.set_yscale('log')
        axx.set_xscale('log')
        axx.set_xlabel('time step (s)')
        axx.set_ylabel('maximum position error norm (m)')
        axx.axhline(y=0.1, color='r',linestyle=':')
        plt.show()

    if playwith_benchmark or benchmark_portion_to_evaluate != 3:
        quit()

current_propagator_settings = Util.get_propagator_settings(flight_path_angle_at_atmosphere_entry,
                                                           atmospheric_entry_interface_altitude,
                                                           bodies,
                                                           Util.get_termination_settings(simulation_start_epoch),
                                                           dependent_variables_to_save,
                                                           current_propagator)

# Create Shape Optimization Problem object
dynamics_simulator = numerical_simulation.SingleArcSimulator(
    bodies, current_integrator_settings, current_propagator_settings, print_dependent_variable_data=False )


### OUTPUT OF THE SIMULATION ###
# Retrieve propagated state and dependent variables
state_history = dynamics_simulator.state_history
unprocessed_state_history = dynamics_simulator.unprocessed_state_history
dependent_variable_history = dynamics_simulator.dependent_variable_history

if write_results_to_file:
    save2txt(state_history, 'simulation_state_history.dat', current_dir + '/SimulationOutput')
    save2txt(dependent_variable_history, 'simulation_dependent_variable_history.dat', current_dir + '/SimulationOutput')


simulation_result = np.vstack(list(state_history.values()))
epochs_vector = np.vstack(list(state_history.keys()))
epochs_plot = (epochs_vector - epochs_vector[0]) / constants.JULIAN_DAY

# Forces assessment

dependent_variables = np.vstack(list(dependent_variable_history.values()))

aero_acc = dependent_variables[:,0:3]
grav_acc = dependent_variables[:,3]
altitude = dependent_variables[:,4]
flight_path_angle = dependent_variables[:,5]
relative_speed = dependent_variables[:,6]

spacecraft_velocity_states = simulation_result[:,3:6]

drag_direction = -Util.unit_vector(spacecraft_velocity_states)
lift_direction = Util.rotate_vectors_by_given_matrix(Util.rotation_matrix(Util.z_axis, np.pi/2), drag_direction)

drag_acc = np.zeros((len(dependent_variables), 3))
lift_acc = np.zeros((len(dependent_variables), 3))
for i in range(len(dependent_variables)):
    drag_acc[i, :] = np.dot(aero_acc[i, :], drag_direction[i, :]) * drag_direction[i, :]
    lift_acc[i, :] = np.dot(aero_acc[i, :], lift_direction[i, :]) * lift_direction[i, :]

noise_level = 1e-6

drag_acc = LA.norm(drag_acc, axis=1)
lift_acc = LA.norm(lift_acc, axis=1)
drag_acc_mod = np.delete(drag_acc, np.where(drag_acc <= noise_level))
lift_acc_mod = np.delete(lift_acc, np.where(lift_acc <= noise_level))

entry_epochs_cells = list(np.where(drag_acc > noise_level)[0])


downrange = np.zeros(len(entry_epochs_cells))

for i, cell in enumerate(entry_epochs_cells):
    current_position = simulation_result[cell, 0:3]


atmosphere_interfaces_cell_no = np.where(drag_acc > 1e-2)[0][[0,-1]]

atmosphere_altitude_interfaces = altitude[atmosphere_interfaces_cell_no].reshape(2)
atmosphere_fpa_interfaces = (flight_path_angle[atmosphere_interfaces_cell_no[0]], flight_path_angle[atmosphere_interfaces_cell_no[1]])

Fig_f, axs_f = plt.subplots(3, 1, figsize=(7,7), sharex='col')

axs_f[0].plot(epochs_plot[entry_epochs_cells].reshape(len(drag_acc_mod)), drag_acc_mod, label='drag')
axs_f[0].plot(epochs_plot[np.where(lift_acc > noise_level)].reshape(len(lift_acc_mod)), lift_acc_mod, label='lift')
axs_f[0].plot(epochs_plot[entry_epochs_cells].reshape(len(drag_acc_mod)), grav_acc[entry_epochs_cells].reshape(len(drag_acc_mod)), label='gravity')
axs_f[0].set(ylabel='acceleration [m/s^2]')
axs_f[0].legend()

axs_f[1].plot(epochs_plot[entry_epochs_cells].reshape(len(drag_acc_mod)), altitude[entry_epochs_cells].reshape(len(drag_acc_mod))/1e3)
axs_f[1].axhline(y=atmosphere_altitude_interfaces[0]/1e3, color='grey', linestyle='dotted')
axs_f[1].axhline(y=atmosphere_altitude_interfaces[1]/1e3, color='grey', linestyle='dotted')
axs_f[1].set(ylabel='altitude [km]')

axs_f[2].plot(epochs_plot[entry_epochs_cells].reshape(len(drag_acc_mod)), flight_path_angle[entry_epochs_cells].reshape(len(drag_acc_mod))*180/np.pi)
axs_f[2].axhline(y=atmosphere_fpa_interfaces[0]*180/np.pi, color='grey', linestyle='dotted')
axs_f[2].axhline(y=atmosphere_fpa_interfaces[1]*180/np.pi, color='grey', linestyle='dotted')
axs_f[2].set(xlabel='epoch [days]', ylabel='f.p.a. [deg]')

Fig_vh, axs_vh = plt.subplots(figsize = (6,5))
axs_vh.plot(relative_speed[entry_epochs_cells].reshape(len(drag_acc_mod))/1e3, altitude[entry_epochs_cells].reshape(len(drag_acc_mod))/1e3)
axs_vh.set(xlabel='speed [km/s]', ylabel='altitude [km]')

### Prints for insight
print('\n')

states_to_evaluate = [0,-1]
states_names = ['initial', 'final']
states_dict = dict()
for i, current_state_to_eval in enumerate(states_to_evaluate):
    curr_state = simulation_result[current_state_to_eval, :]
    curr_position = curr_state[0:3]
    curr_velocity = curr_state[3:6]

    term1 = LA.norm(curr_velocity) ** 2 - Util.central_body_gravitational_parameter / LA.norm(curr_position)
    term2 = np.dot(curr_position, curr_velocity)
    curr_eccentricity_vector = (term1 * curr_position - term2 * curr_velocity) / Util.central_body_gravitational_parameter
    curr_eccentricity = LA.norm(curr_eccentricity_vector)

    curr_orbital_energy = Util.orbital_energy(LA.norm(curr_position), LA.norm(curr_velocity))

    add_string = ''
    conjug = ' and'

    if current_state_to_eval == -1:
        final_orbit_sma = - Util.central_body_gravitational_parameter / (2 * curr_orbital_energy)
        final_orbit_orbital_period = 2*np.pi * np.sqrt(final_orbit_sma**3/Util.central_body_gravitational_parameter)
        add_string = f' and orbital period {final_orbit_orbital_period/constants.JULIAN_DAY:.3f} days'
        conjug = ','

    states_dict[states_names[i]] = curr_orbital_energy
    print(f'The {states_names[i]} orbit has eccentricity {curr_eccentricity:.5f}{conjug} specific energy {curr_orbital_energy / 1e3:.3f} kJ/m' + add_string)

delta_E = states_dict['final'] - states_dict['initial']
print(f'The difference in orbital specific energy is {abs(delta_E/1e3):.3f} kJ/m')


# Plot 3-D Trajectory
fig = plt.figure()
ax = plt.axes(projection='3d')



# draw jupiter
u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
x = Util.jupiter_radius * np.cos(u) * np.sin(v)
y = Util.jupiter_radius * np.sin(u) * np.sin(v)
z = Util.jupiter_radius * np.cos(v)
ax.plot_wireframe(x, y, z, color="saddlebrown")

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('Jupiter aerocapture trajectory')

# draw post-ae possibly flyby moon orbit
for moon in Util.galilean_moons_data.keys():
    moon_sma = Util.galilean_moons_data[moon]['SMA']
    theta_angle = np.linspace(0, 2*np.pi, 200)
    x_m = moon_sma * np.cos(theta_angle)
    y_m = moon_sma * np.sin(theta_angle)
    z_m = np.zeros(len(theta_angle))
    ax.plot3D(x_m, y_m, z_m, 'b')

xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
ax.set_xlim3d(XYZlim)
ax.set_ylim3d(XYZlim)
ax.set_zlim3d(XYZlim * 0.75)
ax.set_aspect('auto')


ax.plot3D(simulation_result[:,0], simulation_result[:,1], simulation_result[:,2], 'gray')

if plot_error_wrt_benchmark and use_benchmark:
    fig2, ax2 = plt.subplots(figsize=(6,5))

    bench1 = np.loadtxt(benchmark_output_path + 'benchmark_1_states.dat')

    bench1_epochs = bench1[:,0]

    bench1_state_hist = dict()
    for e_num, epoch in enumerate(list(bench1_epochs)):
        bench1_state_hist[epoch] = bench1[e_num, 1:7]


    benchmark_interpolator_settings = interpolators.lagrange_interpolation(6,
             boundary_interpolation = interpolators.extrapolate_at_boundary)
    benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(bench1_state_hist,
                                                                                                    benchmark_interpolator_settings)

    state_difference = dict()

    for epoch in state_history.keys():
        state_difference[epoch] = state_history[epoch] - benchmark_state_interpolator.interpolate(epoch)



    ax2.plot(epochs_plot[8:-1], LA.norm(np.vstack(list(state_difference.values()))[:,0:3], axis=1)[8:-1])
    ax2.plot(bench_diff_epochs_plot, position_error, color='firebrick')
    ax2.set(xlabel='days', ylabel='error [m]')
    ax2.set_yscale('log')

Fig3, ax3 = plt.subplots(2, 1, figsize= (6,5))
time_steps = np.diff(epochs_vector, n=1, axis=0)
ax3[0].plot(epochs_plot[:-1], time_steps)
ax3[0].scatter(epochs_plot[:-1], time_steps)

ax3[1].plot(epochs_plot, altitude)

# Fig4, ax4 = plt.subplots(figsize=(6,5))
# bench_error = np.loadtxt(current_dir + '/SimulationOutput/benchmarks/benchmarks_state_difference.dat')
#
# bench_position_error = LA.norm(bench_error[:,1:4], axis=1)
#
# ax4.plot(bench_error[:,0], bench_position_error)
# ax4.set_yscale('log')
# ax4.set_title('Benchmark error')

plt.show()

