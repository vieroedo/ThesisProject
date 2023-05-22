from handle_functions import *
import scipy.optimize as optimize

# desired_value, tolerance, max_iter = 0., 1e-15, 1000


def x_tau_function(tau, a1, a2, a3):
    sin_beta_param = a2 / np.sqrt(a2 ** 2 - 4 * a1 * a3)
    beta_param = np.arcsin(sin_beta_param)
    x_tau = - a2 / (2 * a1) - np.sqrt(a2 ** 2 - 4 * a1 * a3) / (2 * a1) * np.sin(np.sqrt(-a1) * tau - beta_param)
    return x_tau

def x_tau_function_find_max(tau, a1, a2, a3):
    sin_beta_param = a2 / np.sqrt(a2 ** 2 - 4 * a1 * a3)
    beta_param = np.arcsin(sin_beta_param)
    x_tau = - a2 / (2 * a1) - np.sqrt(a2 ** 2 - 4 * a1 * a3) / (2 * a1) * np.sin(np.sqrt(-a1) * tau - beta_param)
    return - x_tau


def phi_function_zero_first_order(x_tau, entry_phi_fpa_dep_variable, k1, alpha_const):
    c = entry_phi_fpa_dep_variable
    phi_squared = c ** 2 - 2 * k1 * (np.exp(x_tau) - 1) - 2 * (1 - alpha_const) * x_tau
    return phi_squared


def phi_function_zero_second_order(x_tau, delta_parameter, c, k1, k_lower, alpha_const, eta_const, lambda_constant):

    x_0_sol = regula_falsi_illinois((0,10), phi_function_zero_first_order, tolerance=1e-16, entry_phi_fpa_dep_variable=c, k1=k1, alpha_const=alpha_const)
    # x_0, = secant_method(phi_function_zero_first_order,5, 10, entry_phi_fpa_dep_variable=c, k1=k1, alpha_const=alpha_const)
    x_0 = x_0_sol[0]

    a1 = 3 / x_0 * (2 * (1 - alpha_const) - (c ** 2 + 4 * k1) / x_0 + 4 * k1 * (np.exp(x_0) - 1) / (x_0 ** 2))
    a2 = -6 * (1 - alpha_const) + 2 * (c ** 2 + 6 * k1) / x_0 - 12 * k1 * (np.exp(x_0) - 1) / (x_0 ** 2)
    a3 = c ** 2

    tau_0 = 0.
    tau_exit = (np.pi + 2 * np.arcsin(a2 / np.sqrt(a2 ** 2 - 4 * a1 * a3))) * 1 / np.sqrt(-a1)


    # tau__x_0_sol = regula_falsi_illinois((tau_0,tau_exit), x_tau_function,x_0,  a1=a1, a2=a2, a3=a3)
    tau__x_0_sol = optimize.fmin(x_tau_function_find_max, (tau_0+tau_exit)/2, (a1, a2, a3), disp=False)
    # tau__x_0, = secant_method(x_tau_function,5, 10, x_0, a1=a1, a2=a2, a3=a3)
    tau__x_0 = tau__x_0_sol[0]

    # BECAUSE: phi_a becomes ZERO at x_0. We might experience numerical errors sooo we keep it like this hehe
    if a1 * x_0 ** 2 + a2 * x_0 + a3 < 0:
        phi_a__fpa_dep_var = 0
    else:
        phi_a__fpa_dep_var = delta_parameter * np.sqrt(a1 * x_0 ** 2 + a2 * x_0 + a3)

    integral_I1 = 0.5 * phi_a__fpa_dep_var * x_0 - a2 / (4 * a1) * (c - phi_a__fpa_dep_var) + (
                4 * a1 * a3 - a2 ** 2) / (8 * a1) * tau__x_0

    integral_I2 = -1 / a1 * (c - phi_a__fpa_dep_var) - a2 / (2 * a1) * tau__x_0

    factor1 = (1 + lambda_constant ** 2) / k1
    factor2 = factor1 * (1 - alpha_const)
    integral_I = factor1 * c * x_0 - 0.5 * k_lower * alpha_const * x_0 ** 2 - \
                 factor2 * tau__x_0 * x_0 - factor1 * integral_I1 + factor2 * integral_I2
    phi_squared = c**2 - 2*k1*(np.exp(x_tau)-1) -2*(1-alpha_const)*x_tau + 2*eta_const*alpha_const*integral_I

    return phi_squared


def calculate_tau_boundaries_second_order_equations(entry_radius, entry_velocity, entry_flight_path_angle,
                                                    C_D0 = vehicle_cd, C_L = vehicle_cl, K_hypersonic = 0.,
                                                    vehicle_mass = vehicle_mass,
                                                    vehicle_surface = vehicle_reference_area):

    entry_g_acc = jupiter_gravitational_parameter / entry_radius ** 2
    entry_density = jupiter_atmosphere_exponential(entry_radius - jupiter_radius)

    entry_phi__fpa_dep_variable = - np.sqrt(entry_radius / jupiter_scale_height) * np.sin(entry_flight_path_angle)

    E_star = 1 / (2 * np.sqrt(K_hypersonic * C_D0))

    B_constant = (entry_density * entry_radius * vehicle_surface) / (2 * vehicle_mass) * np.sqrt(C_D0 / K_hypersonic)

    lambda_constant = C_L / np.sqrt(C_D0 / K_hypersonic)

    eta_constant = (entry_density * vehicle_surface * C_D0) / vehicle_mass * np.sqrt(
        jupiter_scale_height * entry_radius)

    c_const = entry_phi__fpa_dep_variable
    k_lower = 2 * E_star / (np.sqrt(entry_radius / jupiter_scale_height) * B_constant)
    k1 = B_constant * lambda_constant
    alpha_constant = entry_g_acc * entry_radius / entry_velocity ** 2

    x1_sol = regula_falsi_illinois((0, 10), phi_function_zero_second_order,
                                delta_parameter=1, c=c_const, k1=k1, k_lower=k_lower,
                                alpha_const=alpha_constant, eta_const=eta_constant, lambda_constant=lambda_constant)
    x1 = x1_sol[0]

    a1 = 3 / x1 * (2 * (1 - alpha_constant) - (c_const ** 2 + 4 * k1) / x1 + 4 * k1 * (np.exp(x1) - 1) / (x1 ** 2))
    a2 = -6 * (1 - alpha_constant) + 2 * (c_const ** 2 + 6 * k1) / x1 - 12 * k1 * (np.exp(x1) - 1) / (x1 ** 2)
    a3 = c_const ** 2

    tau_0 = 0 # atmospheric entry
    # tau_0 = -2 * np.arcsin(a2 / np.sqrt(a2 ** 2 - 4 * a1 * a3)) * 1 / np.sqrt(-a1)
    # tau_exit = np.pi + 2*np.arcsin(a2/np.sqrt(a2**2-4*a1*a3) * 1/np.sqrt(-a1))
    tau_exit = (np.pi + 2 * np.arcsin(a2 / np.sqrt(a2 ** 2 - 4 * a1 * a3))) * 1 / np.sqrt(-a1)

    # tau_x1_sol = regula_falsi_illinois((tau_0,tau_exit), x_tau_function,x1, a1=a1, a2=a2, a3=a3)
    tau_x1_sol = optimize.fmin(x_tau_function_find_max, (tau_0+tau_exit)/2, (a1, a2, a3), disp=False)
    tau_x1 = tau_x1_sol[0]

    return tau_0, tau_x1, tau_exit, a1, a2, a3


def second_order_approximation_aerocapture(tau, tau_x1, a1, a2, a3, entry_radius, entry_velocity, entry_flight_path_angle,
                                           C_D0 = vehicle_cd, C_L = vehicle_cl, K_hypersonic = 0.,
                                           vehicle_mass = vehicle_mass, vehicle_surface = vehicle_reference_area,
                                           nose_radius = vehicle_nose_radius):
    # finish it so that it does everything of the entry traj

    # delta_parameter = ...  # + or - 1
    # tau = np.sqrt(entry_radius/jupiter_scale_height) * theta__range_angle
    # velocity = ...


    entry_g_acc = jupiter_gravitational_parameter / entry_radius ** 2
    entry_density = jupiter_atmosphere_exponential(entry_radius-jupiter_radius)

    entry_phi__fpa_dep_variable = - np.sqrt(entry_radius / jupiter_scale_height) * np.sin(entry_flight_path_angle)

    # h_penetration_depth = (radius - entry_radius) / entry_radius
    # u_velocity_nondim = velocity**2 / (entry_g_acc * entry_radius)

    E_star = 1/(2 * np.sqrt(K_hypersonic * C_D0))

    B_constant = (entry_density * entry_radius * vehicle_surface)/(2*vehicle_mass) * np.sqrt(C_D0/K_hypersonic)

    lambda_constant = C_L/np.sqrt(C_D0/K_hypersonic)

    # y = np.exp(-entry_radius*h_penetration_depth/jupiter_scale_height)

    eta_constant = (entry_density * vehicle_surface * C_D0) / vehicle_mass * np.sqrt(jupiter_scale_height*entry_radius)

    # velocity_dep_variable = 1/eta_constant * 2 * np.log(entry_velocity/velocity)

    # fpa_dep_variable = - np.sqrt(entry_radius/jupiter_scale_height) * np.sin(flight_path_angle)

    c_const = entry_phi__fpa_dep_variable
    k_lower = 2 * E_star / (np.sqrt(entry_radius/jupiter_scale_height) * B_constant)
    k1 = B_constant * lambda_constant
    alpha_constant = entry_g_acc * entry_radius / entry_velocity**2


    # x1, = regula_falsi_illinois((-1000, 1000), phi_function_zero_second_order,
    #                            delta_parameter=delta_parameter, c=c_const, k1=k1, k_lower=k_lower,
    #                            alpha_const=alpha_constant, eta_const=eta_constant, lambda_constant=lambda_constant)
    #
    #
    # a1 = 3 / x1 * ( 2 * (1 - alpha_constant) - (c_const**2+4*k1)/x1 + 4*k1*(np.exp(x1)-1)/(x1**2))
    # a2 =  -6 * (1 - alpha_constant) - 2*(c_const**2+6*k1)/x1 + 12*k1*(np.exp(x1)-1)/(x1**2)
    # a3 = c_const**2

    x_tau = x_tau_function(tau, a1, a2, a3)

    delta_parameter = np.ones(len(tau)) * -1
    delta_equal_1_cells = np.where(tau < tau_x1)
    delta_parameter[delta_equal_1_cells] = 1
    # if tau < tau_x1:
    #     delta_parameter = 1
    # else:
    #     delta_parameter = -1

    phi_a__fpa_dep_var = delta_parameter * np.sqrt(a1* x_tau**2 + a2*x_tau + a3)

    integral_I1 = 0.5 * phi_a__fpa_dep_var * x_tau - a2/(4*a1) * (c_const-phi_a__fpa_dep_var) + (4*a1*a3-a2**2)/(8*a1)*tau

    integral_I2 = -1/a1 * (c_const - phi_a__fpa_dep_var) - a2/(2*a1) * tau

    factor1 = (1+ lambda_constant**2)/k1
    factor2 = factor1 * (1-alpha_constant)

    integral_I = factor1 * c_const * x_tau - 0.5*k_lower*alpha_constant*x_tau**2 - \
                 factor2 * tau * x_tau - factor1 * integral_I1 + factor2 * integral_I2

    integral_J = factor1 * c_const * tau - factor1 * x_tau - 0.5 * factor2*tau**2 - k_lower * alpha_constant * integral_I2

    phi__fpa_dep_var = delta_parameter * np.sqrt(c_const**2 - 2*k1*(np.exp(x_tau)-1) -2*(1-alpha_constant)*x_tau +
                                                 2*eta_constant*alpha_constant*integral_I)

    nu__velocity_dep_var = factor1 * (c_const - phi__fpa_dep_var) - k_lower * alpha_constant * x_tau - \
                           factor2 * tau - eta_constant * k_lower * alpha_constant * integral_I + \
                           eta_constant * alpha_constant * factor1 * integral_J


    exp_argument = x_tau - eta_constant * nu__velocity_dep_var
    drag = entry_g_acc * (1+ lambda_constant**2)*B_constant*np.exp(exp_argument) / (2*alpha_constant*E_star)
    lift = entry_g_acc * lambda_constant * B_constant * np.exp(exp_argument)/alpha_constant

    radius = - jupiter_scale_height * x_tau + entry_radius
    velocity = np.sqrt(entry_velocity**2/np.exp(eta_constant*nu__velocity_dep_var))
    flight_path_angle = np.arcsin(phi__fpa_dep_var/(-np.sqrt(entry_radius/jupiter_scale_height)))
    density = jupiter_atmosphere_exponential(radius - jupiter_radius)

    wall_heat_flux = 0.6556E-8 * (entry_density/nose_radius)**0.5 * entry_velocity**3 * np.exp(0.5*x_tau - 3/2*eta_constant*nu__velocity_dep_var)

    range_angle = tau * np.sqrt(
        jupiter_scale_height / (atmospheric_entry_altitude + jupiter_radius))

    return_values = (radius, velocity, flight_path_angle, density, drag, lift, wall_heat_flux, range_angle)
    additional_values = []
    return return_values, additional_values

