import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from handle_functions import plot_grid

########################################################################################################################
# GENERAL-TASK PLOTS ###################################################################################################
########################################################################################################################

decision_variable_list = ['InterplanetaryVelocity', 'EntryFpa']
objectives_list = ['payload_mass_fraction', 'total_radiation_dose_krad', 'benefit_over_insertion_burn']
total_list = decision_variable_list + objectives_list

decision_variable_nice_labels = [r'$V_{J\,\infty}$ [m/s]', r'$\gamma_E$ [deg]']
objectives_nice_labels = [r'$f_{payload}$ [-]', 'Radiation Dose [krad]', 'Aerocapture Benefit [-]']
total_nice_names = decision_variable_nice_labels + objectives_nice_labels


def get_label(quantity: str):
    if quantity not in total_list:
        raise ValueError(f'Invalid name for quantity. Available options are {total_list}')
    if len(total_nice_names) != len(total_list):
        raise Exception('We have a problem here with labeling hehe. go check the global variables')

    quantity_index = total_list.index(quantity)
    selected_label = total_nice_names[quantity_index]
    return selected_label


def plot_with_colorbar(df: pd.DataFrame,
                       x_axis: str, y_axis: str, coloring: str,
                       directly_show: bool = False,
                       figsize: tuple = (10,8), fontsize=15,
                       xlabel='', ylabel='', colorbarlabel=''):
    if x_axis not in df.columns:
        raise ValueError(f'Invalid name for x_axis. available options are {df.columns}')
    if y_axis not in df.columns:
        raise ValueError(f'Invalid name for y_axis. available options are {df.columns}')
    if coloring not in df.columns:
        raise ValueError(f'Invalid name for coloring. available options are {df.columns}')

    xlabel = get_label(x_axis) if xlabel == '' else xlabel
    ylabel = get_label(y_axis) if ylabel == '' else ylabel
    colorbarlabel = get_label(coloring) if colorbarlabel == '' else colorbarlabel

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(df[x_axis], df[y_axis], c=df[coloring], cmap='viridis')

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(colorbarlabel, size=fontsize)
    cbar.ax.tick_params(labelsize=fontsize*5/6)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    # ax.set_title(f'Decision Variables with color gradient for the {coloring}', fontsize=fontsize)
    ax = plot_grid(ax, fontsize*5/6)

    if directly_show:
        plt.show()

    return fig, ax


########################################################################################################################
# SPECIFIC-TASK PLOTS ##################################################################################################
########################################################################################################################


def plot_decision_variables(df, decision_variables, objective, objectives_list):
    if objective not in objectives_list:
        raise ValueError(f"Invalid objective. Available options are: {objectives_list}")

    decision_var1 = decision_variables[0]
    decision_var2 = decision_variables[1]

    plt.figure(figsize=(10, 8))
    plt.scatter(df[decision_var1], df[decision_var2], c=df[objective], cmap='viridis')
    plt.colorbar(label=objective)
    plt.xlabel(decision_var1)
    plt.ylabel(decision_var2)
    plt.title(f'Decision Variables with color gradient for {objective}')
    plt.grid(True)
    plt.show()


def plot_decision_variables_3D(df, decision_variables, objective, objectives_list):
    if objective not in objectives_list:
        raise ValueError(f"Invalid objective. Available options are: {objectives_list}")

    decision_var1 = decision_variables[0]
    decision_var2 = decision_variables[1]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(df[decision_var1], df[decision_var2], df[objective], c=df[objective], cmap='viridis')
    fig.colorbar(scatter, label=objective, shrink=0.6)

    ax.set_xlabel(decision_var1)
    ax.set_ylabel(decision_var2)
    ax.set_zlabel(objective)

    plt.title(f'3D plot of Decision Variables with color gradient for {objective}')
    plt.grid(True)
    plt.show()


def plot_decision_variables_3D_ver2(df, decision_variables, objective, objectives_list):
    if objective not in objectives_list:
        raise ValueError(f"Invalid objective. Available options are: {objectives_list}")

    decision_var1 = decision_variables[0]
    decision_var2 = decision_variables[1]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # scatter = ax.scatter(df[decision_var1], df[decision_var2], df[objective], c=df[objective], cmap='viridis')
    # fig.colorbar(scatter, label=objective, shrink=0.6)

    # Projecting onto the x-y plane
    scatter = ax.scatter(df[decision_var1], df[decision_var2], zs=df[objective].min(), zdir='z', s=20, c=df[objective], cmap='viridis', depthshade=False)

    # Projecting onto the y-z plane
    ax.scatter(df[decision_var1].min(), df[decision_var2], df[objective], zdir='z', s=20, c=df[objective], cmap='viridis', depthshade=False)

    # Projecting onto the x-z plane
    ax.scatter(df[decision_var1], df[decision_var2].max(), df[objective], zdir='z', s=20, c=df[objective], cmap='viridis', depthshade=False)

    fig.colorbar(scatter, ax=ax, label=objective, shrink=0.6)


    ax.set_xlabel(decision_var1)
    ax.set_ylabel(decision_var2)
    ax.set_zlabel(objective)

    plt.title(f'3D plot of Decision Variables with color gradient for {objective}')
    plt.grid(True)
    plt.show()

