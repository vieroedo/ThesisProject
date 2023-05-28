import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from handle_functions import plot_grid
from tudatpy.util import pareto_optimums
from tudatpy import plotting
from auto_annotate import AutoAnnotate
from adjustText import adjust_text
import copy as cp
import os

########################################################################################################################
# GENERAL-TASK PLOTS ###################################################################################################
########################################################################################################################

decision_variable_list = ['InterplanetaryVelocity', 'EntryFpa']
objectives_list = ['payload_mass_fraction', 'total_radiation_dose_krad', 'benefit_over_insertion_burn']
total_list = decision_variable_list + objectives_list + ['final_eccentricity']

decision_variable_nice_labels = [r'$V_{J\,\infty}$ [m/s]', r'$\gamma_E$ [deg]']
objectives_nice_labels = [r'$f_{payload}$ [-]', 'Radiation Dose [krad]', 'Aerocapture Benefit [-]']
total_nice_names = decision_variable_nice_labels + objectives_nice_labels + [r'$e_{final}$ [-]']

decision_variable_abbreviations = decision_variable_list
objectives_abbreviations = ['payloadMF', 'RadDose', 'benefitAE']
total_abbreviations = decision_variable_abbreviations + objectives_abbreviations + ['final_eccentricity']

def get_label(quantity: str):
    if quantity not in total_list:
        raise ValueError(f'Invalid name for quantity. Available options are {total_list}')
    if len(total_nice_names) != len(total_list):
        raise Exception('We have a problem here with labeling hehe. go check the global variables')

    quantity_index = total_list.index(quantity)
    selected_label = total_nice_names[quantity_index]
    return selected_label


def get_abbreviation(quantity: str):
    if quantity not in total_list:
        raise ValueError(f'Invalid name for quantity. Available options are {total_list}')
    if len(total_abbreviations) != len(total_list):
        raise Exception('We have a problem here with labeling hehe. go check the global variables')

    quantity_index = total_list.index(quantity)
    selected_abbreviation = total_abbreviations[quantity_index]
    return selected_abbreviation


def plot_with_colorbar(df: pd.DataFrame,
                       x_axis: str, y_axis: str, coloring: str,
                       directly_show: bool = False,
                       figsize: tuple = (10,8), fontsize=15,
                       xlabel='', ylabel='', colorbarlabel='',
                       save_fig: bool = False,
                       save_dir: str = ''):

    if x_axis not in df.columns:
        raise ValueError(f'Invalid name for x_axis. available options are {df.columns}')
    if y_axis not in df.columns:
        raise ValueError(f'Invalid name for y_axis. available options are {df.columns}')
    if coloring not in df.columns:
        raise ValueError(f'Invalid name for coloring. available options are {df.columns}')
    if save_fig and save_dir == '':
        raise ValueError('No directory selected for saving the file.')

    xlabel = get_label(x_axis) if xlabel == '' else xlabel
    ylabel = get_label(y_axis) if ylabel == '' else ylabel
    colorbarlabel = get_label(coloring) if colorbarlabel == '' else colorbarlabel

    fig, ax = plt.subplots(figsize=figsize) #, constrained_layout=True)
    scatter = ax.scatter(df[x_axis], df[y_axis], c=df[coloring], cmap='viridis')

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(colorbarlabel, size=fontsize)
    cbar.ax.tick_params(labelsize=fontsize*5/6)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    # ax.set_title(f'Decision Variables with color gradient for the {coloring}', fontsize=fontsize)
    ax = plot_grid(ax, fontsize*5/6)

    locs = ax.get_xticks()
    # Set new ticks that skip every other one
    ax.set_xticks(locs[::2])

    if save_fig:
        os.mkdir(save_dir) if not os.path.exists(save_dir) else None
        filename = f'{get_abbreviation(x_axis)}_vs_{get_abbreviation(y_axis)}__{get_abbreviation(coloring)}.png'
        fig.savefig(save_dir + filename)

    if directly_show:
        plt.show()

    return fig, ax


def pareto_front_plot(df: pd.DataFrame,
                      objective1: str, objective2: str,
                      decision_variable: str,
                      are_objectives_min_max: list = (max, max),
                      best_candidate_index = None,
                      plot_title: str = '',
                      show_optimum_labels: bool = False,
                      save_fig: bool = False,
                      pareto_zoom: bool = False,
                      save_dir: str = '',
                      fontsize: int = 15
                      ) -> (plt.Figure, plt.Axes):

    if objective1 not in objectives_list or objective2 not in objectives_list:
        raise ValueError(f'Invalid name for one of the two objectives. Allowed values are {objectives_list}')
    if decision_variable not in decision_variable_list + ['final_eccentricity']:
        raise ValueError(f'Invalid name for the selected decision variable. Allowed values are {decision_variable_list}')
    if save_fig and save_dir == '':
        raise ValueError('No directory selected for saving the file.')

    objective_points = df.loc[:, [objective1, objective2]].to_numpy()
    optimum_mask = pareto_optimums(objective_points, are_objectives_min_max)
    optimum_points = objective_points[optimum_mask]

    fig, ax = plotting.pareto_front(
        x_objective=df[objective1],
        y_objective=df[objective2],
        x_label=get_label(objective1),
        y_label=get_label(objective2),
        title=plot_title,
        operator=are_objectives_min_max,
        c_parameter=df[decision_variable],
        c_label=get_label(decision_variable),
        cmap="plasma",
        alpha=0.65,
        fontsize=fontsize
    )
    # ax.set_xlabel(get_label(objective1), fontsize=fontsize)
    # ax.set_ylabel(get_label(objective2), fontsize=fontsize)
    ax.scatter(optimum_points[:, 0], optimum_points[:, 1], color='r', marker='o', facecolor='none')


    filename = f'{get_abbreviation(objective1)}_vs_{get_abbreviation(objective2)}__{get_abbreviation(decision_variable)}_paretoFront.png'

    if pareto_zoom:
        ax = find_best_candidate(df, ax, objective1, objective2, optimum_points, best_candidate_index,
                                 show_optimum_labels=True, fontsize=fontsize)

        # fig_zoom = cp.deepcopy(fig)
        # ax_zoom = cp.deepcopy(ax)
        xdistance = abs(min(optimum_points[:, 0]) - max(optimum_points[:, 0]))
        ydistance = abs(min(optimum_points[:, 1]) - max(optimum_points[:, 1]))
        xlim = (min(optimum_points[:, 0]) - xdistance/2, max(optimum_points[:, 0]) + xdistance/2)
        ylim = (min(optimum_points[:, 1]) - ydistance/2, max(optimum_points[:, 1]) + ydistance/2)
        zoom_figure(fig, ax, xlim, ylim) #, save_fig=True, save_dir=save_dir + '/zooms/', filename=filename)
        filename = 'zoom_' + filename
        save_dir = save_dir + '/zooms/'

    fig.tight_layout()

    if save_fig:
        os.mkdir(save_dir) if not os.path.exists(save_dir) else None
        fig.savefig(save_dir + filename)

    return fig, ax


def find_best_candidate(df: pd.DataFrame, ax: plt.Axes, objective1, objective2, optimum_points: np.ndarray, best_candidate_index = None, show_optimum_labels: bool = False, fontsize: int = 15):
    opt_points_shape = np.shape(optimum_points)
    if opt_points_shape[1] != 2:
        raise ValueError('Invalid number of columns for optimum points. it must be 2')
    selected_objectives_df = pd.DataFrame(optimum_points, columns=[objective1, objective2])
    mask1 = df[objective1].isin(selected_objectives_df[objective1])
    mask2 = df[objective2].isin(selected_objectives_df[objective2])
    optimum_points_df = df[mask1 & mask2]

    # if best_candidate_index == -1:
    #     print('No candidate(s) selected. List of all the optimum points found:')
    #     print(optimum_points_df)
    #     return ax
    #
    if best_candidate_index is not None:
        if type(best_candidate_index) not in [list, tuple]:
            raise TypeError('indices for best candidate(s) are accepted only in iterable type (list, tuple)')
        number_of_candidates = len(best_candidate_index)
        for i in range(number_of_candidates):
            optimal_solution = optimum_points_df.loc[best_candidate_index[i]]
            print('\n################################################################')
            print(f'Values of the optimal solution for objectives {get_label(objective1)}, {get_label(objective2)}:')
            print(optimal_solution)
            print('\n')

            annotation_position = (optimal_solution[objective1], optimal_solution[objective2])
            ax.annotate(str(best_candidate_index[i]), xy=annotation_position, xytext=(-80, -20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->'), fontsize=fontsize*5/6)
        return ax

    print('\n################################################################')
    print('No candidate(s) selected. List of all the optimum points found:')
    print(optimum_points_df)
    print('\n')

    number_of_points = len(optimum_points[:,0])
    indices = list(optimum_points_df.index)
    if show_optimum_labels:
        for i in range(number_of_points):
            annotation_position = (optimum_points[i,0], optimum_points[i,1])
            ax.annotate(f'{indices[i]}', xy=annotation_position, xytext=(-20*((-1)**i),20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->'), fontsize=fontsize*5/6)
    # texts = [plt.text(x, y, label) for x, y, label in zip(optimum_points[:,0], optimum_points[:,1], indices)]
    # adjust_text(texts)

    return ax


def zoom_figure(fig: plt.Figure, ax: plt.Axes,
                xlim=None, ylim=None,
                save_fig: bool = False, save_dir: str = '', filename: str = ''):
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()
    if len(xlim) != 2 or len(ylim) != 2:
        raise ValueError('Invalid xlim or ylim values. Couple of values are accepted.')
    if (save_fig and save_dir == '') or (save_fig and filename == ''):
        raise ValueError('No directory selected for saving the file.')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if save_fig:
        os.mkdir(save_dir) if not os.path.exists(save_dir) else None
        fig.savefig(save_dir + 'zoom_' + filename)

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


def plot_decision_variables_3D(df, decision_variables, objective, objectives_list, directly_show: bool = False):
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

    if directly_show:
        plt.show()

    return fig, ax


def plot_decision_variables_3D_ver2(df, decision_variables, objective, objectives_list, directly_show: bool = False):
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

    if directly_show:
        plt.show()

    return fig, ax

