import matplotlib.pyplot as plt
import json
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns
import itertools


def plot_error_selection(ax, data, row_key, key, values, colors, add_label=False, ylabel="", xlabel="fops"):
    data_selected = data.loc[data[key].isin(values)]
    it = 0
    for i, row in data_selected.iterrows():
        iter_points = list(range(0, len(row[row_key])))
        if add_label:

            ax.semilogy(iter_points, row[row_key], c=colors[it], marker='.',
                        label='{0}, dofs={1}'.format(row['K'], row['dofs']))
        else:
            print(it)
            ax.semilogy(iter_points, row[row_key], c=colors[it], marker='.')

        it += 1


def plot_error(result_pd, **kwargs):
    J1 = "$J_C$"
    J2 = "$J_S$"

    # rows = result_pd.loc[result_pd['M'].isin(m_sizes) & result_pd['K'].isin(RK_orders) & result_pd['D'].isin(dg_orders)]
    # exclude = ~(result_pd['M'].isin([16]) & result_pd['D'].isin([1]) & result_pd['K'].isin(['RK2']))

    # rows = result_pd.loc[(result_pd['final error'] <= 3.0)]
    rows = result_pd.sort_values(by=['D', 'K', "cost_function_type"])

    colors = sns.color_palette("hls", 4)
    print(len(colors))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey="all")
    plot_error_selection(axs[0], rows, "error", "cost_function_type", [J1], colors, False)
    plot_error_selection(axs[1], rows, "error", "cost_function_type", [J2], colors, True)
    fig.legend(loc=9, ncol=4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    axs[0].set_xlabel("Iterations")
    axs[1].set_xlabel("Iterations")
    axs[0].set_title("Sensor Cost Function {}".format(J1))
    axs[1].set_title("Sensor Cost Function {}".format(J2))
    # axs[2].set_xlabel("Iterations")
    axs[0].set_ylabel("$L_2$-error")
    plt.savefig('errors_LOH1_test2.png', bbox_inches="tight")
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(10, 4), sharey="all")
    # controllability
    colors = sns.color_palette("hls", 4)
    plot_error_selection(axs[0, 0], rows, "error_sx", "cost_function_type", [J1], colors)
    plot_error_selection(axs[1, 0], rows, "error_sy", "cost_function_type", [J1], colors)
    # sensor
    plot_error_selection(axs[0, 1], rows, "error_sx", "cost_function_type", [J2], colors, True)
    plot_error_selection(axs[1, 1], rows, "error_sy", "cost_function_type", [J2], colors)
    # sensor with stabilization
    # plot_error_selection(axs[0, 2], rows, "error_sx", "cost_function_type", ["SensorCostFunction_st"], colors)
    # plot_error_selection(axs[1, 2], rows, "error_sy", "cost_function_type", ["SensorCostFunction_st"], colors)

    fig.legend(loc=9, ncol=4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    axs[1, 0].set_xlabel("Iterations")
    axs[1, 1].set_xlabel("Iterations")
    axs[0, 0].set_ylabel("$L_2$-error in $s_x$")
    axs[1, 0].set_ylabel("$L_2$-error in $s_y$")
    axs[0, 0].set_title("Sensor Cost Function {}".format(J1))
    axs[0, 1].set_title("Sensor Cost Function {}".format(J2))
    plt.savefig('errors_xy_LOH1_test2.png', bbox_inches="tight")
    plt.show()


def plot_J_selection(ax, data, key, values, colors, add_label=False, ylabel="", xlabel="fops"):
    data_selected = data.loc[data[key].isin(values)]
    it = 0
    for i, row in data_selected.iterrows():
        iter_points = list(range(0, len(row['J'])))
        J = row['J']
        if add_label:
            ax.semilogy(iter_points, J, c=colors[it], marker='.',
                        label='{0}, dofs={1}'.format(row['K'], row['dofs']))
        else:
            ax.semilogy(iter_points, J, c=colors[it], marker='.')
        it += 1


def plot_DJ_selection(ax, data, comp, key, values, colors, add_label=False, ylabel="", xlabel="fops"):
    data_selected = data.loc[data[key].isin(values)]
    it = 0
    for i, row in data_selected.iterrows():
        iter_points = list(range(0, len(row['J'])))
        # bounds = '$(s_x, s_y) \in \Omega$' if len(row['bounds']) != 0 else '$(s_x, s_y) \in \mathbb{R}$'
        dj = np.array(row['DJ'])
        if add_label:
            ax.plot(iter_points, dj[:, comp], c=colors[it], marker='.',
                    label='{0}, dofs={1}'.format(row['K'], row['dofs']))
        else:
            ax.plot(iter_points, dj[:, comp], c=colors[it], marker='.')
        it += 1


def plot_J_DJ(result_pd, **kwargs):
    # rows = result_pd.loc[result_pd['M'].isin(m_sizes) & result_pd['K'].isin(RK_orders)
    # & result_pd['D'].isin(dg_orders)]
    # exclude = ~(result_pd['M'].isin([16]) & result_pd['D'].isin([1]) & result_pd['K'].isin(['RK2']))
    # rows = result_pd.loc[(result_pd['final error'] <= 3.0)]

    J1 = "$J_R$"
    J2 = "$J_S$"
    rows = result_pd.sort_values(by=['D', 'K', "cost_function_type"])
    colors = sns.color_palette("hls", 4)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey="all")
    plot_J_selection(axs[0], rows, "cost_function_type", [J1], colors, False)
    plot_J_selection(axs[1], rows, "cost_function_type", [J2], colors, True)
    axs[1].yaxis.get_major_locator().numticks = 4
    fig.legend(loc=9, ncol=4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    axs[0].set_ylabel("Cost Function")
    axs[0].set_xlabel("Iterations")
    axs[1].set_xlabel("Iterations")
    axs[0].set_title("Sensor Cost Function {}".format(J1))
    axs[1].set_title("Sensor Cost Function {}".format(J2))
    plt.savefig('J_LOH1_test2.png', bbox_inches="tight")
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(10, 4), sharey="row")
    plot_DJ_selection(axs[0, 0], rows, 0, "cost_function_type", [J1], colors, False)
    plot_DJ_selection(axs[0, 1], rows, 1, "cost_function_type", [J1], colors, False)
    plot_DJ_selection(axs[1, 0], rows, 0, "cost_function_type", [J2], colors, True)
    plot_DJ_selection(axs[1, 1], rows, 1, "cost_function_type", [J2], colors, False)
    fig.legend(loc=9, ncol=4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    axs[1, 0].set_xlabel("Iterations")
    axs[1, 1].set_xlabel("Iterations")
    axs[0, 0].set_title("Sensor Cost Function {}".format(J1))
    axs[0, 1].set_title("Sensor Cost Function {}".format(J2))
    axs[0, 0].set_ylabel("$DJ s_x$")
    axs[1, 0].set_ylabel("$DJ s_y$")
    plt.savefig('DJ_LOH1_test2.png', bbox_inches="tight")
    plt.show()


from src.equations.elastic.LOH1_newton_solver import plot_ic, plot_contour


def plot_iterations_with_ic(result_pd):
    fig, axs = plt.subplots(1, 1, figsize=(14, 6))
    s_m = result_pd["opt exact"][0]

    c = plot_ic(32, 2, 0, s_m)
    plt.axes(axs)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']

    for i in range(0, result_pd.shape[0]):  #
        row = result_pd.iloc[i]
        ic_values = np.array(row['iter_params'])
        offset_x = -1 if ic_values[0, 0] == 58 else 0.5

        if row["final error"] < 1.0:
            # if row["final error"] < 0.023:
            plt.scatter(ic_values[0, 0], ic_values[0, 1], zorder=2, marker="o", color='tab:orange')
            plt.text(ic_values[0, 0] + offset_x, ic_values[0, 1] + 0.5,
                     "{}".format(np.round(row["final error"], 3)),
                     c="k", zorder=2, bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 1})

            # plt.plot(ic_values[:, 0], ic_values[:, 1], zorder=2, linestyle=":", marker=".", color='w')
        else:
            plt.scatter(ic_values[0, 0], ic_values[0, 1], zorder=2, marker="d", color='tab:pink')
            if ic_values[0, 0] == 58:
                plt.text(ic_values[0, 0] - 1.5, ic_values[0, 1] + 0.5,
                         "{}".format(np.round(row["final error"], 3)),
                         c="k", zorder=2, bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 1})
            else:
                plt.text(ic_values[0, 0] + 0.5, ic_values[0, 1] + 0.5,
                         "{}".format(np.round(row["final error"], 3)),
                         c="k", zorder=2, bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 1})

    non_converged = []
    converged = []
    for i in range(0, result_pd.shape[0]):
        row = result_pd.iloc[i]
        error_ic = row["error"][0]
        error_f = row["error"][-1]
        if error_f < 1.0:
            converged.append([error_ic, error_f])
        else:
            non_converged.append([error_ic, error_f])
    converged = np.array(converged)
    non_converged = np.array(non_converged)

    # axs[0].scatter(converged[:, 0], converged[:, 1], marker="x")
    # axs[1].loglog(non_converged[:, 0], non_converged[:, 1], marker=".")
    plt.scatter(26, 32, zorder=2, marker="x", color='r', s=50)
    plt.ylabel("y")
    plt.xlabel("x")
    cax = fig.add_axes([axs.get_position().x1 + 0.01, axs.get_position().y0, 0.02, axs.get_position().height])
    plt.colorbar(c, cax=cax)

    # plt.text(26, 30, "$(s_x,s_y)$", c="k", zorder=2, bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 1})
    fig.subplots_adjust(right=0.8)
    plt.title("sig11")
    # plt.xlim(-1, 60)
    # plt.ylim(-1, 35)
    # plt.legend()
    plt.savefig("LOH1_all_start.png", dpi=500)
    plt.show()


def convert_json_to_pandas(folder="opt_output/L-BFGS-B"):
    # _direct_solver
    list_files = os.listdir(os.path.join(os.getcwd(), folder))
    json_files = []

    data_pd = pd.DataFrame()
    for file in list_files:
        with open(os.path.join(os.getcwd(), folder, file), 'r') as fp:
            jobj = json.load(fp)
            json_files.append(jobj)
            run_id = file.split('_')[-1].split('.')[0]
            data_pd = pd.concat([data_pd, extract_table_row(jobj, run_id)], axis=1)
    data_pd = data_pd.T
    data_pd = data_pd.sort_values(by=['final error', 'M', 'D', 'K'])
    data_pd = data_pd.reset_index()
    return data_pd


def extract_table_row(result, run_id):
    # settings
    m_size = result['model_settings']['m_size']
    dg_order = result['model_settings']['dg_order']
    rk_order = result['model_settings']['RK_order']
    ftol = result['settings']['options']['ftol']
    if 'bounds' in result.keys():
        bounds = result['bounds']
    else:
        bounds = []

    # table
    total_run_time = result['total_run_time']
    forward_solve_time = result['forward_solve_time']
    Tit = result['store_opt_res']['nfev']
    opt = result['optimal_parameters']
    opt_exact = result['exact_parameters']

    # plot
    #
    term_idx = len(result['J'])
    J = result['J']

    # term = lambda f1, f2, ftol: abs((f1 - f2) / max(f1, f2, 1)) <= ftol
    # for i in range(1, term_idx):
    #    if term(J[i - 1], J[i], 10e-13):
    #        term_idx = i
    #        break

    J = J[0:term_idx]
    cost_function_type = result["cost_function_type"]
    DJ = np.array(result['DJ'][0:term_idx])

    error = np.array(result['Error'][0:term_idx])
    opt_iter_values = np.array(result['m'][0:term_idx])

    error_per_control = [None] * len(opt)

    # for k in range(0, len(opt)):
    diff = np.subtract(opt_iter_values, opt_exact)
    for i in range(0, len(opt)):
        error_per_control[i] = list(map(lambda x: np.linalg.norm(x), diff[:, i]))

    J_type = {"ControllabilityCostFunction": "$J_C$",
              "SensorCostFunction_st": "$J_S$",
              "SensorCostFunction": "$J_K$"}  # cost_function_type
    J_short = "$J_C$"  # J_type[cost_function_type]
    table_row = {"control": "[sx, sy]", 'M': m_size, "D": dg_order, "K": rk_order,
                 "dofs": result['model_settings']['DF'],
                 "bounds": bounds, "t_f": forward_solve_time, "t_c": total_run_time, "ftol": ftol, "Tit": Tit,
                 "final error": error[-1], 'final error_sx': error_per_control[0][-1],
                 'final error_sy': error_per_control[1][-1],
                 "opt param": np.round(opt, 5), "opt exact": opt_exact, 'error': error,
                 'error_sx': error_per_control[0], 'error_sy': error_per_control[1],
                 "cost_function_type": J_short, 'J': J, 'DJ': DJ, 'iter_params': result['m']}

    table_row = pd.DataFrame.from_dict(table_row, orient='index')

    return table_row


def get_table_results(data_pd):
    pd.set_option('display.max_columns', None)
    data_pd = data_pd.sort_values(by=['D', 'K', "cost_function_type"])  # key=lambda x: x.str.len()
    print(data_pd.keys())
    table_data = data_pd[
        ['M', 'D', 'dofs', 'K', "cost_function_type", 'Tit', 'final error', 'opt param']]

    divide = " c|" + "|".join([" c " for i in range(0, len(table_data.keys()) - 2)]) + "|c "
    print(divide)
    table_string = " & ".join(table_data.keys()) + "\\\\" + '\n'
    for i, row in table_data.iterrows():
        row_string = []
        for j, col in row.iteritems():
            if not type(col) == str:
                if j == 't_f' or j == 't_c':
                    col = np.round(col, 2)
                else:
                    col = np.round(col, 5)

            if j == 'bounds':
                col = '$\Omega$' if len(row['bounds']) != 0 else '$\mathbb{R}$'

            row_string.append(str(col).replace(']', ')').replace('[', '('))

        table_string += " & ".join(row_string) + "\\\\" + '\n'

    print(table_string)

    return table_data


if __name__ == '__main__':
    result_pd = convert_json_to_pandas("opt_output/s_ic_variation")  #

    table_data = get_table_results(result_pd)

    table_data = table_data.sort_values(by=['final error'])  # key=lambda x: x.str.len()
    print(table_data.keys())
    print(table_data)

    # plot_error(result_pd)
    # plot_J_DJ(result_pd, J='J', DJ='DJ')
    plot_iterations_with_ic(result_pd)
