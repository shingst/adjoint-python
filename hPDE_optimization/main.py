#!/usr/bin/env python

# import mpi4py.MPI as MPI
from src.equations.advection.test_cases_advection import advection_sinus_test, advection_taylor_test, \
    advection_rk_convergence_test, advection_gradient_test
from src.equations.elastic.test_cases_elastic_wave import elastic_wave_2d_taylor_test
from src.equations.elastic.LOH1_newton_solver import taylor_test_LOH1
from src.optimization.opt_elastic.optimization_LOH1 import optimize_LOH1, optimize_LOH1_lame_params
import pandas as pd
import os
import sys
import numpy as np
import time


def save_to_csv(file_name, results):
    if os.path.isfile(file_name):
        dataframe = pd.read_csv(file_name)
        dataframe = dataframe.append(pd.DataFrame(results, columns=dataframe.columns), ignore_index=True)
    else:
        dataframe = pd.DataFrame(results,
                                 columns=["equation", "m_size", "dg_order", "DF", "RK", "error", "J_type", "J",
                                          "DJ", "conv_rate"])
    print(dataframe)
    dataframe.to_csv(file_name, index=False)


def pandas_to_latex_table(table_data):
    headers = table_data.keys()
    divide = " c|" + "|".join([" c " for i in range(0, len(headers) - 2)]) + "|c "
    print(divide)
    table_string = " & ".join(headers) + "\\\\" + '\n'
    for i, row in enumerate(table_data):
        table_string += " & ".join(list(map(str, np.round(row, 8)))) + "\\\\" + '\n'
    print(table_string)


if __name__ == '__main__':

    print(sys.argv)
    if len(sys.argv) > 1:
        run_scrip = sys.argv[1]  # "optimization_LOH1"
        print(run_scrip)
    else:
        run_scrip = "LOH1_lame"
        m_size = 16
        dg_order = 1
        rk_order = "RK1"

    if run_scrip == "advection_taylor_test":
        file_name = 'advection_{}.csv'.format(time.time())
        results = advection_taylor_test(dims=[2], m_sizes=[8, 16], rk_orders=['RK2'], dg_orders=[1, 2],
                                        only_gradient=False, cost_func_type="JC")
        save_to_csv(file_name, results)

    elif run_scrip == "elastic_wave_taylor_test":
        # elastic
        # m_size = int(sys.argv[2])
        results = elastic_wave_2d_taylor_test(m_sizes=[8], dg_orders=[1], rk_orders=['RK2'], only_gradient=True)
        print(results)
        save_to_csv("elastic_wave_{}.csv".format(time.time()), results)
    elif run_scrip == "LOH1_taylor_test":
        # m_size = int(sys.argv[2])
        results = taylor_test_LOH1(m_sizes=[16], dg_orders=[1], rk_orders=['RK2'],
                                   use_mock_measurments=False, only_gradient=False)
        print(results)
        save_to_csv("LOH1_vs3.csv", results)

    elif run_scrip == "LOH1_ic_tests":
        m_size = 8  # int(sys.argv[2])
        dg_order = 1  # int(sys.argv[3])
        rk_order = "RK1"  # + str(sys.argv[4])
        print(sys.argv[2], sys.argv[3])
        m_s = [int(sys.argv[2]), int(sys.argv[3])]
        optimize_LOH1(m_size=m_size, dg_order=dg_order, rk_order=rk_order, ic_values=m_s)

    elif run_scrip == "LOH1_ControllabilityCostFunction":
        # LOH1
        m_size = 16  # int(sys.argv[2])
        dg_order = 1  # int(sys.argv[3])
        rk_order = "RK1"  # + str(sys.argv[4])
        optimize_LOH1(m_size=m_size, dg_order=dg_order, rk_order=rk_order,
                      cost_function_type="SensorCostFunction")

    elif run_scrip == "LOH1_lame":
        m_size = 16  # int(sys.argv[2])
        dg_order = 1  # int(sys.argv[3])
        rk_order = "RK2"  # + str(sys.argv[4])
        optimize_LOH1_lame_params(m_size=m_size, dg_order=dg_order, rk_order=rk_order,
                      cost_function_type="ControllabilityCostFunction")

    elif run_scrip == "LOH1_SensorCostFunction":
        m_size = int(sys.argv[2])
        dg_order = int(sys.argv[3])
        rk_order = "RK" + str(sys.argv[4])
        optimize_LOH1(m_size=m_size, dg_order=dg_order, rk_order=rk_order, cost_function_type="SensorCostFunction")
