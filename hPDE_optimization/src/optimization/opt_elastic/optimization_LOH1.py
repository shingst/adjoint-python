#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
from dolfin import *
from dolfin_dg import *
from dolfin_adjoint import *
import time
import json
import os

from src.optimization.cost_functions import TimeTrackingCostFunction, SensorCostFunction, ControllabilityCostFunction, \
    CostFunction
from src.equations.elastic.LOH1_newton_solver import forward_LOH1, LOH1_setup, create_mock_measurments, \
    ic_origin_constants

# eval_cb_list =
derivative_cb_list = {'J': [], 'DJ': [], 'm': []}


def eval_cb_post(j, m):
    # eval_cb.append([j, float(m)])
    print("j", j, "m", type(m))


def derivative_cb_post(j, dj, m):
    # print("j = %f, dj = %f, m = %f." % (j, dj, float(m)))

    derivative_cb_list['J'].append(j)
    derivative_cb_list['DJ'].append(list(map(float, dj)))
    derivative_cb_list['m'].append(list(map(float, m)))


def optimize_LOH1_lame_params(m_size, dg_order, rk_order, cost_function_type="controllability",
                              folder='src/optimization/opt_elastic/opt_output/'):
    print("cost_function_type", "M", m_size, "dg_order", dg_order, "rk_order")
    # bounds = [(1, 1, 1), (10, 15, 30)]
    bounds = [(0.1), (10)]

    # initial guess
    opt_param_str = "[rho]"  # , mu, lam
    sx = Constant(26, name='sx')
    sy = Constant(32, name='sy')

    total_run_time = time.time()
    material_params, source_term, yz_plane, domain_edges = LOH1_setup(m_size, is_square=False)
    initial_solution = ic_origin_constants(sx, sy)
    print(material_params)

    for p in material_params['layer']:
        print(p, float(material_params['layer'][p]))

    # cp = 4.0
    # cs = 2.0
    # rho = 2.6
    # mu = 10.4
    # lam = 20.8
    s_m = [0.5]
    material_params['layer']['rho'] = Constant(s_m[0])
    # material_params['layer']['mu'] = Constant(s_m[1])
    # material_params['layer']['lam'] = Constant(s_m[2])
    # control parameters
    print(material_params['layer']['rho'])
    rho_layer_control = Control(material_params['layer']['rho'])
    # mu_layer_control = Control(material_params['layer']['mu'])
    # lam_layer_control = Control(material_params['layer']['lam'])

    # cost function setup
    if cost_function_type == 'SensorCostFunction':
        cost_function = SensorCostFunction(yz_plane)
    else:
        cost_function = ControllabilityCostFunction(yz_plane)

    cost_function.start_record()
    u_measured = create_mock_measurments(dg_order=dg_order, tend=1, unit_mesh=yz_plane, f=source_term,
                                         timestepping=rk_order, material_params=material_params,
                                         cost_function=cost_function, origin_s=[26, 32])

    cost_function.end_record()
    print("starting initial annotation solve")
    forward_solve_time = time.time()
    u_sol, t_end, tseps = forward_LOH1(material_params=material_params, f=source_term,
                                       unit_mesh=yz_plane, ic=initial_solution, dg_order=dg_order, tend=1,
                                       timestepping=rk_order, cost_function=cost_function, annotate=True,
                                       show_plots=False,
                                       show_final_plots=False)
    forward_solve_time = time.time() - forward_solve_time
    DF = np.array(u_sol.vector()).shape[0]
    num_ops_per_dofs_update = tseps * int(rk_order[-1])  # number of time steps * rk order

    print("end initial annotation solve")
    if cost_function_type == 'sensor_cost_func':
        J = cost_function.J_final_evaluate(u_sol, 1.0)
    else:
        cost_function.phi_m = u_measured
        J = cost_function.J_final_evaluate(u_sol, 1.0)

    print("J", float(J))

    J_hat = ReducedFunctional(J, [rho_layer_control],  # mu_layer_control, lam_layer_control
                              derivative_cb_post=derivative_cb_post)

    conv_rate = taylor_test(J_hat, material_params['layer']['rho'], Constant(1.0))
    print(conv_rate)

    # optimization routine CG
    log_results = {}
    settings = {"method": "L-BFGS-B",
                "options": {"disp": True, "ftol": 1.0e-8, "maxiter": 100}}

    print("starting minimize")

    if len(bounds) > 0:
        opt_list, res = minimize(J_hat, bounds=bounds, method=settings['method'], options=settings['options'])
    else:
        opt_list, res = minimize(J_hat, method=settings['method'], options=settings['options'])

    total_run_time = time.time() - total_run_time

    # store metrics
    log_results['total_run_time'] = total_run_time
    log_results['forward_solve_time'] = forward_solve_time
    log_results['settings'] = settings

    res['jac'] = res['jac'].tolist()
    res['x'] = res['x'].tolist()

    if settings['method'] == 'L-BFGS-B':
        del res['hess_inv']
        if type(res['message']) is not str:
            res['message'] = ""

    log_results['store_opt_res'] = res
    log_results['cost_function_type'] = cost_function_type
    log_results['opt_param_str'] = opt_param_str
    log_results['bounds'] = bounds
    log_results['model_settings'] = {'dg_order': dg_order, 'm_size': m_size, "DF": DF, 'RK_order': rk_order,
                                     "ops": num_ops_per_dofs_update}
    print(opt_list)
    log_results['optimal_parameters'] = float(opt_list)  # list(map(float, opt_list))
    log_results['exact_parameters'] = s_m
    log_results['J'] = derivative_cb_list['J']
    log_results['DJ'] = derivative_cb_list['DJ']
    log_results['m'] = derivative_cb_list['m']
    log_results['Error'] = [np.linalg.norm(np.subtract(derivative_cb_list['m'][i], s_m)) for i in
                            range(0, len(derivative_cb_list['J']))]

    timestamp = time.time()
    rk = {'RK1': '_1', 'RK2': '_2', 'RK3': '_3'}
    file = folder + settings['method']
    bounds_tag = '' if len(bounds) == 0 else '_b_true'
    file += '/optimize_LOH1_m_{0}_dg_{1}_rk{2}{3}_{4}.json'.format(m_size, dg_order,
                                                                   rk[rk_order], bounds_tag,
                                                                   int(timestamp))

    print(log_results)
    with open(file, 'w') as fp:
        json.dump(log_results, fp)

    print("...................................................................................................")
    print("DJ", res['jac'])
    # rho_layer_control, mu_layer_control, lam_layer_control

    print("initial error: ", np.linalg.norm([material_params['layer']['rho'] - s_m[0]]))  # ,
    # np.linalg.norm([parameters['layer']['mu'] - s_m[1]]),
    # np.linalg.norm([parameters['layer']['lam'] - s_m[3]]))

    print("Optimal parameters", list(map(float, opt_list)))
    print("final error: ", np.linalg.norm([float(opt_list[0]) - s_m[0]]))  # ,
    # np.linalg.norm([float(opt_list[1]) - s_m[1]]),
    # np.linalg.norm([float(opt_list[2]) - s_m[2]]))
    print("...................................................................................................")
    print("\n\n\n")


def optimize_LOH1(m_size, dg_order, rk_order, cost_function_type="controllability",
                  folder='src/optimization/opt_elastic/opt_output/', ic_values=[20, 20], s_m=[26, 32]):
    print("cost_function_type", "M", m_size, "dg_order", dg_order, "rk_order")
    bounds = [(0, 0), (58, 33)]

    # initial guess
    opt_param_str = "[sx, sy]"
    sx = Constant(ic_values[0], name='sx')
    sy = Constant(ic_values[1], name='sy')

    total_run_time = time.time()
    material_params, source_term, yz_plane, domain_edges = LOH1_setup(m_size, is_square=False)
    initial_solution = ic_origin_constants(sx, sy)

    # control parameters
    control_sx = Control(sx)
    control_sy = Control(sy)
    # rho_layer_control = Control(material_properties['layer']['rho'])

    # TODO add some gaussian noise!

    # cost function setup
    if cost_function_type == 'SensorCostFunction':
        cost_function = SensorCostFunction(yz_plane)
    elif cost_function_type == 'TimeTrackingCostFunction':
        cost_function = TimeTrackingCostFunction(yz_plane)
    else:
        print("ControllabilityCostFunction")
        cost_function = ControllabilityCostFunction(yz_plane)

    cost_function.start_record()
    u_measured = create_mock_measurments(dg_order=dg_order, tend=1, unit_mesh=yz_plane, f=source_term,
                                         timestepping=rk_order, material_params=material_params,
                                         cost_function=cost_function, origin_s=s_m)

    cost_function.end_record()
    print("starting initial annotation solve")
    forward_solve_time = time.time()
    u_sol, t_end, tseps = forward_LOH1(material_params=material_params, f=source_term,
                                       unit_mesh=yz_plane, ic=initial_solution, dg_order=dg_order, tend=1,
                                       timestepping=rk_order, cost_function=cost_function, annotate=True,
                                       show_plots=False,
                                       show_final_plots=False)
    forward_solve_time = time.time() - forward_solve_time
    DF = np.array(u_sol.vector()).shape[0]
    num_ops_per_dofs_update = tseps * int(rk_order[-1])  # number of time steps * rk order

    print("end initial annotation solve")
    if cost_function_type == 'sensor_cost_func':
        J = cost_function.J_final_evaluate(u_sol, 1.0)
    else:
        cost_function.phi_m = u_measured
        J = cost_function.J_final_evaluate(u_sol, 1.0)

    print("J", float(J))

    J_hat = ReducedFunctional(J, [control_sx, control_sy], derivative_cb_post=derivative_cb_post)

    # optimization routine CG
    log_results = {}
    settings = {"method": "L-BFGS-B",
                "options": {"disp": True, "ftol": 1.0e-8, "maxiter": 100}}

    print("starting minimize")

    if len(bounds) > 0:
        opt_list, res = minimize(J_hat, bounds=bounds, method=settings['method'], options=settings['options'])
    else:
        opt_list, res = minimize(J_hat, method=settings['method'], options=settings['options'])

    total_run_time = time.time() - total_run_time

    # store metrics
    log_results['total_run_time'] = total_run_time
    log_results['forward_solve_time'] = forward_solve_time
    log_results['settings'] = settings

    res['jac'] = res['jac'].tolist()
    res['x'] = res['x'].tolist()

    if settings['method'] == 'L-BFGS-B':
        del res['hess_inv']
        if type(res['message']) is not str:
            res['message'] = ""

    log_results['store_opt_res'] = res
    log_results['cost_function_type'] = cost_function_type
    log_results['opt_param_str'] = opt_param_str
    log_results['bounds'] = bounds
    log_results['model_settings'] = {'dg_order': dg_order, 'm_size': m_size, "DF": DF, 'RK_order': rk_order,
                                     "ops": num_ops_per_dofs_update}
    log_results['optimal_parameters'] = list(map(float, opt_list))
    log_results['exact_parameters'] = s_m
    log_results['J'] = derivative_cb_list['J']
    log_results['DJ'] = derivative_cb_list['DJ']
    log_results['m'] = derivative_cb_list['m']
    log_results['Error'] = [np.linalg.norm(np.subtract(derivative_cb_list['m'][i], s_m)) for i in
                            range(0, len(derivative_cb_list['J']))]

    timestamp = time.time()
    rk = {'RK1': '_1', 'RK2': '_2', 'RK3': '_3'}
    file = folder + settings['method']
    bounds_tag = '' if len(bounds) == 0 else '_b_true'
    file += '/optimize_LOH1_m_{0}_dg_{1}_rk{2}{3}_{4}.json'.format(m_size, dg_order,
                                                                   rk[rk_order], bounds_tag,
                                                                   int(timestamp))

    print(log_results)
    with open(file, 'w') as fp:
        json.dump(log_results, fp)

    print("...................................................................................................")
    print("DJ", res['jac'])
    print("initial error: ", np.linalg.norm([20 - s_m[0]]), np.linalg.norm([20 - s_m[1]]))
    print("Optimal parameters", list(map(float, opt_list)))
    print("final error: ", np.linalg.norm([float(opt_list[0]) - s_m[0]]), np.linalg.norm([float(opt_list[1]) - s_m[1]]))
    print("...................................................................................................")
    print("\n\n\n")


def m_values_1():
    return [[
        16.0,
        10.0
    ],
        [
            58.0,
            33.0
        ],
        [
            27.020311248081587,
            16.03493235013992
        ],
        [
            29.700714888225434,
            25.683792167031307
        ],
        [
            27.90266716590354,
            19.21121835130789
        ],
        [
            24.347715968912112,
            20.793751436266902
        ],
        [
            25.952344238367367,
            20.003408890038223
        ],
        [
            25.99785502523654,
            19.985236783808013
        ],
        [
            25.995813622363936,
            19.985919031591337
        ]

    ]


if __name__ == '__main__':
    print(sys.argv)
    ic_positions = [[0, 0], [58, 0], [0, 33], [58, 33]]
    if len(sys.argv) > 1:
        m_size = sys.argv[1]
        dg_order = sys.argv[2]
        rk_order = "RK" + str(sys.argv[3])

    else:
        m_size = 4
        dg_order = 1
        rk_order = "RK" + str(1)

    optimize_LOH1(m_size=m_size, dg_order=dg_order, rk_order=rk_order, ic_values=[16, 10],
                  cost_function_type="SensorCostFunction", folder='opt_output/')
