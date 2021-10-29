from dolfin import *
from dolfin_adjoint import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.equations.advection.dg_advection import forward as my_forward, AnalyticalBoundary, gD_derivative
from src.equations.utils import plot_solution_1d, get_measures
from src.optimization.cost_functions import TimeTrackingCostFunction, ControllabilityCostFunction


# ----------------------------------------------------------------------------------------------------------------------
# Test equation
# ----------------------------------------------------------------------------------------------------------------------


# homogeneous
def advection_sinus_test(dim=1):
    source_term = Expression('0.0', degree=3)

    # forward(beta, gD, f, unit_mesh, h, dg_order=3, time_stepping='RK1', objective_func_obj=None,
    #        record_controls=None, out_file=None, show_plots=False, show_final_plots=True)
    beta_x = Constant(0.25)

    ic = Expression('sin(2*pi*(x[0]))', degree=3, name="ic")

    if dim == 1:
        print("DIM 1")
        m_size = 20
        unit_mesh_ = UnitIntervalMesh(m_size)

        beta = Expression(("beta_x",), degree=1, beta_x=beta_x, name="beta_vec")  # as_vector((beta_x,))
        beta.dependencies = [beta_x]
        beta.user_defined_derivatives = {beta_x: Constant((1.0,))}

        # setup boundary function
        gD = Expression("sin(2 * pi * (x[0] - beta_x * t))", degree=1, t=0, beta_x=beta_x)
        gD.dependencies = [beta_x]
        gD.user_defined_derivatives = {
            beta_x: Expression("-2 * pi * t * cos(2 * pi * (x[0] - beta_x * t))", degree=1, t=0, beta_x=beta_x)}

    else:
        print("DIM 2")
        m_size = 20
        unit_mesh_ = UnitSquareMesh(m_size, m_size)
        beta_y = Constant(0.0)
        beta = Expression(("beta_x", "beta_y"), degree=1, beta_x=beta_x, beta_y=beta_y,
                          name="beta_vec")  # as_vector((beta_x,))
        beta.dependencies = [beta_x]
        beta.user_defined_derivatives = {beta_x: Constant((1.0, 0.0))}

        gD = Expression("sin(2 * pi * (x[0] - beta_x * t))", degree=1, t=0, beta_x=beta_x)
        gD.dependencies = [beta_x]
        gD.user_defined_derivatives = {beta_x: gD_derivative(gD)}

    info(parameters, True)
    control = Control(beta_x)
    u_final, t_final = my_forward(beta, gD, ic, source_term, unit_mesh_, h=1 / m_size, annotate=True,
                                  dg_order=2, time_stepping='RK2', out_file=None, show_plots=False)

    if dim == 2:
        plot(u_final)
        plt.show()
        plot_solution_1d(t=t_final, dim=2, function=u_final, analytic=gD)

    error = errornorm(gD, u_final, norm_type='l2', degree_rise=3)
    print("error l2:", error)

    # u_measured = Expression('sin(2*pi*(x[0]- cx*t))', degree=3, cx=Constant(0.1), t=0)

    # dx, dS, ds = get_measures(unit_mesh_)
    # J = assemble((0.5 * (u_final - u_measured) ** 2) * dx)

    # dJdbeta0 = compute_gradient(J, [control])
    # print("Gradient of dJdbeta0:", float(dJdbeta0[0]))

    # h = Constant(0.1)
    # J_hat = ReducedFunctional(J, [control])
    # conv_rate = taylor_test(J_hat, beta_x, h)  # check that J_hat is the derivative
    # print("convergence rate", conv_rate)


def set_up_example(m_size, dim=1):
    source_term = Expression('0.0', degree=1)
    ic = Expression('sin(2*pi*(x[0]))', degree=2, name="ic")
    beta_x = Constant(0.25, name="beta_x")

    if dim == 1:
        unit_mesh_ = UnitIntervalMesh(m_size)
        beta = Expression(("beta_x",), degree=1, beta_x=beta_x, name="beta_vec")  # as_vector((beta_x,))
        beta.dependencies = [beta_x]
        beta.user_defined_derivatives = {beta_x: Constant((1.0,))}

        # setup boundary function
        gD = Expression("sin(2 * pi * (x[0] - beta_x * t))", degree=1, t=0, beta_x=beta_x)
        gD.dependencies = [beta_x]
        gD.user_defined_derivatives = {beta_x: Expression("-2 * pi * t * cos(2 * pi * (x[0] - beta_x * t))",
                                                          degree=1, t=0, beta_x=beta_x)}
    else:
        print("DIM 2")
        unit_mesh_ = UnitSquareMesh(m_size, m_size)
        beta_y = Constant(0.0)
        beta = Expression(("beta_x", "beta_y"), degree=1, beta_x=beta_x, beta_y=beta_y, name="beta_vec")
        beta.dependencies = [beta_x]
        beta.user_defined_derivatives = {beta_x: Constant((1.0, 0.0))}
        gD = Expression("sin(2 * pi * (x[0] - beta_x * t))", degree=1, t=0, beta_x=beta_x)
        gD.dependencies = [beta_x]
        gD.user_defined_derivatives = {beta_x: gD_derivative(gD)}

    return unit_mesh_, source_term, ic, beta, beta_x, gD


def advection_taylor_test(dims=[2], m_sizes=[32], dg_orders=[1], rk_orders=['RK1'],
                          cost_func_type="ControllabilityCostFunction", only_gradient=False):
    results = []  # pd.DataFrame(header=["equation","m_size", "dg_order", "DF", "RK", "error", "DJ", "conv_rate"])
    eq_names = {1: "advection-1D", 2: "advection-2D"}
    parameters["lu_solver"]["verbose"] = False

    for dim in dims:
        for rk_ord in rk_orders:
            for dg_ord in dg_orders:
                for m_size in m_sizes:

                    print("dim", dim, "rk_ord", rk_ord, "dg_ord", dg_ord, "m_size", m_size, "cost_func_type",
                          cost_func_type)

                    unit_mesh_, source_term, ic, beta, beta_x, gD = set_up_example(m_size, dim=dim)
                    control = Control(beta_x)

                    u_measured = Expression('sin(2*pi*(x[0]- cx*t))', degree=2, cx=Constant(0.1), t=1.0)

                    if cost_func_type == "JC":
                        cost_function = ControllabilityCostFunction(unit_mesh=unit_mesh_, phi_m=u_measured)
                    elif cost_func_type == "JT":
                        cost_function = TimeTrackingCostFunction(unit_mesh=unit_mesh_, u_target=u_measured)

                    u_final, t_final = my_forward(beta, gD, ic, source_term, unit_mesh_, h=1 / m_size, annotate=True,
                                                  objective_func_obj=cost_function, dg_order=dg_ord,
                                                  time_stepping=rk_ord, show_plots=False, show_final_plots=False)

                    error = errornorm(gD, u_final, norm_type='l2', degree_rise=2)
                    DF = np.shape(np.array(u_final.vector()))[0]

                    J = cost_function.J_final_evaluate(u_final, t=1.0)
                    conv_rate = None
                    DJ = float(compute_gradient(J, [control])[0])

                    if not only_gradient:
                        h = Constant(0.1)
                        J_hat = ReducedFunctional(J, [control])
                        conv_rate = taylor_test(J_hat, beta_x, h)  # check that J_hat is the derivative

                    results.append([eq_names[dim], m_size, dg_ord, DF, int(rk_ord[-1]),
                                    error, cost_function.get_name(), J, DJ, conv_rate])
    print(results)
    return results


def advection_rk_convergence_test():
    rk_ord = 'RK3'
    dg_ord = 1

    m_sizes = [16, 32, 64, 128, 2 * 128, 4 * 128]  #
    deg_free = []
    errors = len(m_sizes) * [None]
    for m, m_size in enumerate(m_sizes):
        unit_mesh_, source_term, ic, beta, beta_x, gD = set_up_example(m_size, dim=1)

        u_final, t_final = my_forward(beta, gD, ic, source_term, unit_mesh_, h=1 / m_size, dg_order=dg_ord,
                                      time_stepping=rk_ord, tend=1)
        deg_free.append(np.array(u_final.vector()).shape[0])
        errors[m] = errornorm(gD, u_final, norm_type='l2', degree_rise=2)

    convergence_rate = [errors[k - 1] / errors[k] for k in range(1, len(m_sizes))]

    orders = [np.log(errors[k - 1] / errors[k]) / np.log((1 / m_sizes[k - 1]) / (1 / m_sizes[k])) for k in
              range(1, len(m_sizes))]

    print(m_sizes[1::], deg_free[1::], errors[1::], convergence_rate, orders)
    table = np.stack([m_sizes[1::], deg_free[1::], errors[1::], convergence_rate, orders], axis=1)
    print(table)
    to_latex_table(table)


def to_latex_table(table_data):
    headers = ["M", "DF", "L_2-error", "conv. rate", "conv. order"]
    divide = " c|" + "|".join([" c " for i in range(0, len(headers) - 2)]) + "|c "
    print(divide)
    table_string = " & ".join(headers) + "\\\\" + '\n'
    for i, row in enumerate(table_data):
        table_string += " & ".join(list(map(str, np.round(row, 8)))) + "\\\\" + '\n'
    print(table_string)


def advection_gradient_test(dim=1):
    dg_ord = 1
    DJ = {'RK1': [], 'RK2': []}
    rk_ord = 'RK1'
    degrees_freedom = []
    m_sizes = [16]
    for m_size in m_sizes:
        unit_mesh_, source_term, ic, beta, beta_x, gD = set_up_example(m_size, dim=dim)
        control = Control(beta_x)
        u_final, t_final = my_forward(beta, gD, ic, source_term, unit_mesh_, h=1 / m_size, annotate=True,
                                      dg_order=dg_ord, time_stepping=rk_ord, show_plots=False, show_final_plots=True)

        error = errornorm(gD, u_final, norm_type='l2', degree_rise=2)
        print("error l2:", error)
        u_measured = Expression('sin(2*pi*(x[0]- cx*t))', degree=2, cx=Constant(0.1), t=0)
        degrees_freedom.append(np.array(u_final.vector()).shape[0])

        J = assemble((0.5 * (u_final - u_measured) ** 2) * dx)

        dJdbeta0 = compute_gradient(J, [control])

        print("Gradient of dJdbeta0:", float(dJdbeta0[0]))
        DJ[rk_ord].append(float(dJdbeta0[0]))

    print(degrees_freedom)
    print(np.round(DJ['RK1'], 8), np.round(DJ['RK2'], 8))


# ----------------------------------------------------------------------------------------------------------------------
# other test
# ----------------------------------------------------------------------------------------------------------------------

# TODO keep or discard?
def test_approx_J(unit_mesh_, m_size, u_final, u_measured):
    # Test approximation of cost function
    x_vec = unit_mesh_.coordinates().flatten()
    difference = u_final.compute_vertex_values(unit_mesh_) - u_measured.compute_vertex_values(unit_mesh_)
    u_final_vec = u_final.compute_vertex_values(unit_mesh_)
    u_measured_vec = u_measured.compute_vertex_values(unit_mesh_)

    print("SUM APPROX J_eval", 0.5 * (1 / m_size) * np.sum((u_final_vec - u_measured_vec) ** 2))


if __name__ == '__main__':
    # Advection

    test_mode = 2

    if test_mode == 1:  # forward example

        advection_sinus_test(dim=1)
        # advection_sinus_test(dim=2)

    elif test_mode == 2:  # Taylor tests

        advection_taylor_test(m_sizes=[8, 16, 32], dg_orders=[1], cost_func_type="JT")

    elif test_mode == 3:  # Runge-Kutta convergence

        advection_rk_convergence_test()

    elif test_mode == 4:  # print Gradient

        advection_gradient_test(dim=1)
        # advection_gradient_test(dim=2)
