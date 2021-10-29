from dolfin import *
from dolfin_dg import *
from dolfin_adjoint import *
import matplotlib.pyplot as plt

from src.equations.advection.dg_advection import forward as forward, AnalyticalBoundary
# from src.equations.advection.dolfin_dg_advection import forward as forward_dg
from src.equations.utils import plot_solution_1d, get_measures, plot_diagonal, eval_function_along_x
from src.optimization.cost_functions import ControllabilityCostFunction, TimeTrackingCostFunction, CostFunction


# Callback function for the optimizer
# Writes intermediate results to a logfile
def eval_cb(j, m):
    """ The callback function keeping a log """

    print("j", j)
    print("m", m)


def set_up_2dim_example(m_size, dg_order, beta_x_val=0.25):
    source_term = Expression('0.0', degree=1)

    beta_x = Constant(beta_x_val)
    beta_y = Constant(0.0)
    ic = Expression('sin(2*pi*(x[0] + x[1]))', degree=2, name="ic")

    print("DIM 2")
    unit_mesh_ = UnitSquareMesh(m_size, m_size)  # UnitIntervalMesh(m_size)  #

    beta = Expression(("beta_x", "beta_y"), degree=1, beta_x=beta_x, beta_y=beta_y, name="beta_vec")  #
    beta.dependencies = [beta_x]
    beta.user_defined_derivatives = {beta_x: Constant((1.0, 0.0))}

    # setup boundary function
    gD = Expression("sin(2*pi*((x[0] - beta_x*t)))", degree=dg_order, t=0, beta_x=beta_x)
    gD.dependencies = [beta_x]
    gD.user_defined_derivatives = {
        beta_x: Expression("-2 * pi * t * cos(2 * pi * (x[0] - beta_x * t))", degree=1, t=0, beta_x=beta_x)}

    return beta_x, beta_y, ic, unit_mesh_, beta, gD, source_term


def opt_():
    m_size = 16
    dg_order = 1
    rk_order = 'RK2'
    beta_x, beta_y, ic, unit_mesh_, beta, gD, source_term = set_up_2dim_example(m_size, dg_order, beta_x_val=0.25)

    control = Control(beta_x)
    u_measured = Expression("sin(2*pi*((x[0] - beta_x*t)))", degree=dg_order, t=1.0, beta_x=Constant(0.1))
    cost_func = ControllabilityCostFunction(unit_mesh_,
                                            u_measured)  # TimeTrackingCostFunction(u_measured, unit_mesh_)  #

    u_final, t_final = forward(beta, gD, ic, source_term, unit_mesh_, h=1 / m_size, annotate=True,
                               dg_order=dg_order, time_stepping=rk_order, objective_func_obj=cost_func,
                               show_plots=False,
                               show_final_plots=True)

    error = errornorm(gD, u_final, norm_type='l2', degree_rise=2)
    print("error l2:", error)

    u_measured.t = 1.0

    plot_solution_1d(t=1.0, dim=2, function=u_final, analytic=gD, show=False)
    plot_solution_1d(t=1.0, dim=2, function=u_measured, color='g', show=True)

    if type(cost_func) == ControllabilityCostFunction:
        J = cost_func.J_final_evaluate(u_final, 1.0)  # t=1.0
    else:
        J = cost_func.J_final_evaluate(u_final, 1.0)

    print("J", J)

    # h = Constant(0.1)
    # J_hat = ReducedFunctional(J, [control])
    # conv_rate = taylor_test(J_hat, beta_x, h)  # check that J_hat is the derivative
    # print("convergence rate", conv_rate)

    dJdbeta0 = compute_gradient(J, [control])
    print("Gradient of DJ_Dbeta:", float(dJdbeta0[0]))

    J_list = []
    DJ_list = []
    m_list = []

    def derivative_cb_post(j, dj, m):
        # derivative_cb.append([j, float(dj), float(m)])
        J_list.append(j)
        DJ_list.append(float(dj[0]))
        m_list.append(float(m[0]))

    J_hat = ReducedFunctional(J, [control], derivative_cb_post=derivative_cb_post)

    # run optimization
    opt_p, opt_res = minimize(J_hat, bounds=[0.01, 1], method="L-BFGS-B", tol=1.0e-8,
                              options={'disp': True, 'gtol': 1.0e-8, 'maxiter': 100})

    print("beta = ", float(opt_p))
    print(J_list)
    print(DJ_list)
    print(m_list)
    print(opt_res['jac'])

    return m_list


if __name__ == '__main__':
    # opt1d_test_case_sinus()
    # m_list = opt_()
    import numpy as np
    m_size = 16
    dg_order = 1
    rk_order = 'RK2'
    m_list = [0.25, 0.01, 0.1052569981074664, 0.09960580991601325, 0.0998979233241443]  # 0.09989791005092624
    # m_list = [0.25, 0.01, 0.1052569981074664, 0.09960580991601325]
    # [0.5, 0.01, 0.2433929183430722, 0.09756819562301405, 0.10737212443579786, 0.09990129847645875,
    # 0.0999015960267823]

    import seaborn

    # colors = seaborn.color_palette('hls', len(m_list) + 1)

    u_measured = Expression("sin(2*pi*((x[0] - beta_x*t)))", degree=dg_order, t=0, beta_x=Constant(0.1))
    u_measured.t = 1.0

    plot_solution_1d(t=1.0, dim=2, function=u_measured, color='k', linestyle=':',
                     label_="measured $\\Phi(\hat{\\beta}=0.1)$",
                     show=False)

    beta_x, beta_y, ic, unit_mesh_, beta, gD, source_term = set_up_2dim_example(m_size, dg_order, 0.25)
    print(float(beta_x))
    u_final, t_final = forward(beta, gD, ic, source_term, unit_mesh_, h=1 / m_size, annotate=False,
                               dg_order=dg_order, time_stepping=rk_order,
                               objective_func_obj=CostFunction(unit_mesh_),
                               show_plots=False, show_final_plots=False)
    tag = label_ = "approx. $\\phi(\\beta^0={})$".format(0.25)
    plot_solution_1d(t=1.0, dim=2, function=u_final, show=False, label_=tag)

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("$\phi(x,y=0.5,t=1.0)$")
    plt.savefig("advection_ic_2dim.png", dpi=300)
    plt.show()

    plot_solution_1d(t=1.0, dim=2, function=u_measured, color='k', linestyle=':', label_="$\hat{\\beta}=0.1$",
                     show=False)
    i = 1
    for m in m_list:
        beta_x, beta_y, ic, unit_mesh_, beta, gD, source_term = set_up_2dim_example(m_size, dg_order, m)
        print(float(beta_x))
        u_final, t_final = forward(beta, gD, ic, source_term, unit_mesh_, h=1 / m_size, annotate=False,
                                   dg_order=dg_order, time_stepping=rk_order,
                                   objective_func_obj=CostFunction(unit_mesh_),
                                   show_plots=False, show_final_plots=False)
        tag = label_ = "$\\beta^{0}={1}$".format(i-1, np.round(m,4))
        plot_solution_1d(t=1.0, dim=2, function=u_final, show=False, label_=tag)
        i += 1

    plt.legend(loc="lower center")
    plt.xlabel("x")
    plt.ylabel("$\phi(x,y=0.5,t=1.0)$")
    plt.savefig("advection_opt_2dim.png", dpi=300)
    plt.show()
