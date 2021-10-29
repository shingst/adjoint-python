from dolfin import *
# from fenics import *
from fenics_adjoint import *
import matplotlib.pyplot as plt
import numpy as np
import time
import ufl
from src.equations.elastic.dolfin_dg_elastic_wave import forward_elastic
from src.equations.elastic.elastic_wave_newton import forward_elastic_newton
from src.equations.utils import plot_1dim_elastic, plot_multidim, write_to_vtk, get_measures, plot_solution_1d


# ----------------------------------------------------------------------------------------------------------------------
# 1 dimensional test cases
# ----------------------------------------------------------------------------------------------------------------------
def sol_v(x, t, cs):
    return 0.5 * (np.sin(2 * np.pi * (x + cs * t)) + np.sin(2 * np.pi * (x - cs * t)))


def sol_sig(x, t, cs, rho):
    return 0.5 * rho * cs * (np.sin(2 * np.pi * (x + cs * t)) - np.sin(2 * np.pi * (x - cs * t)))


def test_1dim_elastic_wave():
    m_size = 15
    unit_mesh_1d = UnitIntervalMesh(m_size)
    h = 1 / m_size
    mu = Constant(2)
    control_mu = Control(mu)
    rho = Constant(1)
    # lam = 1
    cs = np.sqrt(float(mu) / float(rho))
    cp = cs

    # Set up Dirichlet BC

    sig = "0.5 * 1 * cs * (sin(2 * pi * (x[0] + cs * t)) - sin(2 * pi * (x[0] - cs * t)))"
    v = "0.5 * (sin(2 * pi * (x[0] + cs * t)) + sin(2 * pi * (x[0] - cs * t)))"

    analytic_solution = Expression((sig, v), cs=cs, t=0, degree=3)  # element=V.ufl_element(),
    source_term = Expression(('0', '0'), cs=cs, t=0, degree=3)  # element=V.ufl_element()

    V = VectorFunctionSpace(unit_mesh_1d, 'DG', degree=3, dim=2)

    # gD, f, unit_mesh, out_file: File, dg_order=3, show_plots=False
    material_params = {'mu': mu, 'rho': rho, 'cs': cs, 'cp': cp}
    sol, tend = forward_elastic(material_params, gD=analytic_solution, f=source_term,
                                unit_mesh=unit_mesh_1d, h=h, show_plots=True)

    plot_1dim_elastic(sol, t=1.0, show=False)
    analytic_solution.t = 1.0
    u1 = project(analytic_solution, V)
    plot_1dim_elastic(u1, t=1.0)

    mu_m = Constant(3)
    rho_m = Constant(1)
    cs_m = ufl.sqrt(float(mu_m) / float(rho_m))
    measured = Expression((sig, v), cs=cs_m, t=1.0, degree=3)

    dx, dS, ds = get_measures(unit_mesh_1d)
    J = assemble((0.5 * (sol - measured) ** 2) * dx)

    dJdmu = compute_gradient(J, [control_mu])
    print("Gradient of dJdmu:", float(dJdmu[0]))

    h = Constant(1)
    J_hat = ReducedFunctional(J, [control_mu])
    conv_rate = taylor_test(J_hat, mu, h)  # check that J_hat is the derivative
    print("convergence rate", conv_rate)


def plot_exact_solution_1dim():
    mu = 2
    rho = 1
    cs = sqrt(mu / rho)
    print(cs)

    x = np.linspace(0, 1, 100)
    fig, axs = plt.subplots(1, 2)

    for j, t in enumerate([0, 0.01]):
        v = sol_v(x, t=t, cs=float(cs))
        sig = sol_sig(x, t, cs=float(cs), rho=rho)

        axs[j].plot(x, v, label='v, t=' + str(t))
        axs[j].plot(x, sig, label='sig, t=' + str(t))
        axs[j].legend()

    plt.show()

    u_exact = Expression(("0.5 * rho * cs * (sin(2 * pi * (x[0] + cs * t)) - sin(2 * pi * (x[0] - cs * t)))",
                          "0.5 * (sin(2 * pi * (x[0] + cs * t)) + sin(2 * pi * (x[0] - cs * t)))"), degree=2, rho=rho,
                         cs=cs, t=0)


def write_vector_function(func_sig, func_vel):
    sig11 = as_vector((1, 0, 0, 0, 0))
    sig22 = as_vector((0, 1, 0, 0, 0))
    sig12 = as_vector((0, 0, 1, 0, 0))

    vel_x = as_vector((0, 0, 0, 1, 0))
    vel_y = as_vector((0, 0, 0, 0, 1))



def set_up_2d_example(m_size):
    rho = Constant(1.2)
    lam = Constant(2.2)
    mu = Constant(1.3)

    cp = Constant(sqrt((lam + 2 * mu) / rho))
    cs = Constant(sqrt(mu / rho))
    # cp = Expression("sqrt((lam + 2 * mu) / rho)", degree=1, lam=lam, rho=rho, mu=mu)
    # cs = Expression("sqrt(mu / rho)", degree=1, lam=lam, rho=rho, mu=mu)

    material_parameters = {'cs': cs, 'cp': cp, 'lam': lam, 'mu': mu, 'rho': rho}

    # sol_displacement = Expression(('cos(k*(x[0] - cp*t))', 'cos(k*(x[0] - cs*t))', '0'),
    #                              cs=cs, cp=cp, k=2 * pi, t=0.0, degree=2)
    # sol_velocity = Expression(('cp*k*sin(k*(x[0] - cp*t))', 'cs*k*sin(k*(x[0] - cs*t))', '0'),
    #                          cs=cs, cp=cp, k=2 * pi, t=0.0, degree=2)

    exact_solution = Expression(('-k*sin(2*pi*(x[0] - cp*t))*(lam + 2*mu)',  # sig11
                                 '-k*sin(2*pi*(x[0] - cp*t))*lam',  # sig22
                                 '-2*0.5*k*sin(k*(x[0] - cs*t))*mu',  # sig12
                                 'cp*k*sin(k*(x[0] - cp*t))',  # u
                                 'cs*k*sin(k*(x[0] - cs*t))'),  # v
                                cs=cs, cp=cp, lam=lam, mu=mu, t=0.0, k=2 * pi, degree=2)

    f = Constant((0, 0, 0, 0, 0))

    return material_parameters, exact_solution, f


# ----------------------------------------------------------------------------------------------------------------------
# 2 dimensional test cases
# ----------------------------------------------------------------------------------------------------------------------
def elastic_wave_2d_taylor_test(control_parameter='lam', h=1.0, m_sizes=[8, 16], dg_orders=[1], rk_orders=['RK2'],
                                only_gradient=False):
    results = []  # pd.DataFrame(header=["m_size", "dg_order", "DF", "RK", "error", "DJ", "conv_rate"])
    errors = []
    DFs = []
    for rk_order in rk_orders:
        for dg_order in dg_orders:
            for m_size in m_sizes:
                material_parameters, exact_solution, f = set_up_2d_example(m_size)
                control = Control(material_parameters[control_parameter])
                unit_mesh = UnitSquareMesh(m_size, m_size)
                exact_solution.t = 0.0
                u_end, t_end = forward_elastic_newton(material_parameters, exact_solution, f, unit_mesh,
                                                      h=1 / m_size, dg_order=dg_order, tend=0.2, timestepping=rk_order,
                                                      show_plots=True, show_final_plots=False)

                DF = np.shape(np.array(u_end.vector()))[0]
                DFs.append(DF)
                exact_solution.t = 0.2
                error_l2 = errornorm(exact_solution, u_end, norm_type='l2', degree_rise=1)
                errors.append(error_l2)
                J = assemble(inner(u_end, u_end) * dx)
                conv_rate = None
                DJ = None

                if not only_gradient: # m_size < 64:
                    DJ = float(compute_gradient(J, [control])[0])

                    if not only_gradient:
                        J_hat = ReducedFunctional(J, [control])
                        h = Constant(h)  # check that J_hat is the derivative
                        conv_rate = taylor_test(J_hat, material_parameters[control_parameter], h)
                        results.append(["elastic2D", m_size, dg_order, DF, float(rk_order[-1]), error_l2, J, DJ, conv_rate])

                results.append(["elastic2D", m_size, dg_order, DF, float(rk_order[-1]),
                                error_l2, "None", J, DJ, conv_rate])

    convergence_rate = [errors[k - 1] / errors[k] for k in range(1, len(m_sizes))]
    orders = [np.log(errors[k - 1] / errors[k]) / np.log((1 / m_sizes[k - 1]) / (1 / m_sizes[k])) for k in
              range(1, len(m_sizes))]

    print(m_sizes[1::], DFs[1::], errors[1::], convergence_rate, orders)
    table = np.stack([m_sizes[1::], DFs[1::], errors[1::], convergence_rate, orders], axis=1)
    print(table)
    to_latex_table(table)

    return results


def to_latex_table(table_data):
    headers = ["M", "DF", "L_2-error", "conv. rate", "conv. order"]
    divide = " c|" + "|".join([" c " for i in range(0, len(headers) - 2)]) + "|c "
    print(divide)
    table_string = " & ".join(headers) + "\\\\" + '\n'
    for i, row in enumerate(table_data):
        table_string += " & ".join(list(map(str, np.round(row, 8)))) + "\\\\" + '\n'
    print(table_string)


if __name__ == '__main__':
    print("main")
    # test_1dim_elastic_wave()

    # elastic_wave_2d_test_example()
    elastic_wave_2d_taylor_test(only_gradient=True)
