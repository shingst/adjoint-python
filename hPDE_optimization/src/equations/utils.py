from dolfin import *
from dolfin_adjoint import *
import matplotlib.pyplot as plt
import numpy as np


def unit_hyper_cube(divisions):
    # method from Fenics tutorial book vol 1,
    # 5.1.4 Parameterizing the number of space dimensions
    mesh_classes = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
    d = len(divisions)
    mesh = mesh_classes[d - 1](*divisions)
    return mesh


def set_params_(dimension):
    parameters['form_compiler']["cpp_optimize"] = True
    parameters['form_compiler']["optimize"] = True
    parameters['form_compiler']['representation'] = 'uflacs'

    if dimension == 1:
        parameters["ghost_mode"] = "shared_point"
    elif dimension == 2:
        parameters["ghost_mode"] = "shared_facet"


def get_measures(unit_mesh):
    dx = Measure('dx', domain=unit_mesh)  # area
    dS = Measure('dS', domain=unit_mesh)  # interior boundaries
    ds = Measure('ds', domain=unit_mesh)  # exterior boundaries

    return dx, dS, ds


def iterative_linear_solver(a, L, u, bc):
    # TODO what are the best solver methods + preconditioners for advection system
    problem = LinearVariationalProblem(a, L, u, bc)
    solver = LinearVariationalSolver(problem)

    solver.parameters.linear_solver = 'petsc'
    solver.parameters.preconditioner = 'petsc_amg'

    prm = parameters.krylov_solver
    prm.absolute_tolerance = 1E-10
    prm.relative_tolerance = 1E-6
    prm.maximum_iterations = 1000

    return solver


# Fenics solvers
def linear_solver(a, L, u, bc):
    problem = LinearVariationalProblem(a, L, u, bc)
    solver = LinearVariationalSolver(problem)

    solver.parameters.linear_solver = 'gmres'
    # 'bicgstab', 'cg', 'gmres','minres', 'petsc', 'richardson', 'superlu_dist', 'tfqmr', 'umfpack'
    #  pre-conditioners
    # 'icc', 'ilu', 'petsc_amg', 'sor'
    solver.parameters.preconditioner = 'ilu'
    prm = solver.parameters.krylov_solver  # short form
    prm.absolute_tolerance = 1E-7
    prm.relative_tolerance = 1E-4
    prm.maximum_iterations = 1000

    return solver


def non_linear_solver(F_, q_, bc_, J_, ffc_options_=None):
    problem = NonlinearVariationalProblem(F_, q_, bc_, J=J_, form_compiler_parameters=ffc_options_)
    solver = NonlinearVariationalSolver(problem)
    solverParams = solver.parameters
    solverParams['newton_solver']['absolute_tolerance'] = 1e-2
    solverParams['newton_solver']['relative_tolerance'] = 1E-6
    solverParams['newton_solver']['maximum_iterations'] = 20
    solverParams['newton_solver']['relaxation_parameter'] = 1.0
    solverParams['newton_solver']['linear_solver'] = 'mumps'


def list_solver_params():
    for key in parameters.keys():

        if type(parameters[key]) == Parameters:
            print(key)
            sub_params = parameters[key]
            for key_sub in sub_params.keys():
                print('\t', key_sub, '\t\t\t', sub_params[key_sub])
            print('\n')
        else:
            print(key, '\t\t\t', parameters[key])


def get_dt_cfl(lam, h_max, dg_order, factor=0.5, rk=None):
    if factor is None:
        factor = 0.1 * float(rk[-1]) + 0.2
        print("factor", factor)

    return factor * (1 / (2 * dg_order + 1)) * (h_max / abs(lam))


def get_dt_cfl_terradg(lam, h_max, dg_order, factor=0.5, rk=None):
    if factor is None:
        factor = 0.1 * float(rk[-1]) + 0.2

    return (1 / (dg_order ** 2 + 1)) * (h_max) * factor * (1 / abs(lam))


def write_to_vtk(t, u_old, out_file):
    if out_file is not None:
        out_file << (u_old, t)


def compute_max_error(u_sol, u_exact, mesh, verbose=False):
    vertex_values_u_D = u_exact.compute_vertex_values(mesh)
    vertex_values_u = u_sol.compute_vertex_values(mesh)
    diff = vertex_values_u_D - vertex_values_u

    error_max = np.max(np.abs(diff))
    error_norm = np.linalg.norm(np.abs(diff))

    if verbose:
        print('error_max = ' + str(error_max))
        print('error_l2 = ' + str(error_norm))

    return error_max, error_norm


def plot_solution(phi, analytical_sol, dimension, t, title, show_plots):
    if show_plots:
        if dimension == 2:
            plot(phi, title=title)
            plt.show()
            plot_solution_1d(t=t, dim=dimension, function=phi, analytic=analytical_sol)
            plt.show()
        else:
            plot_solution_1d(t=t, dim=dimension, function=phi, analytic=analytical_sol)
            plt.show()


# TODO use compute_vertex_values
def eval_function_along_x(function, dim, y=0.5, xstart=0, xend=1, param_idx=1, dx=1 / 100):
    # x_vec = np.linspace(xstart + 0.5 * dx, xend - 0.5 * dx, int(1 / dx))
    x_vec = np.linspace(xstart, xend, int(1 / dx))
    z_vec = []

    for x in x_vec:

        if dim == 2:
            z_value = function(x, y)
        else:
            z_value = function(x)

        if np.shape(z_value) != ():
            z_value = z_value[param_idx]

        z_vec.append(z_value)

    return x_vec, z_vec


def eval_function_diagonal(function, xstart=0, xend=sqrt(1 ** 2 + 1 ** 2), param_idx=1, dx=1 / 100):
    # x_vec = np.linspace(xstart + 0.5 * dx, xend - 0.5 * dx, int(1 / dx))
    x_vec = np.linspace(xstart, xend, int(1 / dx))
    z_vec = []

    for x in x_vec:

        z_value = function(x, x)
        if np.shape(z_value) != ():
            z_value = z_value[param_idx]

        z_vec.append(z_value)

    return x_vec, z_vec


def plot_solution_1d(t, dim, function, analytic=None, param_idx=0, show=True, title='', label_='', linestyle='--',
                     color='blue', xstart=0, xend=1, tag=-1):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']
    makers = ['d', '.', '*', '+', "o"]
    if analytic is not None:
        x_vec, z_vec = eval_function_along_x(analytic, dim, param_idx=param_idx, xstart=xstart, xend=xend)
        label_ = 'analytic sol.' if label_ == '' else label_

        if param_idx == 0 or param_idx == 3:
            plt.plot(x_vec, z_vec, label="exact", linestyle='-', color='tab:gray')
        else:
            plt.plot(x_vec, z_vec, linestyle='-', color='tab:gray')

    x_vec, z_vec = eval_function_along_x(function, dim, param_idx=param_idx, xstart=xstart, xend=xend)
    plt.plot(x_vec, z_vec, label=label_, linestyle="--", marker=makers[param_idx],
             color=colors[param_idx], markevery=5)  # color=color

    if show:
        plt.title(title + " t=" + str(np.round(t, 8)))
        plt.legend()
        plt.savefig("elastic_t{}_{}_{}.png".format(t, param_idx, tag))
        plt.close()
        # plt.show()


def plot_diagonal(t, function, analytic=None, param_idx=0, show=True, title='', label_='', color='blue',
                  xstart=0, xend=1):
    if analytic is not None:
        x_vec, z_vec = eval_function_diagonal(analytic, param_idx=param_idx, xstart=xstart, xend=xend)
        label_ = 'analytic sol.' if label_ == '' else label_
        plt.plot(x_vec, z_vec, label=label_, color='r')

    x_vec, z_vec = eval_function_diagonal(function, param_idx=param_idx, xstart=xstart, xend=xend)
    plt.plot(x_vec, z_vec, label=label_, linestyle='--', color=color)

    if show:
        plt.title(title + " t=" + str(np.round(t, 8)))
        plt.legend()
        plt.show()


def plot_multidim(u1, t=0.0):
    if u1.ufl_shape == (2,):
        plot_1dim_elastic(u1, t)
    else:
        plot_2dim_elastic(u1, t)


def plot_2dim_elastic(u1, t=0.0):
    sig11, sig22, sig12, vel_x, vel_y = u1.split()

    c = plot(sig11, title="sig11 at t=" + str(t))
    plt.colorbar(c)
    plt.show()
    c = plot(sig22, title="sig22 at t=" + str(t))
    plt.colorbar(c)
    plt.show()
    c = plot(sig12, title="sig12 at t=" + str(t))
    plt.colorbar(c)
    plt.show()
    c = plot(vel_x, title="vel_x at t=" + str(t))
    plt.colorbar(c)
    plt.show()
    c = plot(vel_y, title="vel_y at t=" + str(t))
    plt.colorbar(c)
    plt.show()


def plot_1dim_elastic(u1, t=0.0, show=True):
    print(u1.ufl_shape)
    sig, v = u1.split()

    plot(sig, title="sig12 at t=" + str(t))
    # plt.show()
    plot(v, title="v at t=" + str(t))
    if show:
        plt.show()


def plot_sig(u1, t=0.0):
    sig11, sig22, sig12 = u1.split()
    c = plot(sig11, title="sig11 at t=" + str(t))
    plt.colorbar(c)
    plt.show()
    c = plot(sig22, title="sig22 at t=" + str(t))
    plt.colorbar(c)
    plt.show()
    c = plot(sig12, title="sig12 at t=" + str(t))
    plt.colorbar(c)
    plt.show()
