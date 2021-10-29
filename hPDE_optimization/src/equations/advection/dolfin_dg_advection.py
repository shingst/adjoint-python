import numpy as np
import time
import matplotlib.pyplot as plt

from dolfin import *
from dolfin_dg import *
from dolfin_adjoint import *

from dolfin_dg.fluxes import ConvectiveFlux, max_abs_of_sequence
from ufl import Max

from src.equations.utils import plot_solution_1d, get_measures, set_params_, get_dt_cfl, write_to_vtk
from src.equations.advection.dg_advection import get_max_flow_speed_float
from src.equations.elastic.dolfin_dg_elastic_wave import HyperbolicOperatorModified, LocalLaxFriedrichsModified


def forward(beta, maxvel, gD, f, unit_mesh, dg_order=3, out_file=None, show_plots=False, vtks=False):
    dimension = unit_mesh.geometric_dimension()
    print(dimension)
    set_params_(dimension=dimension)  # sets compiler parameters
    delta_t = get_dt_cfl(get_max_flow_speed_float(beta), unit_mesh.hmax(), dg_order, factor=0.2)

    dx, dS, ds = get_measures(unit_mesh)
    V = FunctionSpace(unit_mesh, 'DG', dg_order)
    v = TestFunction(V)

    u = TrialFunction(V)
    u_old = project(gD, V)
    u_old.rename('u_old', 'u_old')

    # Convective Operator
    def F_c(U):
        return beta * U

    convective_flux = LocalLaxFriedrichsModified(maxvel)
    ho = HyperbolicOperatorModified(unit_mesh, V, DGDirichletBC(ds, gD), F_c, convective_flux)

    residual = - f * v * dx + dot((1 / delta_t) * (u - u_old), v) * dx
    residual += ho.generate_fem_formulation(u, v)
    print(type(convective_flux.alpha))

    a, L = lhs(residual), rhs(residual)
    u_new = Function(V)

    t = 0
    tend = 1
    steps = int(tend / delta_t)

    write_to_vtk(t, u_old, out_file)

    if show_plots:
        plot(u_old)
        plt.show()
        plot_solution_1d(t=0, dim=dimension, function=u_old, analytic=gD)

    # start time stepping
    for n in range(steps + 1):

        if n == steps:
            delta_t = tend - t

        solve(a == L, u_new)
        u_old.assign(u_new)

        write_to_vtk(t, u_old, out_file)
        t += delta_t
        gD.t = t

        if n % (0.1) * (steps + 1) == 0:
            print("n = ", n, " of steps=", steps + 1)

    if True:
        plot(u_old)
        plt.show()

        plot_solution_1d(t=t, dim=dimension, function=u_old, analytic=gD)

    return u_old, t


def advection_example1():
    unit_mesh_ = UnitSquareMesh(10, 10)  # UnitIntervalMesh(100)

    beta_x = Constant(0.5)
    beta_y = Constant(0.5)
    beta = as_vector((beta_x, beta_y))
    boundary_function = Expression('sin(2*pi*(x[0]- cx*t))', degree=3, cx=beta_x, t=0)
    source_term = Expression('0.0', degree=3)

    control_beta = Control(beta_x)  # dolfin adjoint control point for flow parameter

    u_final, t_final = forward(beta, beta_x, boundary_function, source_term, unit_mesh_, show_plots=True)

    error = errornorm(boundary_function, u_final, norm_type='l2', degree_rise=3)
    print("error l2:", error)

    # define measured velocity field
    beta_measured = Constant(0.1)  # this is normally unknown -> to be optimized
    u_measured = Expression('sin(2*pi*(x[0]- cx*t))', degree=3, cx=beta_measured, t=1)

    # define cost function J (non time dependent)
    dx = Measure('dx', domain=unit_mesh_)
    J = assemble((0.5 * (u_final - u_measured) ** 2) * dx)
    print("cost function J=", J)

    # Currently fails with dolfin_dg as exterior dS formulation is not compatible with dolfin_dg

    u_measured = Expression('sin(2*pi*(x[0]- cx*t))', degree=3, cx=Constant(0.1), t=0)

    dx, dS, ds = get_measures(unit_mesh_)
    J = assemble((0.5 * (u_final - u_measured) ** 2) * dx)

    dJdbeta0 = compute_gradient(J, [control_beta])
    print("Gradient of dJdbeta0:", float(dJdbeta0[0]))

    h = Constant(0.1)
    J_hat = ReducedFunctional(J, [control_beta])
    conv_rate = taylor_test(J_hat, beta_x, h)  # check that J_hat is the derivative
    print("convergence rate", conv_rate)


if __name__ == '__main__':
    advection_example1()
