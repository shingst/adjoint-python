from dolfin import *
from dolfin_adjoint import *
import matplotlib.pyplot as plt
import time
import numpy as np
import ufl
from src.equations.utils import get_dt_cfl, get_dt_cfl_terradg
from src.equations.utils import write_to_vtk, set_params_, get_measures, plot_solution
from src.optimization.cost_functions import TimeTrackingCostFunction, RecordControls

parameters["ghost_mode"] = "shared_facet"
# parameters["form_compiler"]["cpp_optimize"] = True
parameters["std_out_all_processes"] = False


class AnalyticalBoundary(UserExpression):
    def __init__(self, beta_x=Constant(1.0), beta_y=None, **kwargs):
        """ Construct the source function """
        super().__init__(self, **kwargs)
        self.t = 0.0
        self.beta_x = beta_x
        self.beta_y = None
        self.dim = 1 if beta_y is None else 2

    def eval(self, value, x):
        """ Evaluate the source function """
        value[0] = sin(2 * pi * (x[0] - self.beta_x * self.t))

    def value_shape(self):
        return ()


class gD_derivative(UserExpression):
    def __init__(self, gD, **kwargs):
        """ Construct the source function derivative """
        super().__init__(**kwargs)
        self.t = 0.0
        self.beta_x = gD.beta_x
        self.gD = gD  # needed to get the matching time instant

    def eval(self, value, x):
        """ Evaluate the source function's derivative """
        t = self.gD.t
        value[0] = -2 * pi * t * cos(2 * pi * (x[0] - self.beta_x * t))

    def value_shape(self):
        return ()


def forward(beta, gD, ic, f, unit_mesh, h, dg_order, time_stepping, tend=1.0,
            annotate=False, objective_func_obj=None, record_controls=None,
            out_file=None, show_plots=False, show_final_plots=False):
    dimension = unit_mesh.geometric_dimension()
    # set_params_(dimension=dimension)  # sets compiler parameters
    print("dimensions", dimension)

    factor = 0.25
    print(float(beta.beta_x))
    delta_t = (1 / (dg_order ** 2 + 1)) * h * factor * (1 / abs(float(beta.beta_x)))
    print("delta_t", delta_t)

    dx, dS, ds = get_measures(unit_mesh)
    V = FunctionSpace(unit_mesh, 'DG', dg_order)
    u = TrialFunction(V)
    v = TestFunction(V)
    face_normal = FacetNormal(unit_mesh)

    def F_operator(beta, u_old, gD, v, f, n):  # beta, u1, gD, v, f, face_normal

        flux = beta * u_old  # physical flux
        F = dot(grad(v), flux) * dx + inner(v, f) * dx

        if beta.ufl_shape == (2,):
            maxvel = conditional(beta[0] > beta[1], beta[0], beta[1])  # get_max_flow_speed(beta_)
        else:
            maxvel = beta[0]

        flux_dS_ = dot(avg(flux), n('+')) + 0.5 * maxvel('+') * jump(u_old)
        flux_ds_ = dot(0.5 * (flux + gD * beta), n) + 0.5 * maxvel * (u_old - gD)

        F -= inner(flux_dS_, jump(v)) * dS
        F -= inner(flux_ds_, v) * ds

        return F

    if time_stepping == 'RK2':
        a = inner(v, u) * dx

        u1 = project(gD, V, annotate=annotate)
        u2 = Function(V, annotate=annotate)

        L1 = F_operator(beta, u1, gD, v, f, face_normal)  # u_old, beta, gD, v, f, n
        L2 = F_operator(beta, u2, gD, v, f, face_normal)

    elif time_stepping == 'RK3':

        a = inner(v, u) * dx
        u1 = project(gD, V, annotate=annotate)
        u2 = Function(V, annotate=annotate)
        u3 = Function(V, annotate=annotate)

        L1 = F_operator(beta, u1, gD, v, f, face_normal)  # tn
        L2 = F_operator(beta, u2, gD, v, f, face_normal)  # 1/4*tn
        L3 = F_operator(beta, u3, gD, v, f, face_normal)  #

    else:
        u1 = project(ic, V, annotate=annotate)
        u1.rename('u1', 'u1')

        const_delta_t = Constant(1 / delta_t, name='delta_t')
        F_dt = inner(const_delta_t * (u - u1), v) * dx
        F = -F_dt + F_operator(beta, u, gD, v, f, face_normal)
        a, L = lhs(F), rhs(F)

    # storage for computed time step
    u_new = Function(V, annotate=annotate)

    # set up time stepping
    steps = int(tend / delta_t)
    delta_t_cpy = 0.0

    if objective_func_obj is not None:
        objective_func_obj.J_evaluate(u1, 0, delta_t)

    if record_controls is not None:
        record_controls.store_control(u1, t=0.0)

    t = 0
    gD.t = 0

    write_to_vtk(t, u1, out_file)

    plot_solution(u1, gD, dimension, t, title="t=" + str(t), show_plots=show_plots)

    for n in range(1, steps + 1):

        if n == steps:  # final step
            delta_t = tend - t

        if time_stepping == 'RK2':

            solve(a == L1, u_new, annotate=annotate)
            u2.assign(u1 + delta_t * u_new, annotate=annotate)

            gD.t = t + delta_t
            solve(a == L2, u_new, annotate=annotate)

            u1.assign(0.5 * u1 + 0.5 * (u2 + delta_t * u_new), annotate=annotate)

        elif time_stepping == 'RK3':

            solve(a == L1, u_new, annotate=annotate)
            u2.assign(u1 + delta_t * u_new, annotate=annotate)

            gD.t = t + delta_t
            solve(a == L2, u_new, annotate=annotate)
            u3.assign(0.75 * u1 + 0.25 * (u2 + delta_t * u_new), annotate=annotate)

            gD.t = t + 0.5 * delta_t
            solve(a == L3, u_new, annotate=annotate)

            u1.assign((1 / 3) * u1 + (2 / 3) * (u3 + delta_t * u_new), annotate=annotate)

        else:
            # const_delta_t = Constant(1 / delta_t, name='delta_t')
            # F_dt = inner(const_delta_t * (u - u1), v) * dx
            # F = -F_dt + L_operator(beta, u, gD, v, f, face_normal)
            # a, L = lhs(F), rhs(F)

            solve(a == L, u_new, annotate=annotate)
            # solve(F == 0, u_new)
            u1.assign(u_new, annotate=annotate)

        t += delta_t
        gD.t = t

        if objective_func_obj is not None:
            objective_func_obj.J_evaluate(u1, t, delta_t)  # evaluate objective function

        if record_controls is not None:
            record_controls.store_control(u1, t)

        write_to_vtk(t, u1, out_file)

        #if n % int(0.1 * steps) == 0:
        #    print("step n=", n, "of steps=", steps)

        if n % 1 == 0:  # only plot if in serial
            plot_solution(u1, gD, dimension, t, title="t=" + str(t), show_plots=show_plots)

    if show_final_plots:
        plot_solution(u1, gD, dimension, t, title="t=" + str(t), show_plots=show_final_plots)

    print("t final", t, "final_delta_t", delta_t)

    return u1, t


def get_max_flow_speed_float(beta):
    print(beta, type(beta))
    if type(beta) == Expression:
        if beta.ufl_shape == (2,):
            return np.max([float(beta.beta_x), float(beta.beta_y)])
        else:
            return float(beta.beta_x)
    else:
        if beta.ufl_shape == (2,):
            return np.max([float(beta[0]), float(beta[1])])
        else:
            return float(beta[0])


def assemble_system_matrix_RK1(a, L, u1, u, v, beta, f, dx):
    M = assemble(a)
    I = assemble(inner(u, v) * dx)
    I.zero()
    I.ident_zeros()

    F_c_eval_ = beta * u1
    L_first_part = dot(grad(v), F_c_eval_) * dx + inner(v, f) * dx
    b = assemble(L_first_part)
    b_vec = np.array(b)

    # b_boundary = assemble(L_boundary(beta, u1, gD, v, face_normal))
    # b_split = b_vec + b_boundary

    # solve(M, u_tmp.vector(), b_split)

    # complete
    # b = assemble(L)
    # b_array = np.array(b)
    # solve(M, u_new.vector(), b)

    # split
    # f_k = interpolate(f, V)
    # F_k = f_k.vector()

    # K_x = assemble(inner(u, v.dx(0)) * dx)
    # K_y = assemble(inner(u, v.dx(1)) * dx)
    # K = beta[0]*K_x + beta[1]*K_y
    # b_split = M*u1.vector() + delta_t*M*F_k + delta_t*K*u1.vector()

    return


if __name__ == '__main__':
    factor = 0.2
    lam = 0.25
    h = 1 / 32
    for dg_order in [1, 2, 3]:
        delta_t = (1 / (dg_order ** 2 + 1)) * h * factor * (1 / abs(float(lam)))
        print("dg order", dg_order, "delta_t", delta_t)
