import numpy as np
import matplotlib.pyplot as plt
from dolfin import *
from dolfin_dg import *
from dolfin_adjoint import *
import time
import ufl
import os
import json

from dolfin_dg.fluxes import ConvectiveFlux
from dolfin_dg.operators import DGFemFormulation

from src.equations.utils import get_measures, get_dt_cfl, get_dt_cfl_terradg
from src.optimization.cost_functions import SensorCostFunction, VelSensorCostFunction, ControllabilityCostFunction, \
    CostFunction


class LocalLaxFriedrichsModified(ConvectiveFlux):
    """Implementation of symbolic representation of the local-Lax Friedrichs
    flux function
    """

    def __init__(self, material_params):
        divide = material_params['divide']
        cp_layer, cp_hs = material_params['layer']['cp'], material_params['hs']['cp']
        cs_layer, cs_hs = material_params['layer']['cs'], material_params['hs']['cs']

        layer_cond = "( cp_layer > cs_layer ? cp_layer : cs_layer )"
        hs_cond = "(cp_hs > cs_hs ? cp_hs : cs_hs)"
        flow_speed_condition = "x[1] >= divide ? {0} : {1} ".format(layer_cond, hs_cond)

        self.maxvel = Expression(flow_speed_condition,
                                 cp_layer=cp_layer, cs_layer=cs_layer,
                                 cp_hs=cp_hs, cs_hs=cs_hs, divide=divide,
                                 degree=1, name='local_maxvel')

        self.maxvel.dependencies = [cp_layer, cs_layer, cp_hs, cs_hs, divide]

        diff1 = Expression("x[1] >= divide ? ( cp_layer > cs_layer ? 1 : 0 )  : 0", degree=2, divide=divide,
                           cp_layer=cp_layer, cs_layer=cs_layer)
        diff2 = Expression("x[1] >= divide ? ( cp_layer > cs_layer ? 0 : 1 ) : 0", degree=2, divide=divide,
                           cp_layer=cp_layer, cs_layer=cs_layer)
        diff3 = Expression("x[1] >= divide ? 0 : (cp_hs > cs_hs ? 1 : 0)", degree=2, divide=divide, cp_hs=cp_hs,
                           cs_hs=cs_hs)
        diff4 = Expression("x[1] >= divide ? 0 : (cp_hs > cs_hs ? 0 : 1)", degree=2, divide=divide, cp_hs=cp_hs,
                           cs_hs=cs_hs)

        self.maxvel.user_defined_derivatives = {cp_layer: diff1, cs_layer: diff2,
                                                cp_hs: diff3, cs_hs: diff4}

    def setup(self, F_c, u_p, u_m, n):
        pass

    def interior(self, F_c, u, u_p, u_m, n):
        return dot(avg(F_c(u)), n) + 0.5 * self.maxvel('+') * jump(u)

    def exterior(self, F_c, u_p, u_m, n):  # self.alpha
        return 0.5 * (dot(F_c(u_p), n) + dot(F_c(u_m), n) + self.maxvel * (u_p - u_m))


class HyperbolicOperatorModified(DGFemFormulation):
    r"""Base class for the automatic generation of a DG formulation for
    the underlying hyperbolic (1st order) operator of the form

    .. math:: \nabla \cdot \mathcal{F}^c(u)
    """

    def __init__(self, mesh, V, bcs, F_c, H):
        """
        Parameters
        ----------
        mesh
            Problem mesh
        fspace
            Problem function space in which the solution is formulated and
            sought
        bcs
            List of :class:`dolfin_dg.operators.DGBC` to be weakly imposed and
            included in the formulation
        F_c
            One argument function ``F_c(u)`` corresponding to the
            convective flux term
        H
            An instance of a :class:`dolfin_dg.fluxes.ConvectiveFlux`
            describing the convective flux scheme to employ
        """
        DGFemFormulation.__init__(self, mesh, V, bcs)
        self.F_c = F_c
        self.H = H

    def generate_fem_formulation(self, u, v, dx=None, dS=None):
        """Automatically generate the DG FEM formulation

        Parameters
        ----------
        u
            Solution variable
        v
            Test function
        dx
            Volume integration measure
        dS
            Interior facet integration measure

        Returns
        -------
        The UFL representation of the DG FEM formulation
        """

        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        n = ufl.FacetNormal(self.mesh.ufl_domain())

        F_c_eval = self.F_c(u)
        if len(F_c_eval.ufl_shape) == 0:
            F_c_eval = as_vector((F_c_eval,))
        residual = -inner(F_c_eval, grad(v)) * dx

        self.H.setup(self.F_c, u('+'), u('-'), n('+'))
        residual += inner(self.H.interior(self.F_c, u, u('+'), u('-'), n('+')), jump(v)) * dS

        for bc in self.dirichlet_bcs:
            gD = bc.get_function()
            dSD = bc.get_boundary()

            self.H.setup(self.F_c, u, gD, n)
            residual += inner(self.H.exterior(self.F_c, u, gD, n), v) * dSD

        for bc in self.neumann_bcs:
            dSN = bc.get_boundary()

            residual += inner(dot(self.F_c(u), n), v) * dSN

        return residual


class ElasticOperatorLOH1(HyperbolicOperatorModified):

    def global_maxvel_float(self, material_params):
        # domain_divide = material_params['layer']['divide']

        cp_layer, cp_hs = material_params['layer']['cp'], material_params['hs']['cp']
        cs_layer, cs_hs = material_params['layer']['cs'], material_params['hs']['cs']

        return np.max([float(cp_layer), float(cp_hs), float(cs_layer), float(cs_hs)])

    def F_c(self, U):
        sig11, sig22, sig12, u, v = U[0], U[1], U[2], U[3], U[4]

        return as_matrix([[-(self.lam + 2 * self.mu) * u, -self.lam * v],
                          [-self.lam * u, -(self.lam + 2 * self.mu) * v],
                          [-self.mu * v, -self.mu * u],
                          [-(1 / self.rho) * sig11, -(1 / self.rho) * sig12],
                          [-(1 / self.rho) * sig12, -(1 / self.rho) * sig22]])

    def __init__(self, mesh_, V, bcs, material_params):
        """
        Parameters
        ----------
        mesh_
            Problem mesh
        V
            Problem function space in which the solution is formulated and
            sought
        bcs
            List of :class:`dolfin_dg.operators.DGDC` to be weakly imposed
            and included in the formulation

        material_params

        """
        self.material_params = material_params
        self.domain_mesh = mesh_
        self.lam, self.mu, self.rho = create_parameter_expressions(material_params)

        HyperbolicOperatorModified.__init__(self, mesh_, V, bcs, self.F_c, LocalLaxFriedrichsModified(material_params))


def create_parameter_expressions(material_params):
    divide = material_params['divide']

    lam_layer, lam_hs = material_params['layer']['lam'], material_params['hs']['lam']
    mu_layer, mu_hs = material_params['layer']['mu'], material_params['hs']['mu']
    rho_layer, rho_hs = material_params['layer']['rho'], material_params['hs']['rho']

    rho = Expression("x[1] >= divide ? rho_layer : rho_hs", degree=1, name='rho_exp',
                     divide=divide, rho_layer=rho_layer, rho_hs=rho_hs)

    # add dependencies and derivatives
    rho.dependencies = [rho_layer, rho_hs]
    rho.user_defined_derivatives = {rho_layer: Expression("x[1] >= divide ? 1.0 : 0.0", divide=divide, degree=1),
                                    rho_hs: Expression("x[1] >= divide ? 0.0 : 1.0", divide=divide, degree=1)}

    mu = Expression("x[1] >= divide ?  mu_layer : mu_hs", degree=1, name='mu_exp',
                    divide=divide, mu_layer=mu_layer, mu_hs=mu_hs)

    mu.dependencies = [mu_layer, mu_hs]
    mu.user_defined_derivatives = {mu_layer: Expression("x[1] >= divide ? 1.0 : 0.0", divide=divide, degree=1),
                                   mu_hs: Expression("x[1] >= divide ? 0.0 : 1.0", divide=divide, degree=1)}

    lam = Expression(" x[1] >= divide ? lam_layer : lam_hs", degree=1, name='lam_exp',
                     divide=divide, lam_layer=lam_layer, lam_hs=lam_hs)

    lam.dependencies = [lam_layer, lam_hs]
    lam.user_defined_derivatives = {lam_layer: Expression("x[1] >= divide ? 1.0 : 0.0", divide=divide, degree=1),
                                    lam_hs: Expression("x[1] >= divide ? 0.0 : 1.0", divide=divide, degree=1)}

    return lam, mu, rho


def plot_2dim_elastic(u1, t=0.0, k=0):
    adjust_r = 0.9
    fig_size = (6, 4)

    fig, axs = plt.subplots(1, 1, figsize=fig_size)
    c = plot(u1.sub(0, deepcopy=True))  #
    cax = fig.add_axes([axs.get_position().x1 + 0.01, axs.get_position().y0, 0.02, axs.get_position().height])
    plt.colorbar(c, cax=cax)
    plt.axes(axs)
    fig.subplots_adjust(right=adjust_r)
    plt.title("sig11")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("sig11_" + str(k) + str(".png"))
    plt.show()

    fig, axs = plt.subplots(1, 1, figsize=fig_size)
    c = plot(u1.sub(1, deepcopy=True))  #
    cax = fig.add_axes([axs.get_position().x1 + 0.01, axs.get_position().y0, 0.02, axs.get_position().height])
    plt.colorbar(c, cax=cax)
    plt.axes(axs)
    fig.subplots_adjust(right=adjust_r)
    plt.title("sig22")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("sig22_" + str(k) + str(".png"))
    plt.show()

    fig, axs = plt.subplots(1, 1, figsize=fig_size)
    c = plot(u1.sub(2, deepcopy=True))  #
    cax = fig.add_axes([axs.get_position().x1 + 0.01, axs.get_position().y0, 0.02, axs.get_position().height])
    plt.colorbar(c, cax=cax)
    plt.axes(axs)
    fig.subplots_adjust(right=adjust_r)
    plt.title("sig12")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("sig12_" + str(k) + str(".png"))
    plt.show()

    fig, axs = plt.subplots(1, 1, figsize=fig_size)
    c = plot(u1.sub(3, deepcopy=True))  #
    cax = fig.add_axes([axs.get_position().x1 + 0.01, axs.get_position().y0, 0.02, axs.get_position().height])
    plt.colorbar(c, cax=cax)
    plt.axes(axs)
    fig.subplots_adjust(right=adjust_r)
    plt.xlabel("x")
    plt.title("u (x-velocity)")
    plt.ylabel("y")
    plt.savefig("u_" + str(k) + str(".png"))
    plt.show()

    fig, axs = plt.subplots(1, 1, figsize=fig_size)
    c = plot(u1.sub(4, deepcopy=True))  #
    cax = fig.add_axes([axs.get_position().x1 + 0.01, axs.get_position().y0, 0.02, axs.get_position().height])
    plt.colorbar(c, cax=cax)
    plt.axes(axs)
    fig.subplots_adjust(right=adjust_r)
    plt.xlabel("x")
    plt.title("v (y-velocity)")
    plt.ylabel("y")
    plt.savefig("v_" + str(k) + str(".png"))
    plt.show()


def forward_LOH1(material_params, f, unit_mesh, ic, dg_order, tend, timestepping, cost_function,
                 annotate=False, out_file=None, show_plots=False, show_final_plots=False):
    parameters["ghost_mode"] = "shared_facet"
    dx, dS, ds = get_measures(unit_mesh)
    V = VectorFunctionSpace(unit_mesh, 'DG', dg_order, dim=5)

    u1 = project(ic, V, annotate=annotate)
    u1.rename('u1', 'u1')

    # traction free boundary
    gD = FreeSurface(unit_mesh, u1)
    gD.rename('gD', 'gD')

    # Set up boundary condition
    elasticOp = ElasticOperatorLOH1(unit_mesh, V, DGDirichletBC(ds, gD), material_params)

    # determine step size
    maxvel = elasticOp.global_maxvel_float(material_params)
    delta_t = float(get_dt_cfl(maxvel, unit_mesh.hmax(), dg_order, factor=0.5))
    steps = int(tend / delta_t)
    dt = Constant(delta_t, name='dt')
    print("delta_t", delta_t, "maxvel", maxvel, "h", unit_mesh.hmax(), "factor", 0.5)
    u, v = Function(V, name='u', annotate=annotate), TestFunction(V)

    F = - inner(f, v) * dx
    F += dot((1 / dt) * (u - u1), v) * dx  # Euler time stepping
    F += elasticOp.generate_fem_formulation(u, v)

    du = TrialFunction(V)
    J = derivative(F, u, du)

    if timestepping == 'RK2':
        u2 = Function(V, name='u2', annotate=annotate)
        F2 = - inner(f, v) * dx
        F2 += dot((1 / dt) * (u - u2), v) * dx  # Euler time stepping
        F2 += elasticOp.generate_fem_formulation(u, v)

    t = 0
    cost_function.J_evaluate(u1, t, delta_t)

    new_displacement = Function(V)
    if out_file is not None:
        file_sig11 = File("sig11.pvd")
        file_sig11 << (u1.split()[0])

    # hdf5file = HDF5File(MPI.comm_world, "LOH1.hdf5", 'w')
    # hdf5file.write(u1, "u")

    for n in range(0, steps):
        if show_plots:
            plot_2dim_elastic(u1, t, n)
            # plot_2dim_elastic(new_displacement, t)

        if n == steps - 1:
            delta_t = tend - t

        if timestepping == 'RK1':
            solve(F == 0, u, [], J=J, annotate=annotate)
            u1.assign(u, annotate=annotate)

        elif timestepping == 'RK2':
            solve(F == 0, u, [], J=J, annotate=annotate)
            u2.assign(u, annotate=annotate)
            gD.u = u2
            solve(F2 == 0, u, [], J=J, annotate=annotate)
            u1.assign(0.5 * (u1 + u), annotate=annotate)

        gD.u = u1
        t += delta_t
        if out_file is not None:
            file_sig11 << (u1.split()[0])

        # check_points.append(u1)

        # Euler time integration for displacement
        # for i in range(1, len(check_points)):
        #    new_displacement.vector()[:] += 0.5 * delta_t * (check_points[i - 1].vector() + check_points[i].vector())

        cost_function.J_evaluate(u1, t, delta_t)

    if show_final_plots:
        print("final plots")
        plot_2dim_elastic(u1, t, steps)

    print(t)

    return u1, t, steps


class FreeSurface(UserExpression):

    def __init__(self, unit_mesh, phi):
        UserExpression.__init__(self)
        self.unit_mesh = unit_mesh
        self.phi = phi

    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.unit_mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        values_phi1 = np.zeros(5)
        self.phi.eval_cell(values_phi1, x, ufc_cell)

        values[:] = values_phi1[:]

        normalidx = 0 if abs(n[0]) == 1 else 1  # x or y direction
        nsign = -1 if (n[normalidx] < 0) else 1

        values[normalidx] = nsign * values_phi1[normalidx]  # sig11
        values[2] = nsign * values_phi1[2]  # sig12

    def value_shape(self):
        return (5,)


def create_mock_measurments(dg_order, tend, unit_mesh, f, material_params, timestepping, cost_function, origin_s):
    ic = ic_no_dependencies(origin_s[0], origin_s[1])

    u_measured, t_end, tsteps = forward_LOH1(material_params=material_params, f=f, unit_mesh=unit_mesh,
                                             ic=ic, dg_order=dg_order, tend=tend, timestepping=timestepping,
                                             cost_function=cost_function, annotate=False, show_plots=False,
                                             show_final_plots=True)

    return u_measured


def ic_no_dependencies(sx=26, sy=32, degree=1):
    gaussian = "exp(-0.01*pow(x[0] - {0},2) - 0.01*pow(x[1] - {1},2) )".format(sx, sy)
    initial_solution = Expression((gaussian + "*(58-x[0])*(x[0])*(34 - x[1])*(x[1])",
                                   gaussian + "*(58-x[0])*(x[0])*(34 - x[1])*(x[1])", '0', '0', '0'),
                                  degree=degree, name="ic")

    return initial_solution


def ic_origin_constants(sx, sy, degree=1):
    gaussian = "exp(-0.01*pow(x[0] - sx,2) - 0.01*pow(x[1] - sy,2) )"
    initial_solution = Expression((gaussian + "*(58-x[0])*(x[0])*(34 - x[1])*(x[1])",
                                   gaussian + "*(58-x[0])*(x[0])*(34 - x[1])*(x[1])", '0', '0', '0'),
                                  degree=degree, sx=sx, sy=sy, name="initial_solution")
    initial_solution.dependencies = [sx, sy]

    diff_gaussian = "exp(-0.01*pow(sx - x[0],2) - 0.01*pow(sy - x[1],2))"

    diff_sx = Expression(("0.01*2*(x[0] - sx)*" + diff_gaussian + "*(58-x[0])*(x[0])*(34 - x[1])*(x[1])",
                          "0.01*2*(x[0] - sx)*" + diff_gaussian + "*(58-x[0])*(x[0])*(34 - x[1])*(x[1])", '0', '0',
                          '0'),
                         degree=degree, sx=sx, sy=sy, name='diff_sx')

    diff_sy = Expression(("0.01*2*(x[1] - sy)*" + diff_gaussian + "*(58-x[0])*(x[0])*(34 - x[1])*(x[1])",
                          "0.01*2*(x[1] - sy)*" + diff_gaussian + "*(58-x[0])*(x[0])*(34 - x[1])*(x[1])", '0', '0',
                          '0'),
                         degree=degree, sx=sx, sy=sy, name='diff_sy')

    initial_solution.user_defined_derivatives = {sx: diff_sx, sy: diff_sy}

    return initial_solution


def taylor_test_LOH1(m_sizes=[32], dg_orders=[1], rk_orders=['RK1'],
                     use_mock_measurments=False, only_gradient=False, s_m_ic=[20, 20]):
    eq_name = "LOH1"
    results = []
    errors = []
    deg_free = []

    print(rk_orders)
    for rk_ord in rk_orders:
        for dg_ord in dg_orders:

            material_params, source_term, yz_plane, domain_eges = LOH1_setup(m_sizes[-1] * 2, is_square=False)
            sx = Constant(s_m_ic[0], name='sx')  # initial disturbance
            sy = Constant(s_m_ic[1], name='sy')
            initial_solution = ic_origin_constants(sx, sy)

            u_fine, t_end, tstep = forward_LOH1(material_params=material_params, f=source_term,
                                                unit_mesh=yz_plane, ic=initial_solution, dg_order=dg_ord, tend=1,
                                                timestepping=rk_ord,
                                                cost_function=CostFunction(yz_plane),
                                                annotate=False, show_plots=False, show_final_plots=False)
            print(rk_ord, dg_ord)
            for m_size in m_sizes:
                print("rk_ord", rk_ord, "dg_ord", dg_ord, "m_size", m_size)

                material_params, source_term, yz_plane, domain_eges = LOH1_setup(m_size, is_square=False)
                sx = Constant(s_m_ic[0], name='sx')  # initial disturbance
                sy = Constant(s_m_ic[1], name='sy')
                initial_solution = ic_origin_constants(sx, sy)
                control = Control(material_params['layer']['lam'])

                cost_func = ControllabilityCostFunction(yz_plane)
                u_sol, t_end, tstep = forward_LOH1(material_params=material_params, f=source_term,
                                                   unit_mesh=yz_plane, ic=initial_solution, dg_order=dg_ord, tend=1,
                                                   timestepping=rk_ord, cost_function=cost_func, annotate=True,
                                                   show_plots=False, show_final_plots=False)

                dx, dS, ds = get_measures(yz_plane)
                if use_mock_measurments:
                    u_measured = create_mock_measurments(dg_order=dg_ord, tend=1, unit_mesh=yz_plane, f=source_term,
                                                         timestepping=rk_ord, cost_function=cost_func,
                                                         material_params=material_params, origin_s=[26, 32])
                    cost_func.phi_m = u_measured
                    J = cost_func.J_final_evaluate(u_sol, u_measured)
                else:
                    J = assemble(inner(u_sol, u_sol) * dx)

                DF = np.shape(np.array(u_sol.vector()))[0]
                deg_free.append(DF)
                l2_error = errornorm(u_sol, u_fine, norm_type='l2', degree_rise=2)
                errors.append(l2_error)
                DJ = None
                conv_rate = None

                if m_size < 64:
                    DJ = compute_gradient(J, control)
                    if not only_gradient:
                        J_hat = ReducedFunctional(J, [control])
                        h = Constant(10.0)
                        conv_rate = taylor_test(J_hat, sx, h)

                print(rk_ord, rk_ord[-1])
                results.append([eq_name, m_size, dg_ord, DF, int(rk_ord[-1]),
                                l2_error, cost_func.get_name(), J, float(DJ), conv_rate])

    convergence_rate = [errors[k - 1] / errors[k] for k in range(1, len(m_sizes))]

    orders = [np.log(errors[k - 1] / errors[k]) / np.log((1 / m_sizes[k - 1]) / (1 / m_sizes[k])) for k in
              range(1, len(m_sizes))]

    if len(m_sizes) > 2:
        print(m_sizes[1::], deg_free[1::], errors[1::], convergence_rate, orders)
        table = np.stack([m_sizes[1::], deg_free[1::], errors[1::], convergence_rate, orders], axis=1)
        print(table)
        print(results)

    return results

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


def optimize_LOH1():
    # m_sizes = [8, 16]
    # dg_orders = [1, 2, 3]
    # time_order = ['RK1', 'RK2', 'RK3']

    dg_order = 2
    m_size = 8
    RKord = 'RK2'
    cost_function_type = "controllability"  # "sensor_cost_func"  # controllability
    bounds = [(0, 0), (58, 43)]

    # initial guess
    opt_param_str = "[sx, sy]"
    sx = Constant(20, name='sx')
    sy = Constant(20, name='sy')

    s_m = [26, 32]

    total_run_time = time.time()
    material_params, source_term, yz_plane, domain_edges = LOH1_setup(m_size)
    h_x = int(1 / domain_edges[0])
    h_y = int(1 / domain_edges[1])
    print("h_x", h_x, "h_y", h_y)

    # dx, dS, ds = get_measures(yz_plane)
    initial_solution = ic_origin_constants(sx, sy)

    # control parameters
    control_sx = Control(sx)
    control_sy = Control(sy)
    # rho_layer_control = Control(material_properties['layer']['rho'])
    # cost_function.start_record()
    print("cost_function_type", "M", m_size, "dg_order", dg_order, "rk_order")
    u_measured = create_mock_measurments(dg_order=dg_order, tend=1, unit_mesh=yz_plane, f=source_term,
                                         timestepping=RKord, material_params=material_params,
                                         cost_function=CostFunction(yz_plane), origin_s=s_m)

    # cost function setup
    if cost_function_type == 'sensor_cost_func':
        cost_function = SensorCostFunction(yz_plane)
    else:
        cost_function = ControllabilityCostFunction(yz_plane, u_measured)

    # TODO add some gaussian noise!
    # cost_function.end_record()
    print("starting initial annotation solve")
    forward_solve_time = time.time()
    u_sol, t_end, tsteps = forward_LOH1(material_params=material_params, f=source_term,
                                        unit_mesh=yz_plane, ic=initial_solution, dg_order=dg_order, tend=1,
                                        timestepping=RKord, cost_function=cost_function, annotate=True,
                                        show_plots=False,
                                        show_final_plots=False)

    forward_solve_time = time.time() - forward_solve_time
    print("end initial annotation solve")

    # J = assemble(inner(u_sol, u_sol) * dx)
    if cost_function_type == 'sensor_cost_func':
        J = cost_function.J_final_evaluate()
    else:
        J = cost_function.J_final_evaluate(u_sol, u_measured)
    print("J", float(J))

    J_hat = ReducedFunctional(J, [control_sx, control_sy], derivative_cb_post=derivative_cb_post)

    # optimization routine CG
    log_results = {}
    settings = {"method": "L-BFGS-B", "tol": 1.0e-12,
                "options": {"disp": True, "gtol": 1.0e-12, "maxiter": 100}}

    print("starting minimize")

    if len(bounds) > 0:
        opt_list, res = minimize(J_hat, bounds=bounds, method=settings['method'], tol=settings['tol'],
                                 options=settings['options'])
    else:
        opt_list, res = minimize(J_hat, method=settings['method'], tol=settings['tol'],
                                 options=settings['options'])

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
    log_results['model_settings'] = {'dg_order': dg_order, 'm_size': m_size, 'RK_order': RKord}
    log_results['optimal_parameters'] = list(map(float, opt_list))
    log_results['exact_parameters'] = s_m
    log_results['J'] = derivative_cb_list['J']
    log_results['DJ'] = derivative_cb_list['DJ']
    log_results['m'] = derivative_cb_list['m']
    log_results['Error'] = [np.linalg.norm(np.subtract(derivative_cb_list['m'][i], [26, 32])) for i in
                            range(0, len(derivative_cb_list['J']))]

    timestamp = time.time()
    rk = {'RK1': 'rk_1', 'RK2': 'rk_2', 'RK3': 'rk_3'}
    # file = '../opt_output/' + settings['method']
    bounds_tag = '' if len(bounds) == 0 else '_b_true'
    file = 'optimize_LOH1_m_{0}_dg_{1}_rk{2}{3}_{4}.json'.format(m_size, dg_order, rk[RKord], bounds_tag,
                                                                 int(timestamp))
    with open(file, 'w') as fp:
        json.dump(log_results, fp)

    print("...................................................................................................")
    print("initial error: ", np.linalg.norm([20 - 26]), np.linalg.norm([20 - 32]))
    print("Optimal parameters", list(map(float, opt_list)))
    print("final error: ", np.linalg.norm([float(opt_list[0]) - 26]), np.linalg.norm([float(opt_list[1]) - 32]))
    print("...................................................................................................")
    print("\n\n\n")


def LOH1_setup(m_size=20, is_square=False):
    # domain set up
    scaling = 1000
    y_right = 32000 / scaling
    y_left = 26000 / scaling
    z_layer = 1000 / scaling
    z_halfspace = 33000 / scaling

    # plane-y-z
    yz_plane = RectangleMesh(Point(0, 0), Point(y_right + y_left, z_layer + z_halfspace), m_size, m_size)

    # parameter set up
    domain_divide = Constant(z_layer,
                             name='divide')  # depth of the half space, point where the sub-domains are divided

    # flow speeds
    cp_layer = 4000 / scaling
    cs_layer = 2000 / scaling

    cp_hs = 6000 / scaling
    cs_hs = 3464 / scaling

    # define material constants
    rho_layer = Constant(2600 / scaling, name='rho_layer')
    mu_layer = Constant(cs_layer ** 2 * rho_layer, name='mu_layer')
    lam_layer = Constant(cp_layer ** 2 * rho_layer - 2 * mu_layer, name='lam_layer')

    rho_hs = Constant(2700 / scaling, name='rho_hs')
    mu_hs = Constant(cs_hs ** 2 * rho_hs, name='mu_hs')
    lam_hs = Constant(cp_hs ** 2 * rho_hs - 2 * mu_hs, name='lam_hs')

    # define local max velocity as expression depending on the material constants
    # cp = sqrt((lam + 2mu)/rho)
    # cs = sqrt(mu/rho)

    # re-define flow speed as constant depending on lam, rho, mu!
    cp_layer = Constant(sqrt((lam_layer + 2 * mu_layer) / rho_layer), name='cp_layer')
    cs_layer = Constant(sqrt(mu_layer / rho_layer), name='cs_layer')
    cp_hs = Constant(sqrt((lam_hs + 2 * mu_hs) / rho_hs), name='cp_hs')
    cs_hs = Constant(sqrt(mu_hs / rho_hs), name='cs_hs')

    material_params = {
        'layer': {'cp': cp_layer, 'cs': cs_layer, 'rho': rho_layer, 'mu': mu_layer, 'lam': lam_layer},
        'hs': {'cp': cp_hs, 'cs': cs_hs, 'rho': rho_hs, 'mu': mu_hs, 'lam': lam_hs},
        'divide': domain_divide}

    # source function
    source_term = Constant((0, 0, 0, 0, 0), name='f')
    domain_edges = [y_left + y_right, z_layer + z_halfspace]
    return material_params, source_term, yz_plane, domain_edges


def sensor_test():
    ic = ic_no_dependencies(degree=1)

    material_params, source_term, yz_plane, domain_edges = LOH1_setup(m_size=8)

    cost_func = SensorCostFunction(yz_plane)
    cost_func.start_record()
    V = VectorFunctionSpace(yz_plane, 'DG', 2, dim=5)
    phi = project(ic, V)

    plot_2dim_elastic(phi)

    cost_func.J_evaluate(phi, t=0, delta_t=0.1)


def direct_solve():
    timestepping = 'RK1'
    V = FunctionSpace()
    u, v = TrialFunction(V), TestFunction(V)

    # if timestepping == 'RK1':
    #    F = - inner(f, v) * dx  # source term (=0)
    #    F += elasticOp.generate_fem_formulation(u, v)
    #    F += dot((1 / dt) * (u - u1), v) * dx  # time discretization
    #    a, L = lhs(F), rhs(F)
    #    u_new = Function(V, name='u_new')


def plot_ic(m_size, dg_order, component_idx, s_m):
    material_params, source_term, yz_plane, domain_edges = LOH1_setup(m_size=m_size, is_square=False)

    V = VectorFunctionSpace(yz_plane, 'DG', dg_order, dim=5)
    initial_solution = ic_origin_constants(s_m[0], s_m[1])

    phi = project(initial_solution, V)
    c = plot(phi[component_idx], zorder=0)  #
    # plt.colorbar(c)
    return c


import matplotlib.tri as tri


def plot_contour(m_size, dg_order, component_idx, s_m):
    material_params, source_term, yz_plane, domain_edges = LOH1_setup(m_size=m_size, is_square=False)

    V = VectorFunctionSpace(yz_plane, 'DG', dg_order, dim=5)
    initial_solution = ic_origin_constants(s_m[0], s_m[1])

    phi = project(initial_solution, V)

    triang = tri.Triangulation(*yz_plane.coordinates().reshape((-1, 2)).T,
                               triangles=yz_plane.cells())
    Z = phi.sub(component_idx).compute_vertex_values(yz_plane)

    plt.figure()
    plt.contourf(triang, Z)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    # mock_sensors()
    # optimize_LOH1()
    taylor_test_LOH1(m_sizes=[8], dg_orders=[1], rk_orders=['RK1'], use_mock_measurments=False,
                     only_gradient=True)
    # runge_kutta_test()
