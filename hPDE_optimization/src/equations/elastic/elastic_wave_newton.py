import numpy as np
from dolfin import *
from dolfin_dg import *
from dolfin_adjoint import *
import time
import ufl
import matplotlib.pyplot as plt

from dolfin_dg.fluxes import ConvectiveFlux
from dolfin_dg.operators import DGFemFormulation

from src.equations.utils import plot_multidim, plot_1dim_elastic, plot_2dim_elastic, get_dt_cfl, get_measures, \
    write_to_vtk, plot_solution_1d, get_dt_cfl_terradg


# TODO:
# 1. 1dim elastic wave -> correct
# 2. 2dim elastic wave -> TODO test

def get_max_constant(list):
    max_const = Constant(0.0)
    for const in list:
        if max_const < const:
            max_const = const

    return max_const


class LocalLaxFriedrichsModified(ConvectiveFlux):
    """Implementation of symbolic representation of the local-Lax Friedrichs
    flux function
    """

    def __init__(self, maxvel):
        self.alpha = maxvel

    def setup(self, F_c, u_p, u_m, n):
        pass

    def interior(self, F_c, u, n):
        return dot(avg(F_c(u)), n) + 0.5 * self.alpha('+') * jump(u)

    def exterior(self, F_c, u_p, u_m, n):  # self.alpha
        return 0.5 * (dot(F_c(u_p), n) + dot(F_c(u_m), n) + self.alpha * (u_p - u_m))


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
        residual += inner(self.H.interior(self.F_c, u, n('+')), jump(v)) * dS

        for bc in self.dirichlet_bcs:
            gD = bc.get_function()
            dSD = bc.get_boundary()

            self.H.setup(self.F_c, u, gD, n)
            residual += inner(self.H.exterior(self.F_c, u, gD, n), v) * dSD

        for bc in self.neumann_bcs:
            dSN = bc.get_boundary()

            residual += inner(dot(self.F_c(u), n), v) * dSN

        return residual


class ElasticOperator_1dim(HyperbolicOperatorModified):

    def get_maxvel_float(self):
        return float(sqrt(self.mu / self.rho))

    def __init__(self, mesh_, V, bcs, material_parameters):
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

        self.mu = material_parameters['mu']
        self.rho = material_parameters['rho']

        self.F_c = lambda U: as_vector(((-self.mu * U[1],), (-U[0] / self.rho,)))
        self.alpha = sqrt(self.mu / self.rho)  # lambda U, n: ufl.sqrt(self.mu / self.rho)

        HyperbolicOperatorModified.__init__(self, mesh_, V, bcs, self.F_c, LocalLaxFriedrichsModified(self.alpha))


# Define class according to dolfin_dg
class ElasticOperator(HyperbolicOperatorModified):

    def alpha_array(self, U, n):
        # material_domain = "((x[0] > 0.5 && x[0] < 1.5) && (x[1] > (0.4) && x[1] < (0.6))) ? inside : outside"
        # cp = Expression(material_domain, inside=20, outside=2, degree=1)
        # cs = Expression(material_domain, inside=10, outside=1, degree=1)

        cp = self.material_parameters['cp']
        cs = self.material_parameters['cs']
        lambdas = [Constant(cp), Constant(cs), 0]

        return lambdas

    def get_maxvel_float(self):
        return np.max([float(self.material_parameters['cp']), float(self.material_parameters['cs']), 0])

    def __init__(self, mesh_, V, bcs, material_parameters):
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
        self.material_parameters = material_parameters
        rho = self.material_parameters['rho']
        lam = self.material_parameters['lam']
        mu = self.material_parameters['mu']

        comp_str = "sqrt((lam + 2 * mu) / rho) >= sqrt(mu / rho)"
        cp_str = "sqrt((lam + 2 * mu) / rho)"
        cs_str = "sqrt(mu / rho)"
        self.maxvel = Expression("{0} ? {1} : {2}".format(comp_str, cp_str, cs_str), degree=1, rho=rho, mu=mu, lam=lam)

        self.maxvel.dependencies = [rho, mu, lam]
        dmaxvel_drho = " {0} ? {1} : {2}".format(comp_str, "-(1/(2*rho))*{}".format(cp_str),
                                                 "-(1/(2*rho))*{}".format(cs_str))

        dmaxvel_dmu = " {0} ? {1} : {2}".format(comp_str, "1/(2*rho*sqrt((lam + mu)/rho))", "1/(2*mu)*sqrt(mu/rho)")

        dmaxvel_dlam = " {0} ? {1} : {2}".format(comp_str, "1/(2*rho*sqrt((lam + mu)/rho))", "0.0")

        self.maxvel.user_defined_derivatives = {rho: Expression(dmaxvel_drho, rho=rho, mu=mu, lam=lam, degree=1),
                                                mu: Expression(dmaxvel_dmu, rho=rho, mu=mu, lam=lam, degree=1),
                                                lam: Expression(dmaxvel_dlam, rho=rho, mu=mu, lam=lam, degree=1)}

        def F_c(U):
            sig11, sig22, sig12, u, v = U[0], U[1], U[2], U[3], U[4]

            # planar wave
            rho = self.material_parameters['rho']
            lam = self.material_parameters['lam']
            mu = self.material_parameters['mu']

            # Material inclusion
            # lam_in, lam_out = 200, 2
            # mu_in, mu_out = 100, 1
            # material_domain = "((x[0] > 0.5 && x[0] < 1.5) && (x[1] > (0.4) && x[1] < (0.6))) ? inside : outside"
            # lam = Expression(material_domain, inside=lam_in, outside=lam_out, degree=2)
            # mu = Expression(material_domain, inside=mu_in, outside=mu_out, degree=2)

            return as_matrix([[-(lam + 2 * mu) * u, -lam * v],
                              [-lam * u, -(lam + 2 * mu) * v],
                              [-mu * v, -mu * u],
                              [-(1 / rho) * sig11, -(1 / rho) * sig12],
                              [-(1 / rho) * sig12, -(1 / rho) * sig22]])

        HyperbolicOperatorModified.__init__(self, mesh_, V, bcs, F_c, LocalLaxFriedrichsModified(self.maxvel))


def F_operators(material_parameters, phi_old, gD, v, f, face_normal, flux, maxvel, unit_mesh):
    dx, dS, ds = get_measures(unit_mesh)

    flux_eval = flux(phi_old, material_parameters)  # evaluate physical flux at phi_old

    F_op = inner(grad(v), flux_eval) * dx
    F_op += inner(v, f) * dx

    flux_dS_ = dot(avg(flux_eval), face_normal('+')) + 0.5 * Constant(maxvel) * jump(phi_old)
    flux_ds_ = 0.5 * (dot(flux_eval, face_normal) + dot(flux(gD, material_parameters), face_normal)) + 0.5 * Constant(
        maxvel) * (
                       phi_old - gD)
    F_bc = 0
    F_bc -= inner(flux_dS_, jump(v)) * dS
    F_bc -= inner(flux_ds_, v) * ds

    return F_op, F_bc


def weak_formulation_1dim(phi, gD, v, f, material_parameters, unit_mesh):
    maxvel = ufl.Max(material_parameters['cp'],
                     material_parameters[
                         'cs'])  # np.max([parameters['cp'], parameters['cs'], 0])  # np.max([np.sqrt(mu / rho), 0])

    flux = lambda phi, material_parameters: as_vector(((-material_parameters['mu'] * phi[1],),
                                                       (-phi[0] / material_parameters['rho'],)))

    normal = FacetNormal(unit_mesh)

    F_op, F_bc = F_operators(material_parameters, phi, gD, v, f, normal, flux, maxvel, unit_mesh)

    return F_op + F_bc, maxvel  # TODO construct matrices, F_bc


def weak_formulation_2dim(phi, gD, v, f, material_parameters, unit_mesh):
    cp, cs = material_parameters['cp'], material_parameters['cs']
    maxvel = Expression("cp >= cs ? cp : cs", cp=cp, cs=cs, degree=1)

    def flux(phi, material_parameters):
        sig11, sig22, sig12, u, v = phi[0], phi[1], phi[2], phi[3], phi[4]

        # planar wave
        rho = material_parameters['rho']
        lam = material_parameters['lam']
        mu = material_parameters['mu']

        return as_matrix([[-(lam + 2 * mu) * u, -lam * v],
                          [-lam * u, -(lam + 2 * mu) * v],
                          [-mu * v, -mu * u],
                          [-(1 / rho) * sig11, -(1 / rho) * sig12],
                          [-(1 / rho) * sig12, -(1 / rho) * sig22]])

    n = FacetNormal(unit_mesh)

    flux_eval = flux(phi, material_parameters)  # physical flux
    print(grad(v).ufl_shape, flux_eval.ufl_shape)
    F = inner(grad(v), flux_eval) * dx
    F += inner(v, f) * dx

    flux_dS_ = dot(avg(flux_eval), n('+')) + 0.5 * maxvel('+') * jump(phi)
    flux_bc = flux(phi, material_parameters)
    flux_ds_ = dot(0.5 * (flux_eval + flux_bc), n) + 0.5 * maxvel * (phi - gD)

    F -= inner(flux_dS_, jump(v)) * dS
    F -= inner(flux_ds_, v) * ds

    return F


def weak_formulation(phi, gD, v, f, material_parameters, unit_mesh):
    if phi.ufl_shape == (2,):
        print("1dim elastic wave weak form")
        return weak_formulation_1dim(phi, gD, v, f, material_parameters, unit_mesh)
    else:
        return weak_formulation_2dim(phi, gD, v, f, material_parameters, unit_mesh)


def forward_elastic_newton(material_parameters, gD, f, unit_mesh, h, dg_order, tend, timestepping, out_file=None,
                           show_plots=False, show_final_plots=False):
    dx, dS, ds = get_measures(unit_mesh)
    dimension = unit_mesh.geometric_dimension()

    if dimension == 1:
        V = VectorFunctionSpace(unit_mesh, 'DG', dg_order, dim=2)
    else:
        V = VectorFunctionSpace(unit_mesh, 'DG', dg_order, dim=5)

    phi1 = project(gD, V)
    tag = "M_{0}_D_{1}_{2}".format(int(1 / h), dg_order, timestepping)
    if show_plots:
        # plot_multidim(phi1)
        labels = ["$\phi1^{11}$", "$\phi1^{22}$", "$\phi1^{12}$", "$u$", "$v$"]
        for i in range(0, 3):
            plot_solution_1d(0.0, dim=2, function=phi1, analytic=gD, param_idx=i, show=i == 2,
                             label_=labels[i], tag=tag)
        for i in range(3, 5):
            plot_solution_1d(0.0, dim=2, function=phi1, analytic=gD, param_idx=i, show=i == 4,
                             label_=labels[i], tag=tag)

    u, v = Function(V), TestFunction(V)

    parameters["ghost_mode"] = "shared_facet"

    # Set up boundary condition

    if dimension == 2:
        elasticOp = ElasticOperator(unit_mesh, V, DGDirichletBC(ds, gD), material_parameters)
    else:
        elasticOp = ElasticOperator_1dim(unit_mesh, V, DGDirichletBC(ds, gD), material_parameters)

    maxvel = elasticOp.get_maxvel_float()
    factor = 0.01 if dg_order == 1 else 0.1
    delta_t = float(get_dt_cfl_terradg(maxvel, h, dg_order, factor=factor))
    steps = int(tend / delta_t)
    print("maxvel", maxvel, "delta_t", delta_t, " type ", type(delta_t), "h", h)

    # if timestepping == 'RK1':
    F = - inner(f, v) * dx
    F += dot((1 / delta_t) * (u - phi1), v) * dx
    F += elasticOp.generate_fem_formulation(u, v)

    if timestepping == 'RK2':
        phi2 = Function(V, name='phi2')
        F2 = - inner(f, v) * dx
        F2 += dot((1 / delta_t) * (u - phi2), v) * dx  # Euler time stepping
        F2 += elasticOp.generate_fem_formulation(u, v)
    if timestepping == 'RK3':
        phi2 = Function(V, name='phi2')
        F2 = - inner(f, v) * dx
        F2 += dot((1 / delta_t) * (u - phi2), v) * dx  # Euler time stepping
        F2 += elasticOp.generate_fem_formulation(u, v)

        phi3 = Function(V, name='phi3')
        F3 = - inner(f, v) * dx
        F3 += dot((1 / delta_t) * (u - phi2), v) * dx  # Euler time stepping
        F3 += elasticOp.generate_fem_formulation(u, v)

    du = TrialFunction(V)
    J = derivative(F, u, du)
    t = 0

    for n in range(0, steps):

        if timestepping == 'RK2':

            solve(F == 0, u, [], J=J)
            phi2.assign(u)  # phi1 + delta_t * u
            gD.t = t + delta_t
            solve(F2 == 0, u, [], J=J)
            phi1.assign(0.5 * (phi1 + u))
            # phi1.assign(0.5 * (phi1 + phi2 + delta_t*u))

        elif timestepping == 'RK3':
            solve(F == 0, u, [], J=J)
            phi2.assign(u)

            gD.t = gD.t + delta_t
            solve(F2 == 0, u, [], J=J)
            phi3.assign(0.75 * phi1 + 0.25 * (phi2 + u))

            gD.t = gD.t + 0.5 * delta_t
            solve(F3 == 0, u, [], J=J)

            phi1.assign((1 / 3) * phi1 + (2 / 3) * (phi3 + u))
        else:
            solve(F == 0, u, [], J=J)
            phi1.assign(u)

        t += delta_t
        gD.t = t  # only set t if gD is Expression

        if n % int((0.1 * steps)) == 0 or n == steps - 1:
            print("n=", n, "remaining steps:", steps - n, "total steps:", steps)

            if show_plots:
                labels = ["$\phi1^{11}$", "$\phi1^{22}$", "$\phi1^{12}$", "$u$", "$v$"]
                for i in range(0, 3):
                    plot_solution_1d(t, dim=2, function=phi1, analytic=gD, param_idx=i, show=i == 2,
                                     label_=labels[i], tag=tag)
                for i in range(3, 5):
                    plot_solution_1d(t, dim=2, function=phi1, analytic=gD, param_idx=i, show=i == 4,
                                     label_=labels[i], tag=tag)

    # plot_multidim(phi1, t)
    # plot_multidim(project(gD,V), t)

    print(delta_t)
    print("\nErrors\n")

    if show_final_plots:
        labels = ["$\phi1^{11}$", "$\phi1^{22}$", "$\phi1^{12}$", "$u$", "$v$"]
        for i in range(0, 3):
            plot_solution_1d(t, dim=2, function=phi1, analytic=gD, param_idx=i, show=i == 2,
                             label_=labels[i], tag=tag)
        for i in range(3, 5):
            plot_solution_1d(t, dim=2, function=phi1, analytic=gD, param_idx=i, show=i == 4,
                             label_=labels[i], tag=tag)
        # plot_multidim(phi1, t=t)

    return phi1, t


if __name__ == '__main__':
    print("main")
