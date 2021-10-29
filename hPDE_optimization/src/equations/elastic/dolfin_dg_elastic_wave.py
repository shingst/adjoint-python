import numpy as np
from dolfin import *
from dolfin_dg import *
from dolfin_adjoint import *
import time
import ufl
import matplotlib.pyplot as plt

from dolfin_dg.fluxes import ConvectiveFlux
from dolfin_dg.operators import DGFemFormulation

from src.equations.utils import plot_multidim, get_measures, plot_solution_1d, get_dt_cfl_terradg


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

            return as_matrix([[-(lam + 2 * mu) * u, -lam * v],
                              [-lam * u, -(lam + 2 * mu) * v],
                              [-mu * v, -mu * u],
                              [-(1 / rho) * sig11, -(1 / rho) * sig12],
                              [-(1 / rho) * sig12, -(1 / rho) * sig22]])

        HyperbolicOperatorModified.__init__(self, mesh_, V, bcs, F_c, LocalLaxFriedrichsModified(self.maxvel))


def forward_elastic(material_parameters, gD, f, unit_mesh, h, dg_order, tend, timestepping, out_file=None,
                    annotate=False, show_plots=False, show_final_plots=False):
    dx, dS, ds = get_measures(unit_mesh)
    dimension = unit_mesh.geometric_dimension()

    if dimension == 1:
        V = VectorFunctionSpace(unit_mesh, 'DG', dg_order, dim=2)
    else:
        V = VectorFunctionSpace(unit_mesh, 'DG', dg_order, dim=5)

    phi1 = project(gD, V)

    if show_plots:
        plot_multidim(phi1)

    u, v = TrialFunction(V), TestFunction(V)
    parameters["ghost_mode"] = "shared_facet"

    # Set up boundary condition
    if dimension == 2:
        elasticOp = ElasticOperator(unit_mesh, V, DGDirichletBC(ds, gD), material_parameters)
    else:
        elasticOp = ElasticOperator_1dim(unit_mesh, V, DGDirichletBC(ds, gD), material_parameters)

    maxvel = elasticOp.get_maxvel_float()

    delta_t = float(get_dt_cfl_terradg(maxvel, h, dg_order, factor=0.1))
    steps = int(tend / delta_t)
    print("maxvel", maxvel, "delta_t", delta_t, " type ", type(delta_t))
    a = dot(u, v) * dx

    if timestepping >= 'RK2':
        L1 = inner(f, v) * dx
        L1 -= elasticOp.generate_fem_formulation(phi1, v)

        phi2 = Function(V, name='phi2')
        L2 = inner(f, v) * dx
        L2 -= elasticOp.generate_fem_formulation(phi2, v)

        if timestepping == 'RK3':
            phi3 = Function(V, name='phi3')
            L3 = inner(f, v) * dx
            L3 -= elasticOp.generate_fem_formulation(phi3, v)
    else:
        # Euler
        L1 = inner(f, v) * dx - elasticOp.generate_fem_formulation(phi1, v)
        const_delta_t = Constant(1 / delta_t, name='delta_t')
        F = L1 - inner(const_delta_t * (u - phi1), v) * dx
        a, L1 = lhs(F), rhs(F)

    gD.t = 0.0
    t = 0

    u_new = Function(V)
    for n in range(0, steps):

        if timestepping == 'RK3':
            solve(a == L1, u_new, annotate=annotate)
            phi2.assign(phi1 + delta_t * u_new, annotate=annotate)

            gD.t = gD.t + delta_t
            solve(a == L2, u_new, annotate=annotate)
            phi3.assign(0.75 * phi1 + 0.25 * (phi2 + delta_t * u_new), annotate=annotate)

            gD.t = gD.t + 0.5 * delta_t
            solve(a == L3, u_new, annotate=annotate)
            phi1.assign((1 / 3) * phi1 + (2 / 3) * (phi3 + delta_t * u_new), annotate=annotate)

        elif timestepping == 'RK2':
            solve(a == L1, u_new)
            phi2.assign(phi1 + delta_t * u_new)

            gD.t = t + delta_t
            solve(a == L2, u_new)
            phi1.assign(0.5 * phi1 + 0.5 * (phi2 + delta_t * u_new))
        else:
            solve(a == L1, u_new)
            phi1.assign(u_new)

        t += delta_t
        gD.t = t  # only set t if gD is Expression

        if n % int((0.1 * steps)) == 0:
            print("n=", n, "remaining steps:", steps - n, "total steps:", steps)

            if show_plots:
                for i in range(0, 5):
                    plot_solution_1d(t, dim=2, function=phi1, analytic=gD, param_idx=i, show=True,
                                     title=" idx=" + str(i))
    # plot_multidim(phi1, t)
    # plot_multidim(project(gD,V), t)

    print(delta_t)
    print("\nErrors\n")

    if show_final_plots:
        for i in range(0, 5):
            plot_solution_1d(t, dim=2, function=phi1, analytic=gD, param_idx=i, show=True, title=" idx=" + str(i))

        plot_multidim(phi1, t=t)

    return phi1, t


if __name__ == '__main__':
    print("main")
