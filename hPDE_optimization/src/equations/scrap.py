# ---------------------------------------
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import sys


def fem_formulation(V, delta_t, phi, v, q_old, u_exact, beta):
    print("Using FEM")

    bcs = DirichletBC(V, u_exact, "on_boundary")
    F = (Constant(1 / delta_t) * (phi - q_old) * v + dot(beta, grad(phi)) * v) * dx

    return F, bcs


def domain_boundaries():
    """
    Boundary definitions
    :return: strings defining inflow, outflow, top_bottom
    """
    inflow = 'near(x[0], 0)'
    outflow = 'near(x[0], 1)'
    top_bottom = 'near(x[1], 0) || near(x[1], 1)'

    return inflow, outflow, top_bottom


def dirichlet_bcs(V_dg, u_in, u_out, u_exact):
    """
    Create periodic boundaries

    :param V_dg: function space
    :param u_in: inflow expression
    :param u_out: outflow expression
    :param u_exact: exact solution expression for top/bottom
    :return: list of boundaries
    """
    inflow, outflow, walls = domain_boundaries()
    bcu_top_bottom = DirichletBC(V_dg, u_exact, walls, method='geometric')
    bcp_inflow = DirichletBC(V_dg, u_in, inflow, method='geometric')
    bcp_outflow = DirichletBC(V_dg, u_out, outflow, method='geometric')

    return [bcu_top_bottom, bcp_inflow, bcp_outflow]


class Beta(UserExpression):
    def __init__(self, c0=Constant(0.5), c1=Constant(0.5), **kwargs):
        """ Construct the source function """
        super().__init__(self, **kwargs)
        self.t = 0.0
        self.c0 = c0
        self.c1 = c1

    def eval(self, value, x):
        """ Evaluate the source function """
        if x[0] < 0.5:
            value[0] = self.c0
            value[1] = 0
        else:
            value[0] = self.c1
            value[1] = 0

    def value_shape(self):
        return (2,)


class BetaDerivative(UserExpression):
    def __init__(self, c0=Constant(0.5), c1=Constant(0.5), Source=None, **kwargs):
        """ Construct the source function derivative """
        super().__init__(**kwargs)
        self.t = 0.0
        self.c0 = c0
        self.c1 = c1
        self.source = Source  # needed to get the matching time instant

    def eval(self, value, x):
        """ Evaluate the source function's derivative """
        if x[0] < 0.5:
            value[0] = 0.0
            value[1] = 0.0
        else:
            value[0] = 0.0
            value[1] = 0.0

    def value_shape(self):
        return (2,)


def forward_test():
    FSPACE = 'DG'
    vel = -0.2
    dt = 0.0001
    tmax = 1

    parameters["ghost_mode"] = "shared_vertex"

    class PeriodicDomain(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0)

        def map(self, x, y):
            y[0] = x[0] - 1.0

    constrained_domain = PeriodicDomain()

    mesh = UnitIntervalMesh(1000)
    V = FunctionSpace(mesh, FSPACE, 1, constrained_domain=constrained_domain)
    u = TrialFunction(V)
    v = TestFunction(V)

    uf = Function(V)  # New value
    upf = Function(V)  # Previous value

    u0 = Expression("sin(2*pi*(x[0] - 0.2*t))", degree=2, t=1)
    project(u0, V, function=upf)

    pyplot.figure(figsize=(10, 6))
    pyplot.plot(mesh.coordinates()[:, 0], upf.compute_vertex_values(), label='t=%g' % t)

    uc0 = Constant(vel)
    u_conv = Constant([vel])
    if FSPACE == 'CG':
        eq = (Constant(1 / dt) * (u - upf) * v + uc0 * u.dx(0) * v) * dx
    elif FSPACE == 'DG':
        n = FacetNormal(mesh)
        flux_nU = u * (dot(u_conv, n) + abs(dot(u_conv, n))) / 2
        flux = jump(flux_nU)  # flux_nU('+') - flux_nU('-')
        eq = Constant(1 / dt) * (u - upf) * v * dx - u * uc0 * v.dx(0) * dx + flux * jump(v) * dS

    a, L = lhs(eq), rhs(eq)

    t = 0

    for t in numpy.arange(numpy.ceil(tmax / dt)) * dt:
        solve(a == L, uf)
        upf.assign(uf)

        # if near(t, round(t)):
        #    pyplot.plot(mesh.coordinates()[:, 0], upf.compute_vertex_values(), label='t=%g' % t)

        print(t)

    pyplot.plot(mesh.coordinates()[:, 0], upf.compute_vertex_values(), label='t=%g' % t)
    pyplot.ylim(-1.1, 1.1)
    pyplot.legend(loc='best')
    pyplot.show()


# old advection forward
def forward_old(beta, unit_mesh, q_exact, tend=1, dg_order=3, flux_method='LaxF',
                time_stepping='RK1', vtks=False, plots=False):
    max_flow_speed = get_max_flow_speed(beta)
    delta_t = get_dt_cfl(float(max_flow_speed), unit_mesh.hmax(), dg_order, factor=0.5)

    dx = Measure('dx', domain=unit_mesh)  # area
    dS = Measure('dS', domain=unit_mesh)  # interior boundaries
    ds = Measure('ds', domain=unit_mesh)  # exterior boundaries

    time_steps = int(tend / delta_t)
    f = Constant(0.0)

    V = FunctionSpace(unit_mesh, 'DG', dg_order)
    phi, v = Function(V), TestFunction(V)
    q_exact.t = 0
    q_old = project(q_exact, V)

    if plots:
        plot(q_old)
        plt.title("initial solution t=0")
        plt.show()

    n = FacetNormal(unit_mesh)
    F_u = beta * phi

    if 'upwind' == flux_method:
        un = (dot(beta, n) + abs(dot(beta, n))) / 2.0
        flux = jump(un * phi)
        flux_ds = un * phi - un * q_exact
    else:  # 'laxF'
        max_eigenvalue = get_max_flow_speed(beta)
        flux = dot(avg(F_u), n('+')) + 0.5 * max_eigenvalue * jump(phi)
        # flux_ds = dot(0.5 * (F_u + beta * q_exact), n) + 0.5 * max_eigenvalue * jump(phi)

    F = dot(Constant(1 / delta_t) * (phi - q_old), v) * dx - dot(grad(v), F_u) * dx - v * f * dx
    F += dot(flux, jump(v)) * dS + dot(flux, v) * ds  # + dot(v, u_exact * phi) * ds + dot(v, un * phi) * ds

    t = 0
    times = [t, ]

    out_file = None
    if vtks and out_file is not None:
        out_file = File("../vtks/tests/{}/advection2d.pvd".format(int(time.time())))
        out_file << (q_old, t)

    for n in range(time_steps):
        solve(F == 0, phi)
        q_old.assign(phi)

        if n % 100 == 0 and plots:
            plot(q_old)
            plt.title("solution t={}".format(t))
            plt.show()
            print("at time t", t)

        t += delta_t
        q_exact.t = t
        times.append(t)

        if vtks and out_file is not None:
            out_file << (q_old, t)

    plot(q_old)
    plt.title("solution t={}".format(t))
    plt.show()

    return q_old


def adjoint_system():
    y1 = np.array(
        [-0.20677272882130085, -0.19773576836617324, -0.1800568059919554, -0.1545084971874737, -0.12220742564187125,
         -0.08456530317942905, -0.04322727117869955, 0.0, 0.04322727117869955, 0.08456530317942905, 0.12220742564187145,
         0.1545084971874737, 0.18005680599195534, 0.19773576836617307, 0.20677272882130027, 0.2067727288213003,
         0.19773576836617315, 0.1800568059919554, 0.1545084971874738, 0.12220742564187125, 0.08456530317942935,
         0.04322727117869976, 0.0, -0.04322727117869955, -0.08456530317942894, -0.12220742564187113,
         -0.15450849718747361, -0.1800568059919555, -0.1977357683661731, -0.20677272882130054])

    y15 = np.array(
        [-2.3853244179298665, -2.6919938314794987, -2.8810101969738886, -2.9441125922397746, -2.8785431398051533,
         -2.6871675393785597, -2.3783498230206472, -1.9655868068080753, -1.4669182152060374, -0.9041382585449305,
         -0.3018431214486899, 0.3136440084588507, 0.9154233901658103, 1.4771943770337355, 1.974404881260222,
         2.3853244179298665, 2.6919938314794987, 2.881010196973885, 2.944112592239772, 2.878543139805151,
         2.6871675393785557, 2.3783498230206472, 1.9655868068080768, 1.4669182152060405, 0.904138258544932,
         0.3018431214486899, -0.3136440084588507, -0.9154233901658088, -1.477194377033734, -1.9744048812602204])

    y_end = np.array(
        [1.3372683027305228, 1.1870234437754625, 0.9848999643569731, 0.7397316304217058, 0.4622334746108214,
         0.16453349791714567, -0.14035738215482793, -0.43911397111719125, -0.7186791724391204, -0.9668346453201737,
         -1.1727348048133757, -1.3273808259302877, -1.4240139354741632, -1.4584108028608251, -1.429068119930995,
         -1.3372683027305214, -1.187023443775462, -0.9848999643569731, -0.7397316304217058, -0.4622334746108216,
         -0.1645334979171461, 0.1403573821548275, 0.4391139711171909, 0.7186791724391204, 0.9668346453201738,
         1.172734804813376, 1.3273808259302893, 1.4240139354741639, 1.458410802860826, 1.4290681199309963])

    n = 30
    h = 1 / n
    delta_t = 1 / n

    x = np.linspace(0 + 0.5 * h, 1 - 0.5 * h, int(1 / h))
    print(len(x))
    print(h)

    beta = 0.25
    k = 2 * np.pi

    # t = 1.0
    y = [y_end]
    for j, t in enumerate([]):
        drhs_dbeta = lambda x: list(map(lambda x: -k * t * np.cos(k * (x - t * beta)), x))
        diff = lambda x: list(map(lambda x: np.sin(k * (x - t * 0.25)) - np.sin(k * (x - t * 0.1)), x))

        print("exact ", drhs_dbeta(x))
        print("AD drhsdbeta ", y[j])

        plt.plot(x, drhs_dbeta(x), label='exact t=' + str(t))
        plt.plot(x, y[j], label='AD t=' + str(t), linestyle='--')
        plt.legend()
    # plt.show()

    t = 1.0
    # dJdbeta = h * np.dot(diff(x), np.array(drhs_dbeta(x)))
    # print(dJdbeta)

    print("2dim")

    # x_2dim = n * list(x)
    # dJdbeta = h * h * np.dot(diff(x_2dim), np.array(drhs_dbeta(x_2dim)))
    # print(dJdbeta)

    lam0_1dim = np.array(
        [-1.3500737302606325, -1.1989000515390031, -0.9953286876043914, -0.7482566839045226, -0.468482272584093,
         -0.16823293792437996, 0.13936898339382983, 0.4408798113711482, 0.723122076015493, 0.973760436013258,
         1.1818407923362528, 1.3382690349328326, 1.4362084989755397, 1.4713787599200154, 1.4422427085973508,
         1.3500737302606325, 1.1989000515390031, 0.9953286876043914, 0.7482566839045226, 0.4684822725840931,
         0.16823293792438015, -0.13936898339382955, -0.440879811371148, -0.723122076015493, -0.973760436013258,
         -1.1818407923362528, -1.3382690349328326, -1.4362084989755397, -1.4713787599200154, -1.4422427085973508
         ])
    N = len(lam0_1dim)
    x = np.linspace(h, 1 - h, N)
    b = 0.25
    b_m = 0.1
    T = 1.0
    lam1_exact = list(
        map(lambda x: 2 * (np.sin(k * ((x + b * 0.0) - T * b)) - k * np.sin(k * ((x + b * 0.0) - T * b_m))), x))

    lam0_exact = list(
        map(lambda x: 2 * (np.sin(k * ((x + b * t) - T * b)) - k * np.sin(k * ((x + b * t) - T * b_m))), x))

    h = 1 / N

    print(np.divide(lam0_exact, -lam0_1dim))
    print(np.divide(-lam0_1dim, lam0_exact))

    plt.plot(x, lam1_exact, label='lam1 exact')
    plt.plot(x, lam0_exact, label='lam0 exact')
    plt.plot(x, -8 * lam0_1dim, label='lam 0')
    plt.legend()
    plt.show()


def sym_tests():
    divide = Constant(0.9)
    lam_layer = Constant(1.0)
    lam_hs = Constant(2.0)

    from sympy.printing import ccode
    import sympy as sp

    divide_sym, lam_layer_sym, lam_hs_sym, x = sp.symbols('divide, lam_layer, lam_hs, x[1]')

    lam_sym = sp.Piecewise((lam_layer_sym, x > divide_sym), (lam_hs_sym, x <= divide_sym))

    df = lam_sym.diff(divide_sym, 1)
    print("lam_sym w.r.t. divide", ccode(df))

    # df = Expression(ccode(df), degree=2, divide=divide)

    def create_sym_exp():
        cs_layer, cp_layer, cs_hs, cp_hs, x = sp.symbols('cs_layer, cp_layer, cs_hs, cp_hs, x[1]')

        layer_maxvel = sp.Piecewise((cs_layer, cs_layer > cp_layer), (cp_layer, cs_layer <= cp_layer))
        hs_maxvel = sp.Piecewise((cs_hs, cs_hs > cp_hs), (cp_hs, cs_hs <= cp_hs))

        maxvel_sym = sp.Piecewise((layer_maxvel, x > divide_sym), (hs_maxvel, x <= divide_sym))

        dmaxvel = maxvel_sym.diff(divide_sym, 1)

        # print("maxvel w.r.t. cp_layer ", ccode(maxvel_sym.diff(cp_hs, 1)))
        print("maxvel w.r.t. divide ", ccode(dmaxvel))

        return maxvel_sym, dmaxvel

    print("\n -------------------------------- \n")
    maxvel_sym, dmaxvel = create_sym_exp()
    cs_layer, cp_layer, cs_hs, cp_hs = Constant(1.0), Constant(2.0), Constant(3.0), Constant(4.0)
    unit_mesh = UnitSquareMesh(50, 50)
    V = FunctionSpace(unit_mesh, 'DG', 3)

    maxvel_exp = Expression(ccode(maxvel_sym), degree=1, cs_layer=cs_layer, cp_layer=cp_layer, cs_hs=cs_hs, cp_hs=cp_hs,
                            divide=divide)
    # u = project(maxvel_exp, V)
    # plot(u)
    # plt.show()


print(sys.argv)

from dolfin import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def tmp():
    def mesh2triang(mesh):
        xy = mesh.coordinates()
        return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

    def plot(obj):
        plt.gca().set_aspect('equal')
        if isinstance(obj, Function):
            mesh = obj.function_space().mesh()
            if (mesh.geometry().dim() != 2):
                raise (AttributeError)
            if obj.vector().size() == mesh.num_cells():
                C = obj.vector().array()
                plt.tripcolor(mesh2triang(mesh), C)
            else:
                C = obj.compute_vertex_values(mesh)
                plt.tripcolor(mesh2triang(mesh), C, shading='gouraud')
        elif isinstance(obj, Mesh):
            if (obj.geometry().dim() != 2):
                raise (AttributeError)
            plt.triplot(mesh2triang(obj), color='k')

    # example
    mesh = UnitSquareMesh(10, 10)
    plt.figure()
    plot(mesh)
    plt.show()
    Q = FunctionSpace(mesh, "CG", 1)
    F = interpolate(Expression("x[0]"), Q)
    plot(F)
    plt.show()


if __name__ == '__main__':
    forward_times = np.array([[15, 30.708099], [30, 46.934199], [60, 177.796064]])
    reverse_times = np.array([[15, 4.652970], [30, 9.41144], [60, 54.21980]])
    plt.plot(forward_times[:, 0] ** 2, forward_times[:, 1], label="forward", marker=".")
    plt.plot(reverse_times[:, 0] ** 2, reverse_times[:, 1], label="reverse", marker=".")
    plt.legend()
    plt.ylabel("Compute time [s]")
    plt.xlabel("Number of dofs")
    plt.savefig("compute_time.png", dpi=200)
    plt.show()
