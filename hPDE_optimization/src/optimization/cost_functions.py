from dolfin import *
from dolfin_adjoint import *
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.integrate import simps

from src.equations.utils import get_dt_cfl, get_dt_cfl_terradg
from src.equations.utils import write_to_vtk, plot_solution_1d, set_params_, get_measures, \
    eval_function_along_x, plot_solution

parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["cpp_optimize"] = True


class TimeTrackingCostFunction:

    def __init__(self, u_target, unit_mesh, t_start=0):
        self.J_values_at_time_k = []
        self.J_time_eval = None

        self.t_start = t_start
        self.t_end = None
        self.u_target = u_target
        self.unit_mesh = unit_mesh

    def J_evaluate(self, u, t, delta_t):
        self.u_target.t = t  # make sure same time step is used!
        J_eval = assemble((0.5 * (u - self.u_target) ** 2) * dx)
        self.J_values_at_time_k.append(float(delta_t) * J_eval)
        self.t_end = t

    def J_final_evaluate(self, u_final, t):
        self.u_target.t = t  # make sure same time step is used!
        J_eval_final = assemble((0.5 * (u_final - self.u_target) ** 2) * dx)
        J = 0
        for i in range(1, len(self.J_values_at_time_k)):
            J += 0.5 * (self.J_values_at_time_k[i - 1] + self.J_values_at_time_k[i])

        return J + J_eval_final

    def get_name(self):
        return "TimeTrackingCostFunction"


class CostFunction:
    def __init__(self, unit_mesh):
        self.dx = Measure('dx', domain=unit_mesh)

    def J_evaluate(self, phi, t, delta_t):
        pass

    def end_record(self):
        pass

    def start_record(self):
        pass

    def get_name(self):
        return "CostFunction"

    def J_final_evaluate(self, phi, phi_m):
        pass


class ControllabilityCostFunction(CostFunction):
    def __init__(self, unit_mesh, phi_m=None):
        CostFunction.__init__(self, unit_mesh)
        self.dx = Measure('dx', domain=unit_mesh)
        self.phi_m = phi_m

    def J_evaluate(self, phi, t, delta_t):
        pass

    def get_name(self):
        return "ControllabilityCostFunction"

    def J_final_evaluate(self, phi, t):
        if self.phi_m is Expression:
            self.phi_m.t = t
        return assemble(0.5 * ((phi - self.phi_m) ** 2) * self.dx)


class SensorCostFunction():

    def __init__(self, unit_mesh, t_start=0):
        self.J_evals = []
        self.J_time = None
        self.phi_measurments = []
        self.t_m = []
        self.record_mode = True

        self.index = 0
        self.t_start = t_start
        self.t_end = None

        self.unit_mesh = unit_mesh
        self.mesh_func = MeshFunction('size_t', unit_mesh, 1)

        class Sensors(SubDomain):
            def inside(self, x, on_boundary):
                # return near(x[1], 34) and (near(x[0], 5) or near(x[0], 10) or near(x[0], 20)
                #                           or near(x[0], 30) or near(x[0], 40) or near(x[0], 50))
                return near(x[1], 34)

        self.sensors = Sensors()
        self.mesh_func.set_all(0)
        self.sensors.mark(self.mesh_func, 1)

        self.ds_domain = Measure('ds', domain=self.unit_mesh, subdomain_data=self.mesh_func)
        self.ds_sensors = self.ds_domain(1)

    def end_record(self):
        self.record_mode = False

    def start_record(self):
        self.record_mode = True

    def get_name(self):
        return "SensorCostFunction"

    def J_evaluate(self, phi, t, delta_t):
        if self.record_mode:
            self.phi_measurments.append(phi)
            self.t_m.append(t)
        else:
            assert np.isclose(t, self.t_m[self.index])
            phi_m = self.phi_measurments[self.index]

            J_phi = assemble(0.5 * (phi_m - phi) ** 2 * self.ds_sensors)
            J = delta_t * J_phi

            if len(self.t_m)-1 == self.index:
                print("adding stabilization term")
                J += J_phi

            self.J_evals.append(J)
            self.t_end = t

            self.index += 1

    def J_final_evaluate(self, phi, t):
        self.J_time = sum(self.J_evals)

        return self.J_time


class VelSensorCostFunction(SensorCostFunction):

    def __init__(self, phi, unit_mesh, t_start):
        SensorCostFunction.__init__(phi, unit_mesh, t_start)

    def record_measurments(self, phi_m, t):
        SensorCostFunction.record_measurments(self, phi_m, t)

    def J_evaluate(self, phi, t, delta_t):
        assert np.isclose(t, self.t_m[self.index])
        phi_m = self.phi_measurments[self.index]

        sig11, sig22, sig12, u, v = self.phi.split()
        sig11_m, sig22_m, sig12_m, u_m, v_m = phi_m.split()

        vel = as_vector((u, v))
        vel_m = as_vector((u_m, v_m))

        J_vel = assemble(0.5 * (vel - vel_m) * self.dx_sensors)

        self.J_evals.append(delta_t * J_vel)
        self.t_end = t

        self.index += 1

    def J_final_evaluate(self):
        SensorCostFunction.J_final_integrate(self)


# TODO really needed? only if derivative of J/dofs is computed! not really needed for normal setup!
class RecordControls:
    """
        Records the control variables for dolfin-adjoint during forward

    """

    def __init__(self, control_points_u=[]):
        self.control_points_u = control_points_u
        self.controls = []

    def store_control(self, u, t):
        if len(self.control_points_u) > 0 and np.isclose(t, self.control_points_u[0], atol=10e-5):
            self.controls.append(Control(u))
            self.control_points_u = self.control_points_u[1::]
            print(self.control_points_u)
