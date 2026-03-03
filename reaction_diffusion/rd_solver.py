# reaction_diffusion/rd_solver.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import dolfinx
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import dolfinx.fem.petsc 
import ufl
from ufl import grad, inner, dx, derivative, dot
import basix.ufl 
import petsc4py.PETSc as PETSc

class ReactionDiffusionProblem:
    def __init__(self, F_ufl, w, bcs, J_ufl):
        self.L_form = fem.form(F_ufl)
        self.a_form = fem.form(J_ufl)
        self.w = w
        self.bcs = bcs
        self._A = dolfinx.fem.petsc.create_matrix(self.a_form)
        self._b = self.w.x.petsc_vec.duplicate()

    def F(self, snes, x, b):
        x.copy(self.w.x.petsc_vec)
        self.w.x.scatter_forward()
        with b.localForm() as b_local:
            b_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, self.L_form)
        dolfinx.fem.petsc.apply_lifting(b, [self.a_form], [self.bcs], [x], -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, self.bcs, x, -1.0)

    def J(self, snes, x, A, P):
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, self.a_form, bcs=self.bcs)
        A.assemble()

class CoupledRDSolver2D:
    def __init__(self, res=32, D1=1.0, D2=1.0, k=0.5, k_max=2.0):
        self.domain = mesh.create_unit_square(MPI.COMM_WORLD, res, res)
        self.Re_max = k_max
        self.D1 = fem.Constant(self.domain, PETSc.ScalarType(D1))
        self.D2 = fem.Constant(self.domain, PETSc.ScalarType(D2))
        self.k = fem.Constant(self.domain, PETSc.ScalarType(k))

        el = basix.ufl.element("Lagrange", self.domain.ufl_cell().cellname(), 1)
        self.W = fem.functionspace(self.domain, basix.ufl.mixed_element([el, el]))
        self.V0, _ = self.W.sub(0).collapse()
        self.V1, _ = self.W.sub(1).collapse()

        self.domain.topology.create_connectivity(self.domain.topology.dim-1, self.domain.topology.dim)
        facets = mesh.exterior_facet_indices(self.domain.topology)
        u_zero = fem.Function(self.V0); u_zero.x.array[:] = 0.0
        dofs0 = fem.locate_dofs_topological((self.W.sub(0), self.V0), self.domain.topology.dim-1, facets)
        dofs1 = fem.locate_dofs_topological((self.W.sub(1), self.V1), self.domain.topology.dim-1, facets)
        self.bcs = [fem.dirichletbc(u_zero, dofs0, self.W.sub(0)), fem.dirichletbc(u_zero, dofs1, self.W.sub(1))]

        # Persistent Objects
        self.w = fem.Function(self.W)
        self.theta1_param = fem.Constant(self.domain, PETSc.ScalarType(0.0))
        self.theta2_param = fem.Constant(self.domain, PETSc.ScalarType(0.0))
        
        u, v = ufl.split(self.w)
        phi, psi = ufl.TestFunctions(self.W)
        x_c = ufl.SpatialCoordinate(self.domain)
        
        f1 = self.theta1_param * ufl.sin(ufl.pi * x_c[0]) * ufl.sin(ufl.pi * x_c[1])
        f2 = self.theta2_param * ufl.sin(ufl.pi * x_c[0]) * ufl.cos(ufl.pi * x_c[1] * 0.5)
        
        self.F_ufl = (self.D1 * inner(grad(u), grad(phi)) * dx + self.k * u * v * phi * dx - f1 * phi * dx
                    + self.D2 * inner(grad(v), grad(psi)) * dx + self.k * u * v * psi * dx - f2 * psi * dx)
        self.J_ufl = derivative(self.F_ufl, self.w)
        
        self.problem = ReactionDiffusionProblem(self.F_ufl, self.w, self.bcs, self.J_ufl)
        self.snes = PETSc.SNES().create(self.domain.comm)
        self.snes.setFunction(self.problem.F, self.problem._b)
        self.snes.setJacobian(self.problem.J, self.problem._A)
        self.snes.setType("newtonls")
        self.snes.getKSP().setType("preonly")
        self.snes.getKSP().getPC().setType("lu")
        self.snes.getKSP().getPC().setFactorSolverType("mumps")

        self.bb_tree = geometry.bb_tree(self.domain, self.domain.topology.dim)

    def solve(self, theta1, theta2, tol=1e-10):
        self.theta1_param.value = theta1
        self.theta2_param.value = theta2
        self.w.x.array[:] = 0.0
        self.snes.setTolerances(rtol=tol, atol=tol)
        self.snes.solve(None, self.w.x.petsc_vec)
        self.w.x.scatter_forward()
        return self.w, self.snes.getConvergedReason() > 0

    def get_fast_observations(self, w_mixed, obs_points, cells):
        """Evaluation using pre-calculated cells for N-scaling speed."""
        u = w_mixed.sub(0)
        v = w_mixed.sub(1)
        results = []
        for i, point in enumerate(obs_points):
            # Evaluate both species components
            u_val = u.eval(point.reshape(1,3), [cells[i]])
            v_val = v.eval(point.reshape(1,3), [cells[i]])
            results.extend([u_val[0], v_val[0]])
        return np.array(results)