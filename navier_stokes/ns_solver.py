import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import dolfinx
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import dolfinx.fem.petsc 
import ufl
from ufl import grad, inner, dot, div, dx, derivative
import basix.ufl 
import petsc4py.PETSc as PETSc

class NavierStokesSolver2D:
    def __init__(self, res=32, mu=0.5, Re_max=2.0):
        self.domain = mesh.create_unit_square(MPI.COMM_WORLD, res, res)
        self.Re_max = Re_max
        self.mu = fem.Constant(self.domain, PETSc.ScalarType(mu))
        
        v_el = basix.ufl.element("Lagrange", self.domain.ufl_cell().cellname(), 2, shape=(self.domain.geometry.dim,))
        p_el = basix.ufl.element("Lagrange", self.domain.ufl_cell().cellname(), 1)
        self.W = fem.functionspace(self.domain, basix.ufl.mixed_element([v_el, p_el]))
        self.V, _ = self.W.sub(0).collapse()

        self.domain.topology.create_connectivity(self.domain.topology.dim-1, self.domain.topology.dim)
        facets = mesh.exterior_facet_indices(self.domain.topology)
        u_zero = fem.Function(self.V); u_zero.x.array[:] = 0.0
        dofs = fem.locate_dofs_topological((self.W.sub(0), self.V), self.domain.topology.dim-1, facets)
        self.bc = fem.dirichletbc(u_zero, dofs, self.W.sub(0))

        self.w = fem.Function(self.W)
        self.theta_param = fem.Constant(self.domain, PETSc.ScalarType(0.0))
        u, p = ufl.split(self.w)
        v, q = ufl.TestFunctions(self.W)
        x_c = ufl.SpatialCoordinate(self.domain)
        psi = ufl.sin(ufl.pi * x_c[0]) * ufl.sin(ufl.pi * x_c[1])
        f = self.theta_param * ufl.as_vector((psi.dx(1), -psi.dx(0)))
        
        self.F_ufl = (self.mu * inner(grad(u), grad(v)) * dx + inner(dot(u, grad(u)), v) * dx 
                    - inner(p, div(v)) * dx + inner(q, div(u)) * dx - inner(f, v) * dx)
        self.J_ufl = derivative(self.F_ufl, self.w)
        
        self._L_form = fem.form(self.F_ufl)
        self._a_form = fem.form(self.J_ufl)
        self.A_mat = dolfinx.fem.petsc.create_matrix(self._a_form)
        self.b_vec = self.w.x.petsc_vec.duplicate()

        self.snes = PETSc.SNES().create(self.domain.comm)
        self.snes.setFunction(self._assemble_F, self.b_vec)
        self.snes.setJacobian(self._assemble_J, self.A_mat)
        self.snes.setType("newtonls")
        self.snes.getKSP().setType("preonly")
        self.snes.getKSP().getPC().setType("lu")
        self.snes.getKSP().getPC().setFactorSolverType("mumps")
        self.bb_tree = geometry.bb_tree(self.domain, self.domain.topology.dim)

    def _assemble_F(self, snes, x, b):
        x.copy(self.w.x.petsc_vec); self.w.x.scatter_forward()
        b.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, self._L_form)
        dolfinx.fem.petsc.apply_lifting(b, [self._a_form], [[self.bc]], [x], -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, [self.bc], x, -1.0)

    def _assemble_J(self, snes, x, A, P):
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, self._a_form, bcs=[self.bc])
        A.assemble()

    def solve(self, theta, tol=1e-11):
        self.theta_param.value = theta
        self.w.x.array[:] = 0.0
        self.snes.setTolerances(rtol=tol, atol=tol)
        self.snes.solve(None, self.w.x.petsc_vec)
        return self.w, self.snes.getConvergedReason() > 0

    def get_fast_observations(self, w_mixed, obs_points, cells):
        return w_mixed.sub(0).eval(obs_points, cells).flatten()