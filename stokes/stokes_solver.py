import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import dolfinx
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import dolfinx.fem.petsc 
import ufl
from ufl import grad, inner, div, dx, derivative
import basix.ufl 
import petsc4py.PETSc as PETSc

class StokesProxyProblem:
    def __init__(self, F_ufl, w, bcs, J_ufl):
        self.L_form = fem.form(F_ufl)
        self.a_form = fem.form(J_ufl)
        self.w = w
        self.bcs = bcs
        
    def assemble_residual(self, snes, x, b):
        x.copy(self.w.x.petsc_vec)
        self.w.x.scatter_forward()
        with b.localForm() as b_local:
            b_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, self.L_form)
        dolfinx.fem.petsc.apply_lifting(b, [self.a_form], [self.bcs], [x], -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, self.bcs, x, -1.0)

    def assemble_jacobian(self, snes, x, A, P):
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, self.a_form, bcs=self.bcs)
        A.assemble()

class StokesSolver2D:
    def __init__(self, res=32, mu=0.1):
        self.domain = mesh.create_unit_square(MPI.COMM_WORLD, res, res)
        self.mu = fem.Constant(self.domain, PETSc.ScalarType(mu))
        self.Re_max = 5.0 
        
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
        
        self.F_ufl = (self.mu * inner(grad(u), grad(v)) - div(v)*p + q*div(u) - inner(f, v)) * dx
        self.J_ufl = derivative(self.F_ufl, self.w)
        
        self.problem = StokesProxyProblem(self.F_ufl, self.w, [self.bc], self.J_ufl)
        self.A_mat = dolfinx.fem.petsc.create_matrix(self.problem.a_form)
        self.b_vec = self.w.x.petsc_vec.duplicate()
        
        self.snes = PETSc.SNES().create(self.domain.comm)
        self.snes.setFunction(self.problem.assemble_residual, self.b_vec)
        self.snes.setJacobian(self.problem.assemble_jacobian, self.A_mat)
        self.snes.setType("newtonls")
        self.snes.getKSP().setType("preonly")
        self.snes.getKSP().getPC().setType("lu")
        self.snes.getKSP().getPC().setFactorSolverType("mumps")
        
        self.bb_tree = geometry.bb_tree(self.domain, self.domain.topology.dim)

    def solve(self, theta, tol=1e-12):
        self.theta_param.value = theta
        self.w.x.array[:] = 0.0
        self.snes.setTolerances(rtol=tol, atol=tol)
        self.snes.solve(None, self.w.x.petsc_vec)
        self.w.x.scatter_forward()
        
        # FIX: Interpolate to standalone velocity function for safe evaluation
        u_out = fem.Function(self.V)
        u_out.interpolate(self.w.sub(0))
        return u_out

    def get_fast_observations(self, u_standalone, obs_points, cells):
        # eval on collapsed function is robust in v0.9.0
        return u_standalone.eval(obs_points, cells).flatten()