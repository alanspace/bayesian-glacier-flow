"""
FEM Solver Wrapper for Data Generation

Wraps the existing FEniCSx glacier flow solver for batch data generation.
Provides a clean interface for running multiple simulations with different parameters.

Author: Shek Lun Leung
Date: January 2026
"""

import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

# Add existing FEM code to path
# Add existing FEM code to path
# Path(__file__).parents[3] is numerical_analysis
fem_path = Path(__file__).parents[3] / "picard-fem-glacier-flow" / "src"
sys.path.insert(0, str(fem_path))

try:
    import helpfunctions
    from dolfinx import fem, io
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    from petsc4py import PETSc
    import ufl
    import basix.ufl
    FEM_AVAILABLE = True
except ImportError:
    FEM_AVAILABLE = False
    print("Warning: FEniCSx not available. FEM solver will not work.")


class GlacierFEMSolver:
    """
    Wrapper around FEniCSx implementation of glacier flow equations.
    
    Solves the First-Order Stokes approximation with Glen's law rheology.
    """
    
    def __init__(self, mesh_path: Optional[str] = None):
        """
        Args:
            mesh_path: Path to glacier mesh (XDMF format)
        """
        if not FEM_AVAILABLE:
            raise ImportError("FEniCSx required. Install via: conda install -c conda-forge fenics-dolfinx")
        
        self.mesh_path = mesh_path
        self.mesh = None
        self.V = None  # Function space
        self.facet_tags = None
        
        if mesh_path and os.path.exists(mesh_path):
            self._load_mesh(mesh_path)
    
    def _load_mesh(self, mesh_path: str):
        """Load glacier mesh from file."""
        self.mesh = helpfunctions.loadmesh(mesh_path)
        
        # Create function space (P1 elements)
        element = basix.ufl.element("Lagrange", "triangle", 1)
        self.V = fem.functionspace(self.mesh, element)
        
        # Mark boundaries
        self.facet_tags = helpfunctions.mark_bed_surface(
            self.mesh, bed_id=2, surface_id=1
        )
    
    def solve_linear(
        self,
        eta: float = 1e13,
        rho: float = 910.0,
        g: float = 9.81
    ) -> Tuple[np.ndarray, float]:
        """
        Solve linear problem with constant viscosity.
        
        Args:
            eta: Constant viscosity (Pa·s)
            rho: Ice density (kg/m³)
            g: Gravitational acceleration (m/s²)
            
        Returns:
            velocity: Velocity values at DOFs
            solve_time: Wall-clock time in seconds
        """
        start_time = time.time()
        
        # Boundary conditions
        fdim = self.mesh.topology.dim - 1
        bed_facets = self.facet_tags.find(2)
        bed_dofs = fem.locate_dofs_topological(self.V, fdim, bed_facets)
        bcs = [fem.dirichletbc(PETSc.ScalarType(0), bed_dofs, self.V)]
        
        # Surface elevation
        h_fun = helpfunctions.get_h(self.V, self.facet_tags, surface_id=1)
        
        # Variational formulation
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        
        a = (2 * eta * ufl.Dx(u, 0) * ufl.Dx(v, 0) + 
             0.5 * eta * ufl.Dx(u, 1) * ufl.Dx(v, 1)) * ufl.dx
        
        L = -rho * g * ufl.Dx(h_fun, 0) * v * ufl.dx
        
        # Solve
        problem = LinearProblem(
            a, L, bcs=bcs,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix='glacier_linear_'
        )
        uh = problem.solve()
        
        solve_time = time.time() - start_time
        
        return uh.x.array.copy(), solve_time
    
    def solve_nonlinear(
        self,
        A: float = 1e-16,
        epsilon: float = 1e-10,
        rho: float = 910.0,
        g: float = 9.81,
        max_iter: int = 50,
        tol: float = 1e-6
    ) -> Tuple[np.ndarray, int, float]:
        """
        Solve nonlinear problem with Glen's law using Picard iteration.
        
        Args:
            A: Flow parameter (Pa^-3 yr^-1)
            epsilon: Regularization parameter
            rho: Ice density  
            g: Gravity
            max_iter: Maximum Picard iterations
            tol: Convergence tolerance
            
        Returns:
            velocity: Velocity values at DOFs
            iterations: Number of iterations to convergence
            solve_time: Total wall-clock time
        """
        start_time = time.time()
        
        # Convert A to SI units
        sec_per_yr = 31557600.0
        A_sec = A / sec_per_yr
        
        # Boundary conditions
        fdim = self.mesh.topology.dim - 1
        bed_facets = self.facet_tags.find(2)
        bed_dofs = fem.locate_dofs_topological(self.V, fdim, bed_facets)
        bcs = [fem.dirichletbc(PETSc.ScalarType(0), bed_dofs, self.V)]
        
        # Surface elevation
        h_fun = helpfunctions.get_h(self.V, self.facet_tags, surface_id=1)
        
        # Picard iteration
        u_k = fem.Function(self.V)
        u_k.x.array[:] = 0.0
        
        for iteration in range(max_iter):
            # Compute viscosity from current velocity
            ux = ufl.Dx(u_k, 0)
            uz = ufl.Dx(u_k, 1)
            strain_term = ux**2 + uz**2 + epsilon
            eta_k = 0.25 * (A_sec**(-1/3)) * (strain_term**(-1.0/3.0))
            
            # Solve linear problem with fixed viscosity
            u = ufl.TrialFunction(self.V)
            v = ufl.TestFunction(self.V)
            
            a = (2 * eta_k * ufl.Dx(u, 0) * ufl.Dx(v, 0) + 
                 0.5 * eta_k * ufl.Dx(u, 1) * ufl.Dx(v, 1)) * ufl.dx
            
            L = -rho * g * ufl.Dx(h_fun, 0) * v * ufl.dx
            
            problem = LinearProblem(
                a, L, bcs=bcs,
                petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                petsc_options_prefix='glacier_nonlinear_'
            )
            u_next = problem.solve()
            
            # Check convergence
            diff = u_next.x.array - u_k.x.array
            norm_diff = np.linalg.norm(diff)
            norm_u = np.linalg.norm(u_next.x.array)
            
            rel_error = norm_diff / norm_u if norm_u > 1e-10 else norm_diff
            
            # Update
            u_k.x.array[:] = u_next.x.array[:]
            
            if rel_error < tol:
                solve_time = time.time() - start_time
                return u_k.x.array.copy(), iteration + 1, solve_time
        
        # Max iterations reached
        solve_time = time.time() - start_time
        return u_k.x.array.copy(), max_iter, solve_time
    
    def get_dof_coordinates(self) -> np.ndarray:
        """Get spatial coordinates of degrees of freedom."""
        return self.V.tabulate_dof_coordinates()


if __name__ == "__main__":
    print("FEM Solver Wrapper Demo\n" + "="*50)
    
    # Path to existing Arolla mesh
    # Path to existing Arolla mesh
    mesh_path = str(
        Path(__file__).parents[3] / 
        "picard-fem-glacier-flow" / "data" / "arolla.xdmf"
    )
    
    if os.path.exists(mesh_path):
        print(f"Loading mesh from: {mesh_path}")
        
        solver = GlacierFEMSolver(mesh_path)
        
        # Test linear solve
        print("\n1. Testing linear solver...")
        velocity, time_linear = solver.solve_linear(eta=1e13)
        print(f"   ✓ Solved in {time_linear:.3f}s")
        print(f"   Velocity range: [{velocity.min():.2e}, {velocity.max():.2e}] m/s")
        
        # Test nonlinear solve
        print("\n2. Testing nonlinear solver...")
        velocity_nl, iters, time_nl = solver.solve_nonlinear(
            A=1e-16, epsilon=1e-10
        )
        print(f"   ✓ Converged in {iters} iterations ({time_nl:.3f}s)")
        print(f"   Velocity range: [{velocity_nl.min():.2e}, {velocity_nl.max():.2e}] m/s")
        
        # Get DOF coordinates
        coords = solver.get_dof_coordinates()
        print(f"\n3. Mesh info:")
        print(f"   DOFs: {len(coords)}")
        print(f"   Domain: x ∈ [{coords[:,0].min():.0f}, {coords[:,0].max():.0f}] m")
        print(f"           z ∈ [{coords[:,1].min():.0f}, {coords[:,1].max():.0f}] m")
        
        print("\n✅ FEM solver wrapper ready!")
    else:
        print(f"❌ Mesh not found at: {mesh_path}")
        print("   Run from picard-fem-glacier-flow directory or adjust path")
