"""Step 2 of the monodomain + heart–torso uncoupled pipeline: extracellular recovery.

The heart mesh here is the 3D rabbit ventricle (Zenodo 6340066), but the
weak form and PETSc setup below are dimension-agnostic — the same code path
served the 2D demo in ``legacy_2d/forward/extracellular.py``.

Given the transmembrane potential V_m on the heart,
the extracellular potential u_e satisfies the pure-
Neumann elliptic problem

    − div((σ_i + σ_e) ∇ u_e)  =  div(σ_i ∇ V_m)        in  Ω_H
    (σ_i + σ_e) ∇ u_e · n  = − σ_i ∇ V_m · n           on  Gamma_H.

Weak form (multiply by w ∈ H¹(Ω_H) and integrate by parts on both sides):

    ∫ (σ_i + σ_e) ∇ u_e · ∇ w dΩ  =  − ∫ σ_i ∇ V_m · ∇ w dΩ.

This problem is well-posed only up to an additive constant.
We follow the dolfinx tutorial

https://jsdokken.com/dolfinx-tutorial/chapter2/singular_poisson.html

and attach the constant nullspace to the assembled matrix via PETSc,
then pin the gauge after the solve by subtracting the spatial mean.
"""
from __future__ import annotations

import dolfinx
import dolfinx.fem.petsc
import ufl
from mpi4py import MPI
from petsc4py import PETSc


def recover_extracellular(
    v_m: dolfinx.fem.Function,
    sigma_i: float,
    sigma_e: float,
) -> dolfinx.fem.Function:
    """Solve the pure-Neumann elliptic problem for u_e.

    Args:
        v_m: transmembrane potential, a CG1 Function on the heart mesh.
        sigma_i, sigma_e: bulk intra-/extracellular conductivities
            (scalars in this isotropic setting; the same form
            generalises to UFL tensor coefficients on fibre-aware meshes).

    Returns:
        A new :class:`dolfinx.fem.Function` carrying u_e (with zero mean
        over Ω_H) on the same function space as ``v_m``.
    """
    V = v_m.function_space
    mesh = V.mesh
    comm = mesh.comm

    u = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)

    a = dolfinx.fem.form(
        (sigma_i + sigma_e) * ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
    )
    L = dolfinx.fem.form(
        -sigma_i * ufl.inner(ufl.grad(v_m), ufl.grad(w)) * ufl.dx
    )

    # assemble the linear system Au = b
    A = dolfinx.fem.petsc.assemble_matrix(a)
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(L)
    b.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE,
    ) # ghost DOFs are local copies of DOFs owned by another process
    # ADD_VALUES: when communicating ghost contributions back to the owner,
    # add them to the existing value
    # REVERSE: communicates from ghost copies to owning process

    # Constants span the operator kernel. Attach to A so the Krylov
    # method projects them out each iteration, and orthogonalise b
    # explicitly: ∫b·1 vanishes analytically but rounding leaves a small
    # inconsistent component that would otherwise prevent convergence.
    nullspace = PETSc.NullSpace().create(constant=True, comm=comm) # creates PETSc Nullspace object cont. const.
    assert nullspace.test(A) # checks if A * 1 ~= 0, i.e. if nullspace is correct
    A.setNullSpace(nullspace) # attaches nullspace to A so Krylov methods know it is singular
    A.setNearNullSpace(nullspace) # for preconditioners
    nullspace.remove(b) # removes any component of b in the nullspace direction, i.e. makes ∫b·1 = 0 exactly

    u_e = dolfinx.fem.Function(V, name="u_e") # allocates output function for u_e

    # PETSc Krylov solver
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    # GMRES tolerates HYPRE-AMG coarse spaces that aren't strictly SPD after
    # constant-nullspace projection (which CG flags as "indefinite mat" on
    # the 3D rabbit). Pure-Neumann elliptic is symmetric in continuous form,
    # so a few GMRES iterations behave just like CG in practice.
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.getPC().setType(PETSc.PC.Type.HYPRE) # HYPRE: preconditioner for elliptic problems
    ksp.setTolerances(rtol=1.0e-10)
    ksp.setGMRESRestart(50)
    ksp.setFromOptions() # command-line options can override the above settings
    ksp.solve(b, u_e.x.petsc_vec) # solve Au = b for u, storing result in u_e.x.petsc_vec

    reason = ksp.getConvergedReason() # checks convergence
    if reason < 0:
        raise RuntimeError(
            f"Krylov solver failed to converge: {reason} ({ksp.view()})"
        )

    u_e.x.scatter_forward() # updates ghost DOFs -> consistent across processes

    # Gauge fix: Krylov returns an arbitrary representative in the
    # constant kernel — subtract the spatial mean to get a zero mean extracellular potential.
    # Even when solving with nullspace attached -> u_e is determined up to a constant
    mean = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(u_e * ufl.dx)),
        op=MPI.SUM,
    ) # integral of u_e over heart mesh, summed across processes
    one = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))
    volume = comm.allreduce(
        dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(one * ufl.dx)
        ),
        op=MPI.SUM,
    ) # volume of heart mesh, summed across processes
    u_e.x.array[:] -= mean / volume # enforces zero mean on heart mesh
    u_e.x.scatter_forward()

    # free PETSc objects
    ksp.destroy()
    A.destroy()
    b.destroy()
    nullspace.destroy()
    return u_e
