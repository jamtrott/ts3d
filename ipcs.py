# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

import argparse

from dolfinx import fem, io, mesh as dmesh
import dolfinx.fem.petsc as petsc_fem
import numpy as np
import ufl
import os
from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD


def compute_l2_time_err(dt: np.float64, errors: np.ndarray):
    return np.sqrt(dt * sum(errors))


def compute_eoc(errors: np.ndarray):
    return np.log(errors[:-1] / errors[1:]) / np.log(2)


def IPCS(r_lvl: int, t_lvl: int, outdir: str, degree_u=2,
         jit_options: dict = {"cffi_extra_compile_args": ["-Ofast", "-march=native"], "cffi_libraries": ["m"]}):
    # Define mesh and function spaces
    N = 25 * 2**r_lvl
    mesh = dmesh.create_rectangle(comm, [np.array([-1.0, -1.0]), np.array([1.0, 1.0])], [N, N], dmesh.CellType.triangle)
    celldim = mesh.topology.dim
    facetdim = celldim - 1
    degree_p = degree_u - 1
    error_raise = 3
    V = fem.VectorFunctionSpace(mesh, ("CG", degree_u))
    Q = fem.FunctionSpace(mesh, ("CG", degree_p))

    # Temporal parameters
    t = 0
    dt = PETSc.ScalarType(0.1 * 0.5**t_lvl)
    T = 1

    # Physical parameters
    nu = 0.01
    f = fem.Constant(mesh, PETSc.ScalarType((0, 0)))

    # Define functions for the variational form
    uh = fem.Function(V)
    uh.name = "Velocity"
    u_tent = fem.Function(V)
    u_tent.name = "Tentative_velocity"
    u_old = fem.Function(V)
    ph = fem.Function(Q)
    ph.name = "Pressure"
    phi = fem.Function(Q)
    phi.name = "Phi"

    def u_ex(t, nu):
        """
        Wrapper to generate a function for interpolating given any pair of t and nu
        for the analytical expression of u
        """
        return lambda x: np.row_stack((
            -np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
            * np.exp(-2.0 * nu * np.pi**2 * t),
            np.cos(np.pi * x[1]) * np.sin(np.pi * x[0])
            * np.exp(-2.0 * nu * np.pi**2 * t)))

    def p_ex(t, nu):
        """
        Wrapper to generate a function for interpolating given any pair of t and nu
        for the analytical expression of
        """
        return lambda x: -0.25 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))\
            * np.exp(-4.0 * nu * np.pi**2 * t)

    # Interpolate initial guesses
    uh.interpolate(u_ex(t, nu))
    u_old.interpolate(u_ex(t - dt, nu))
    ph.interpolate(p_ex(t, nu))

    # Define variational forms
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)

    # ----Step 1: Tentative velocity step----
    w_time = fem.Constant(mesh, 3 / (2 * dt))
    w_diffusion = fem.Constant(mesh, PETSc.ScalarType(nu))
    a_tent = (w_time * ufl.inner(u, v) + w_diffusion
              * ufl.inner(ufl.grad(u), ufl.grad(v))) * dx
    L_tent = (ufl.inner(ph, ufl.div(v)) + ufl.inner(f, v)) * dx
    L_tent += fem.Constant(mesh, 1 / (2 * dt)) *\
        ufl.inner(4 * uh - u_old, v) * dx
    # BDF2 with implicit Adams-Bashforth
    bs = 2 * uh - u_old
    a_tent += ufl.inner(ufl.grad(u) * bs, v) * dx

    # Temam-device
    a_tent += 0.5 * ufl.div(bs) * ufl.inner(u, v) * dx

    # Find boundary facets and create boundary condition
    bndry_facets = dmesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofsV = fem.locate_dofs_topological(V, facetdim, bndry_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(u_ex(t + dt, nu))
    bcs_tent = [fem.dirichletbc(u_bc, bdofsV)]

    # Compile forms and assemble forms
    a_tent = fem.form(a_tent, jit_options=jit_options)
    A_tent = petsc_fem.assemble_matrix(a_tent, bcs=bcs_tent)
    A_tent.assemble()
    L_tent = fem.form(L_tent, jit_options=jit_options)
    b_tent = petsc_fem.assemble_vector(L_tent)
    b_tent.assemble()

    # ----Step 2: Pressure correction step----
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)
    a_corr = ufl.inner(ufl.grad(p), ufl.grad(q)) * dx
    L_corr = - w_time * ufl.inner(ufl.div(u_tent), q) * dx
    nullspace = PETSc.NullSpace().create(constant=True)

    # Compile forms and assemble forms
    a_corr = fem.form(a_corr, jit_options=jit_options)
    A_corr = petsc_fem.assemble_matrix(a_corr)
    A_corr.setNullSpace(nullspace)
    A_corr.assemble()
    L_corr = fem.form(L_corr, jit_options=jit_options)
    b_corr = petsc_fem.assemble_vector(L_corr)
    b_corr.assemble()

    # ----Step 3: Velocity update----
    a_up = ufl.inner(u, v) * dx
    L_up = (ufl.inner(u_tent, v) - w_time**(-1) * ufl.inner(ufl.grad(phi), v)) * dx
    a_up = fem.form(a_up, jit_options=jit_options)
    A_up = petsc_fem.assemble_matrix(a_up)
    A_up.assemble()

    L_up = fem.form(L_up, jit_options=jit_options)
    b_up = petsc_fem.assemble_vector(L_up)
    b_up.assemble()

    # Setup solvers
    solver_tent = PETSc.KSP().create(comm)
    solver_tent.setType("preonly")
    solver_tent.setTolerances(rtol=1.0e-14)
    solver_tent.getPC().setType("lu")
    solver_tent.getPC().setFactorSolverType("mumps")
    solver_tent.setOperators(A_tent)

    solver_corr = PETSc.KSP().create(comm)
    solver_corr.setType("preonly")
    solver_corr.setTolerances(rtol=1.0e-14)
    solver_corr.getPC().setType("lu")
    solver_corr.getPC().setFactorSolverType("mumps")
    solver_corr.setOperators(A_corr)

    solver_up = PETSc.KSP().create(comm)
    solver_up.setType("preonly")
    solver_up.setTolerances(rtol=1.0e-14)
    solver_up.getPC().setType("lu")
    solver_up.getPC().setFactorSolverType("mumps")
    solver_up.setOperators(A_up)

    # Create spaces for error approximation
    V_err = fem.VectorFunctionSpace(mesh, ("CG", degree_u + error_raise))
    Q_err = fem.FunctionSpace(mesh, ("CG", degree_p + error_raise))
    u_err = fem.Function(V_err)
    p_err = fem.Function(Q_err)

    # Create file for output
    outfile = io.XDMFFile(comm, f"{outdir}/output.xdmf", "w")
    outfile.write_mesh(mesh)

    # Solve problem
    l2_u = np.zeros(int(T / dt), dtype=np.float64)
    l2_p = np.zeros(int(T / dt), dtype=np.float64)
    vol_form = fem.form(fem.Constant(mesh, PETSc.ScalarType(1)) * dx, jit_options=jit_options)
    vol = mesh.comm.allreduce(fem.assemble_scalar(vol_form), op=MPI.SUM)

    # Form for normalizing phi due to lack of DirichletBCs
    norm_form = fem.form(phi * dx)

    # Setup error forms
    error_u_L2 = fem.form(ufl.inner(uh - u_err, uh - u_err) * ufl.dx,
                          jit_options=jit_options)
    error_p_L2 = fem.form(ufl.inner(ph - p_err, ph - p_err) * ufl.dx,
                          jit_options=jit_options)

    i = 0
    outfile.write_function(uh, t)
    outfile.write_function(ph, t)
    while (t <= T - 1e-3):
        t += dt
        # Update BC and exact solutions
        u_bc.interpolate(u_ex(t, nu))
        u_err.interpolate(u_ex(t, nu))
        p_err.interpolate(p_ex(t, nu))

        # Solve step 1
        A_tent.zeroEntries()
        petsc_fem.assemble_matrix(A_tent, a_tent, bcs=bcs_tent)  # type: ignore
        A_tent.assemble()
        with b_tent.localForm() as b_local:
            b_local.set(0.0)
        petsc_fem.assemble_vector(b_tent, L_tent)
        petsc_fem.apply_lifting(b_tent, [a_tent], [bcs_tent])
        b_tent.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc_fem.set_bc(b_tent, bcs_tent)
        solver_tent.solve(b_tent, u_tent.vector)
        u_tent.x.scatter_forward()

        # Solve step 2
        with b_corr.localForm() as b_local:
            b_local.set(0.0)
        petsc_fem.assemble_vector(b_corr, L_corr)
        b_corr.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b_corr.assemble()
        solver_corr.solve(b_corr, phi.vector)
        phi.x.scatter_forward()

        # Normalize pressure correction
        phi_avg = mesh.comm.allreduce(fem.assemble_scalar(norm_form) / vol, op=MPI.SUM)
        avg_vec = phi.vector.copy()
        with avg_vec.localForm() as avg_local:
            avg_local.set(-phi_avg)
        phi.vector.axpy(1.0, avg_vec)
        phi.x.scatter_forward()

        # Update p and previous u
        ph.vector.axpy(1.0, phi.vector)
        ph.x.scatter_forward()
        uh.vector.copy(result=u_old.vector)
        u_old.x.scatter_forward()

        # Solve step 3
        with b_up.localForm() as b_local:
            b_local.set(0.0)
        petsc_fem.assemble_vector(b_up, L_up)
        b_up.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        solver_up.solve(b_up, uh.vector)
        uh.x.scatter_forward()

        # Compute L2 error norms
        uL2 = mesh.comm.allreduce(fem.assemble_scalar(error_u_L2), op=MPI.SUM)
        pL2 = mesh.comm.allreduce(fem.assemble_scalar(error_p_L2), op=MPI.SUM)
        l2_u[i] = uL2
        l2_p[i] = pL2

        i += 1
        outfile.write_function(phi, t)
        outfile.write_function(u_tent, t)
        outfile.write_function(uh, t)
        outfile.write_function(ph, t)
    outfile.close()
    L2L2u = compute_l2_time_err(dt, l2_u)
    L2L2p = compute_l2_time_err(dt, l2_p)
    b_tent.destroy()
    b_corr.destroy()
    b_up.destroy()
    return L2L2u, L2L2p


if __name__ == "__main__":
    desc = "Script to run convergence study for a manufactured solution for the Navier-Stokes equations" +\
        " using the IPCS splitting scheme."
    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--spatial", default=5, type=int, dest="R_ref",
                        help="Number of spatial refinements")
    parser.add_argument("--temporal", default=5, type=int, dest="T_ref",
                        help="Number of temporal refinements")
    parser.add_argument("--degree-u", default=2, type=int, dest="degree",
                        help="Degree of velocity space")
    parser.add_argument("--outdir", default="results", type=str, dest="outdir",
                        help="Name of output folder")
    args = parser.parse_args()
    R_ref = args.R_ref
    T_ref = args.T_ref
    errors_u = np.zeros((R_ref, T_ref), dtype=np.float64)
    errors_p = np.zeros((R_ref, T_ref), dtype=np.float64)
    os.system(f"mkdir -p {args.outdir}")
    for i in range(R_ref):
        for j in range(T_ref):
            errors_u[i, j], errors_p[i, j] = IPCS(i, j, args.outdir, degree_u=args.degree)
            print(f"{i}, {j}, {errors_u[i, j]}, {errors_p[i, j]}")

    print(f"Temporal eoc u {compute_eoc(errors_u[-1, :])}")
    print(f"Spatial eoc u {compute_eoc(errors_u[:, -1])}")
    print(f"Temporal eoc p {compute_eoc(errors_p[-1, :])}")
    print(f"Spatial eoc p {compute_eoc(errors_p[:, -1])}")
