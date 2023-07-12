import argparse
import os
import sys

import numpy as np
import ufl
from dolfinx import common, fem, io, la, log
from mpi4py import MPI
from petsc4py import PETSc

from create_and_convert_2D_mesh import markers

PROGRAM_VERSION = '1.0.0'

has_tqdm = True
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    has_tqdm = False
    if MPI.COMM_WORLD.rank == 0:
        print("To view progress with progressbar please install tqdm: `pip3 install tqdm`")

def IPCS(comm, xdmffile: str, xdmffacetfile: str, dim: int, degree_u: int,
         out_u, out_p, verbose,
         jit_options: dict = {"cffi_extra_compile_args": ["-Ofast", "-march=native"], "cffi_libraries": ["m"]}):
    assert degree_u >= 2

    if verbose:
        PETSc.Options()["tent_ksp_view"] = ""
        PETSc.Options()["corr_ksp_view"] = ""
        PETSc.Options()["up_ksp_view"] = ""

    # Read in mesh
    with io.XDMFFile(comm, xdmffile, "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")
        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(tdim, tdim)
        mesh.topology.create_connectivity(fdim, tdim)

    with io.XDMFFile(comm, xdmffacetfile, "r") as xdmf:
        mt = xdmf.read_meshtags(mesh, "Facet tags")

    mesh.topology.create_entities(2)
    mesh.topology.create_entities(1)
    if verbose > 0:
        num_cells = np.asarray([mesh.topology.index_map(3).size_local], dtype='l')
        total_cells = np.array(0, dtype='l')
        comm.Reduce([num_cells, MPI.LONG], [total_cells, MPI.LONG], op=MPI.SUM, root=0)
        min_cells = np.array(0, dtype='l')
        comm.Reduce([num_cells, MPI.LONG], [min_cells, MPI.LONG], op=MPI.MIN, root=0)
        max_cells = np.array(0, dtype='l')
        comm.Reduce([num_cells, MPI.LONG], [max_cells, MPI.LONG], op=MPI.MAX, root=0)
        num_faces = np.asarray([mesh.topology.index_map(2).size_local], dtype='l')
        total_faces = np.array(0, dtype='l')
        comm.Reduce([num_faces, MPI.LONG], [total_faces, MPI.LONG], op=MPI.SUM, root=0)
        min_faces = np.array(0, dtype='l')
        comm.Reduce([num_faces, MPI.LONG], [min_faces, MPI.LONG], op=MPI.MIN, root=0)
        max_faces = np.array(0, dtype='l')
        comm.Reduce([num_faces, MPI.LONG], [max_faces, MPI.LONG], op=MPI.MAX, root=0)
        num_edges = np.asarray([mesh.topology.index_map(1).size_local], dtype='l')
        total_edges = np.array(0, dtype='l')
        comm.Reduce([num_edges, MPI.LONG], [total_edges, MPI.LONG], op=MPI.SUM, root=0)
        min_edges = np.array(0, dtype='l')
        comm.Reduce([num_edges, MPI.LONG], [min_edges, MPI.LONG], op=MPI.MIN, root=0)
        max_edges = np.array(0, dtype='l')
        comm.Reduce([num_edges, MPI.LONG], [max_edges, MPI.LONG], op=MPI.MAX, root=0)
        num_vertices = np.asarray([mesh.topology.index_map(0).size_local], dtype='l')
        total_vertices = np.array(0, dtype='l')
        comm.Reduce([num_vertices, MPI.LONG], [total_vertices, MPI.LONG], op=MPI.SUM, root=0)
        min_vertices = np.array(0, dtype='l')
        comm.Reduce([num_vertices, MPI.LONG], [min_vertices, MPI.LONG], op=MPI.MIN, root=0)
        max_vertices = np.array(0, dtype='l')
        comm.Reduce([num_vertices, MPI.LONG], [max_vertices, MPI.LONG], op=MPI.MAX, root=0)
        if comm.rank == 0:
            print("mesh cells: %ld - per process: %ld to %ld"
                  % (total_cells, min_cells, max_cells),
                  file=sys.stderr, flush=True)
            print("mesh faces: %ld - per process: %ld to %ld"
                  % (total_faces, min_faces, max_faces),
                  file=sys.stderr, flush=True)
            print("mesh edges: %ld - per process: %ld to %ld"
                  % (total_edges, min_edges, max_edges),
                  file=sys.stderr, flush=True)
            print("mesh vertices: %ld - per process: %ld to %ld"
                  % (total_vertices, min_vertices, max_vertices),
                  file=sys.stderr, flush=True)

    # Create output files
    if out_u:
        out_u.write_mesh(mesh)
    if out_p:
        out_p.write_mesh(mesh)

    # Define function spaces
    V = fem.VectorFunctionSpace(mesh, ("CG", degree_u))
    Q = fem.FunctionSpace(mesh, ("CG", degree_u - 1))

    if verbose > 0:
        num_Qdofs = np.asarray([Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs], dtype='l')
        total_Qdofs = np.array(0, dtype='l')
        comm.Reduce([num_Qdofs, MPI.LONG], [total_Qdofs, MPI.LONG], op=MPI.SUM, root=0)
        min_Qdofs = np.array(0, dtype='l')
        comm.Reduce([num_Qdofs, MPI.LONG], [min_Qdofs, MPI.LONG], op=MPI.MIN, root=0)
        max_Qdofs = np.array(0, dtype='l')
        comm.Reduce([num_Qdofs, MPI.LONG], [max_Qdofs, MPI.LONG], op=MPI.MAX, root=0)
        num_Vdofs = np.asarray([V.dofmap.index_map.size_local * V.dofmap.index_map_bs], dtype='l')
        total_Vdofs = np.array(0, dtype='l')
        comm.Reduce([num_Vdofs, MPI.LONG], [total_Vdofs, MPI.LONG], op=MPI.SUM, root=0)
        min_Vdofs = np.array(0, dtype='l')
        comm.Reduce([num_Vdofs, MPI.LONG], [min_Vdofs, MPI.LONG], op=MPI.MIN, root=0)
        max_Vdofs = np.array(0, dtype='l')
        comm.Reduce([num_Vdofs, MPI.LONG], [max_Vdofs, MPI.LONG], op=MPI.MAX, root=0)
        if comm.rank == 0:
            print("pressure degrees-of-freedom: %ld - per process: %ld to %ld"
                  % (total_Qdofs, min_Qdofs, max_Qdofs),
                  file=sys.stderr, flush=True)
            print("velocity degrees-of-freedom: %ld - per process: %ld to %ld"
                  % (total_Vdofs, min_Vdofs, max_Vdofs),
                  file=sys.stderr, flush=True)

    # Temporal parameters
    t = 0
    dt = PETSc.ScalarType(1e-2)
    T = 0.05

    # Physical parameters
    nu = 0.001
    f = fem.Constant(mesh, PETSc.ScalarType((0,) * mesh.geometry.dim))
    H = 0.41
    Um = 2.25

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

    # Define variational forms
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)

    # ----Step 1: Tentative velocity step----
    w_time = fem.Constant(mesh, 3 / (2 * dt))
    w_diffusion = fem.Constant(mesh, PETSc.ScalarType(nu))
    a_tent = w_time * ufl.inner(u, v) * dx + w_diffusion * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L_tent = (ufl.inner(ph, ufl.div(v)) + ufl.inner(f, v)) * dx
    L_tent += fem.Constant(mesh, 1 / (2 * dt)) * ufl.inner(4 * uh - u_old, v) * dx
    # BDF2 with implicit Adams-Bashforth
    bs = 2 * uh - u_old
    a_tent += ufl.inner(ufl.grad(u) * bs, v) * dx
    # Temam-device
    a_tent += 0.5 * ufl.div(bs) * ufl.inner(u, v) * dx

    # Find boundary facets and create boundary condition
    inlet_facets = mt.indices[mt.values == markers["Inlet"]]
    inlet_dofs = fem.locate_dofs_topological(V, fdim, inlet_facets)
    wall_facets = mt.indices[mt.values == markers["Walls"]]
    wall_dofs = fem.locate_dofs_topological(V, fdim, wall_facets)
    obstacle_facets = mt.indices[mt.values == markers["Obstacle"]]
    obstacle_dofs = fem.locate_dofs_topological(V, fdim, obstacle_facets)

    def inlet_velocity(t):
        if mesh.geometry.dim == 3:
            return lambda x: ((16 * np.sin(np.pi * t / T) * Um * x[1] * x[2] * (H - x[1]) * (H - x[2]) / (H**4),
                               np.zeros(x.shape[1]), np.zeros(x.shape[1])))
        elif mesh.geometry.dim == 2:
            U = 1.5 * np.sin(np.pi * t / T)
            return lambda x: np.row_stack((4 * U * x[1] * (0.41 - x[1]) / (0.41**2), np.zeros(x.shape[1])))

    u_inlet = fem.Function(V)
    u_inlet.interpolate(inlet_velocity(t))
    zero = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bcs_tent = [fem.dirichletbc(u_inlet, inlet_dofs), fem.dirichletbc(
        zero, wall_dofs, V), fem.dirichletbc(zero, obstacle_dofs, V)]
    a_tent = fem.form(a_tent)# , jit_options=jit_options)
    A_tent = fem.petsc.assemble_matrix(a_tent, bcs=bcs_tent)
    A_tent.assemble()
    L_tent = fem.form(L_tent) #, jit_options=jit_options)
    b_tent = fem.Function(V)

    # Step 2: Pressure correction step
    outlet_facets = mt.indices[mt.values == markers["Outlet"]]
    outlet_dofs = fem.locate_dofs_topological(Q, fdim, outlet_facets)
    bcs_corr = [fem.dirichletbc(PETSc.ScalarType(0), outlet_dofs, Q)]
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)
    a_corr = ufl.inner(ufl.grad(p), ufl.grad(q)) * dx
    L_corr = - w_time * ufl.inner(ufl.div(u_tent), q) * dx
    a_corr = fem.form(a_corr) #, jit_options=jit_options)
    A_corr = fem.petsc.assemble_matrix(a_corr, bcs=bcs_corr)
    A_corr.assemble()

    b_corr = fem.Function(Q)
    L_corr = fem.form(L_corr) #, jit_options=jit_options)

    # Step 3: Velocity update
    a_up = fem.form(ufl.inner(u, v) * dx) #, jit_options=jit_options)
    L_up = fem.form((ufl.inner(u_tent, v) - w_time**(-1) * ufl.inner(ufl.grad(phi), v)) * dx) #, jit_options=jit_options)
    A_up = fem.petsc.assemble_matrix(a_up)
    A_up.assemble()
    b_up = fem.Function(V)

    # Setup solvers
    rtol = 1e-8
    atol = 1e-8
    solver_tent = PETSc.KSP().create(comm)
    solver_tent.setOperators(A_tent)
    solver_tent.setTolerances(rtol=rtol, atol=atol)
    solver_tent.rtol = rtol
    solver_tent.setType("bcgs")
    solver_tent.getPC().setType("jacobi")
    solver_tent.setOptionsPrefix("tent_")
    solver_tent.setFromOptions()
    # solver_tent.setType("preonly")
    # solver_tent.getPC().setType("lu")
    # solver_tent.getPC().setFactorSolverType("mumps")

    solver_corr = PETSc.KSP().create(comm)
    solver_corr.setOperators(A_corr)
    solver_corr.setTolerances(rtol=rtol, atol=atol)
    # solver_corr.setType("preonly")
    # solver_corr.getPC().setType("lu")
    # solver_corr.getPC().setFactorSolverType("mumps")
    solver_corr.setInitialGuessNonzero(True)
    solver_corr.max_it = 200
    solver_corr.setType("gmres")
    solver_corr.getPC().setType("hypre")
    solver_corr.getPC().setHYPREType("boomeramg")
    solver_corr.setOptionsPrefix("corr_")
    solver_corr.setFromOptions()

    solver_up = PETSc.KSP().create(comm)
    solver_up.setOperators(A_up)
    solver_up.setTolerances(rtol=rtol, atol=atol)
    # solver_up.setType("preonly")
    # solver_up.getPC().setType("lu")
    # solver_up.getPC().setFactorSolverType("mumps")
    solver_up.setInitialGuessNonzero(True)
    solver_up.max_it = 200
    solver_up.setType("cg")
    solver_up.getPC().setType("jacobi")
    solver_up.setOptionsPrefix("up_")
    solver_up.setFromOptions()

    # Solve problem
    if out_u:
        out_u.write_function(uh, t)
    if out_p:
        out_p.write_function(ph, t)
    N = int(T / dt)
    if has_tqdm:
        time_range = tqdm(range(N))
    else:
        time_range = range(N)
    for i in time_range:

        if comm.rank == 0:
            print("Time step %d of %d (t=%g):" % (i+1, N, t))

        t += dt
        # Solve step 1
        with common.Timer("~Step 1 (assemble)"):
            u_inlet.interpolate(inlet_velocity(t))
            A_tent.zeroEntries()
            fem.petsc.assemble_matrix(A_tent, a_tent, bcs=bcs_tent)  # type: ignore
            A_tent.assemble()

            b_tent.x.array[:] = 0
            fem.petsc.assemble_vector(b_tent.vector, L_tent)
            fem.petsc.apply_lifting(b_tent.vector, [a_tent], [bcs_tent])
            b_tent.x.scatter_reverse(la.ScatterMode.add)
            fem.petsc.set_bc(b_tent.vector, bcs_tent)
            #solver_tent.solve(b_tent.vector, u_tent.vector)
            #u_tent.x.scatter_forward()

        with common.Timer("~Step 1 (solve)"):
            solver_tent.solve(b_tent.vector, u_tent.vector)
            u_tent.x.scatter_forward()

        if comm.rank == 0:
            print("Step 1 %s after %d iterations" % (
                'converged' if solver_tent.converged else (f'diverged (%s)' % solver_tent.reason),
                solver_tent.its))

        # Solve step 2
        with common.Timer("~Step 2"):
            b_corr.x.array[:] = 0
            fem.petsc.assemble_vector(b_corr.vector, L_corr)
            fem.petsc.apply_lifting(b_corr.vector, [a_corr], [bcs_corr])
            b_corr.x.scatter_reverse(la.ScatterMode.add)
            fem.petsc.set_bc(b_corr.vector, bcs_corr)
            solver_corr.solve(b_corr.vector, phi.vector)
            phi.x.scatter_forward()

            # Update p and previous u
            ph.vector.axpy(1.0, phi.vector)
            ph.x.scatter_forward()

            u_old.x.array[:] = uh.x.array
            u_old.x.scatter_forward()

        if comm.rank == 0:
            print("Step 2 %s after %d iterations" % (
                'converged' if solver_corr.converged else (f'diverged (%s)' % solver_corr.reason),
                solver_corr.its))

        # Solve step 3
        with common.Timer("~Step 3"):
            b_up.x.array[:] = 0
            fem.petsc.assemble_vector(b_up.vector, L_up)
            b_up.x.scatter_reverse(la.ScatterMode.add)
            solver_up.solve(b_up.vector, uh.vector)
            uh.x.scatter_forward()

        if comm.rank == 0:
            print("Step 3 %s after %d iterations" % (
                'converged' if solver_up.converged else (f'diverged (%s)' % solver_up.reason),
                solver_up.its))

        with common.Timer("~IO"):
            if out_u:
                out_u.write_function(uh, t)
            if out_p:
                out_p.write_function(ph, t)

    t_step_1_assemble = comm.gather(common.timing("~Step 1 (assemble)"), root=0)
    t_step_1_solve = comm.gather(common.timing("~Step 1 (solve)"), root=0)
    t_step_2 = comm.gather(common.timing("~Step 2"), root=0)
    t_step_3 = comm.gather(common.timing("~Step 3"), root=0)
    io_time = comm.gather(common.timing("~IO"), root=0)
    if comm.rank == 0:
        print("Time-step breakdown")

        t_step_1_assemble_arr = np.asarray(t_step_1_assemble)
        time_per_run = t_step_1_assemble_arr[:, 1] / t_step_1_assemble_arr[:, 0]
        print(f"Step 1 (assemble): Min time: {np.min(time_per_run):.6g} seconds, Max time: {np.max(time_per_run):.6g} seconds")

        t_step_1_solve_arr = np.asarray(t_step_1_solve)
        time_per_run = t_step_1_solve_arr[:, 1] / t_step_1_solve_arr[:, 0]
        print(f"Step 1 (solve): Min time: {np.min(time_per_run):.6g} seconds, Max time: {np.max(time_per_run):.6g} seconds")

        t_step_2_arr = np.asarray(t_step_2)
        time_per_run = t_step_2_arr[:, 1] / t_step_2_arr[:, 0]
        print(f"Step 2: Min time: {np.min(time_per_run):.6g} seconds, Max time: {np.max(time_per_run):.6g} seconds")

        t_step_3_arr = np.asarray(t_step_3)
        time_per_run = t_step_3_arr[:, 1] / t_step_3_arr[:, 0]
        print(f"Step 3: Min time: {np.min(time_per_run):.6g} seconds, Max time: {np.max(time_per_run):.6g} seconds")

        io_time_arr = np.asarray(io_time)
        time_per_run = io_time_arr[:, 1] / io_time_arr[:, 0]
        print(f"IO {i+1}:   Min time: {np.min(time_per_run):.6g} seconds, Max time: {np.max(time_per_run):.6g} seconds")

    # common.list_timings(comm, [common.TimingType.wall])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run the DFG 2D-3 benchmark\n"
        + "http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('xdmffile', metavar='XDMF-FILE', nargs='?', default=f"meshes/channel2D.xdmf")
    parser.add_argument('xdmffacetfile', metavar='XDMF-FACET-FILE', nargs='?', default=f"meshes/channel2D_facets.xdmf")
    parser.add_argument("--degree-u", default=2, type=int, dest="degree", help="Degree of velocity space")
    _2D = parser.add_mutually_exclusive_group(required=False)
    _2D.add_argument('--3D', dest='threed', action='store_true', help="Use 3D mesh", default=False)
    parser.add_argument("--u-output-file", default=None, type=str, dest="u_outfile", help="name of output file for velocity")
    parser.add_argument("--p-output-file", default=None, type=str, dest="p_outfile", help="name of output file for pressure")
    parser.add_argument("-v", "--verbose", default=0, action='count', dest="verbose", help="be more verbose")
    parser.add_argument('--version', action='version', version='%(prog)s ' + PROGRAM_VERSION)
    args = parser.parse_args()
    dim = 3 if args.threed else 2

    log.set_log_level(log.LogLevel.INFO if args.verbose > 0 else log.LogLevel.WARNING)

    # Create output files
    comm = MPI.COMM_WORLD
    out_u = None
    if args.u_outfile:
        out_u = io.XDMFFile(comm, args.u_outfile, "w")
    out_p = None
    if args.p_outfile:
        out_p = io.XDMFFile(comm, args.p_outfile, "w")

    IPCS(comm, args.xdmffile, args.xdmffacetfile, dim=dim, degree_u=args.degree,
         out_u=out_u, out_p=out_p, verbose=args.verbose)

    if out_u:
        out_u.close()
    if out_p:
        out_p.close()
