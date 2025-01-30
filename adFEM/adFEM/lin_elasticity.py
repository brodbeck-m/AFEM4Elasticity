from enum import Enum
from mpi4py import MPI
from petsc4py import PETSc
import time
import typing

from dolfinx import cpp, fem, mesh, la, io
import ufl

from dolfinx_eqlb.cpp import local_solver_cholesky
from dolfinx_eqlb.eqlb import FluxEqlbSE
from dolfinx_eqlb.eqlb.check_eqlb_conditions import (
    check_divergence_condition,
    check_jump_condition,
    check_weak_symmetry_condition,
)

from .basics import DiscDict, expnum_to_str
from .domain import Domain
from .bcs import BCs


# --- Enum classes ---
class FEType(Enum):
    fem_u = 0
    fem_u_p = 1
    ls = 2


class EEType(Enum):
    none = 0
    gee = 1
    hee = 2
    ls = 3


# --- The spatial discretisation ---
class DiscElast(DiscDict):
    def __init__(
        self,
        degree: typing.Union[int, typing.List[int]],
        fe_type: typing.Optional[FEType] = FEType.fem_u,
        ee_type: typing.Optional[EEType] = EEType.none,
        stress_space: typing.Optional[str] = "RT",
        quadrature_degree: typing.Optional[int] = None,
    ):
        if fe_type == FEType.ls and not isinstance(degree, list):
            raise ValueError(
                "Elasticity with least-squares: Specify degree of u and sigma in a list!"
            )

        if fe_type == FEType.fem_u_p and not isinstance(degree, list):
            raise ValueError(
                "Elasticity with least-squares: Specify degree of u and p in a list!"
            )

        # Constructor of super class
        super().__init__(degree, fe_type, ee_type, stress_space, quadrature_degree)

        if self.fe_type == FEType.ls:
            self.ee_type = EEType.ls

        # Flag, if equilibrated stress is symmetric
        self.symmetric_estress = None

    def specify_equilibration(
        self,
        degree: typing.Optional[int] = None,
        symmetric_stress: typing.Optional[bool] = False,
    ):
        if self.ee_type == EEType.gee and not symmetric_stress:
            raise ValueError("Stresses with weak symmetry condition required!")

        if symmetric_stress and (self.degree < 2):
            raise ValueError(
                "Equilibration of weakly symmetric stresses requires a primal approximation order >2!"
            )

        # Set basic properties
        super().specify_equilibration(degree)

        # Set flag for weakly symmetric, equilibrated stresses
        self.symmetric_estress = symmetric_stress

    def output_name(
        self,
        pi1: float,
        name_domain: str,
        marking: typing.Optional[str] = None,
    ) -> str:
        # Recast material parameter to string
        pival = expnum_to_str(pi1)

        # Type-sting for the discretisation
        if self.fe_type == FEType.fem_u:
            tstr = "fem-u-P{}".format(self.degree)
        elif self.fe_type == FEType.fem_u_p:
            if self.degree[0] == self.degree[1]:
                tstr = "fem-up-P{}".format(self.degree[0])
        elif self.fe_type == FEType.ls:
            tstr = "ls-usig-P{}-{}{}".format(
                self.degree[0], self.dual_space, self.degree[1]
            )

        outname = name_domain + "-linelast_pi1-" + pival + "_" + tstr

        if marking is not None:
            outname += "_" + marking

        return outname


def symgrad(u):
    return ufl.sym(ufl.grad(u))


def Asigma(sig, pi_1):
    return 0.5 * (sig - (pi_1 / (2 * (pi_1 + 1))) * ufl.tr(sig) * ufl.Identity(2))


# Galerkin FEM for elasticity
def weak_form_fem_u(
    pi_1: float, msh: mesh.Mesh, sdisc: DiscElast, f: typing.Any
) -> typing.Tuple[fem.FunctionSpace, ufl.Form]:
    # The spatial dimension
    gdim = msh.geometry.dim

    # The function space
    V = fem.VectorFunctionSpace(msh, ("P", sdisc.degree))

    # Trial- and test functions
    u = ufl.TrialFunction(V)
    v_u = ufl.TestFunction(V)

    # The variational form
    sigma = 2 * symgrad(u) + pi_1 * ufl.div(u) * ufl.Identity(gdim)

    residual = ufl.inner(sigma, symgrad(v_u)) * ufl.dx

    if sdisc.quadrature_degree is None:
        dvol = ufl.dx
    else:
        dvol = ufl.dx(degree=sdisc.quadrature_degree)

    if f is not None:
        residual -= ufl.inner(f, v_u) * dvol

    return V, residual


def weak_form_fem_up(
    pi_1: float, msh: mesh.Mesh, sdisc: DiscElast, f: typing.Any
) -> typing.Tuple[fem.FunctionSpace, ufl.Form]:
    raise NotImplementedError("Elasticity in u-p formulation not implemented!")


# Least-Squares FEM for elasticity
def weak_form_ls(
    pi_1: float, msh: mesh.Mesh, sdisc: DiscElast, f: typing.Any
) -> typing.Tuple[fem.FunctionSpace, ufl.Form]:
    # Material-specific compliance
    Asig = lambda sig: Asigma(sig, pi_1)

    # The function space
    P_u = ufl.VectorElement("P", msh.ufl_cell(), sdisc.degree[0])

    if sdisc.dual_space == "RT":
        P_sig = ufl.FiniteElement("RT", msh.ufl_cell(), sdisc.degree[1])
    elif sdisc.dual_space == "BDM":
        P_sig = ufl.FiniteElement("BDM", msh.ufl_cell(), sdisc.degree[1])
    else:
        raise ValueError("Unknown stress space")

    V = fem.FunctionSpace(msh, ufl.MixedElement([P_u, P_sig, P_sig]))

    # Trial- and test functions
    u, sig1, sig2 = ufl.TrialFunctions(V)
    v_u, v_sig1, v_sig2 = ufl.TestFunctions(V)

    # The variational form
    sig = ufl.as_matrix([[sig1[0], sig1[1]], [sig2[0], sig2[1]]])
    v_sig = ufl.as_matrix([[v_sig1[0], v_sig1[1]], [v_sig2[0], v_sig2[1]]])

    residual = (
        ufl.inner(symgrad(u) - Asig(sig), symgrad(v_u) - Asig(v_sig))
        + ufl.inner(ufl.div(sig), ufl.div(v_sig))
    ) * ufl.dx

    if f is not None:
        if sdisc.quadrature_degree is None:
            dvol = ufl.dx
        else:
            dvol = ufl.dx(degree=sdisc.quadrature_degree)

        residual += ufl.inner(f, ufl.div(v_sig)) * dvol

    return V, residual


# --- The solver ---
def solve(
    pi_1: float,
    domain: Domain,
    bcs: typing.Type[typing.Any],
    sdisc: DiscElast,
    f: typing.Optional[typing.Any] = None,
    outname: typing.Optional[str] = None,
) -> typing.Tuple[typing.List[fem.Function], int, typing.List[float]]:
    """Solves a liner elasticity problem

    Args:
        domain:  The domain
        bcs:     The boundary conditions
        sdisc:   The spatial discretisation
        f:       The source term
        outname: The name of the output file

    Returns:
        The approximate solution,
        The number of degrees of freedom,
        The solution time
    """

    timings = [0.0, 0.0]

    # --- The weak form
    if sdisc.fe_type == FEType.fem_u:
        weak_form = weak_form_fem_u
    elif sdisc.fe_type == FEType.fem_u_p:
        weak_form = weak_form_fem_up
    elif sdisc.fe_type == FEType.ls:
        weak_form = weak_form_ls
    else:
        raise NotImplementedError("Unsupported discretisation")

    V, residual = weak_form(pi_1, domain.mesh, sdisc, f)

    # --- The boundary conditions
    domain.mesh.topology.create_connectivity(1, 2)
    bcs_essnt, bcs_weak = bcs.set(V, domain.facet_functions, domain.ds)

    if bcs_weak is not None:
        residual -= bcs_weak

    # --- The solver
    u_h = fem.Function(V)

    timings[0] -= time.perf_counter()
    # Separate bilinear form
    a = fem.form(ufl.lhs(residual))

    # Assemble equation system
    A = fem.petsc.assemble_matrix(a, bcs=bcs_essnt)
    A.assemble()

    if f is None and bcs_weak is None:
        L = la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
    else:
        l = fem.form(ufl.rhs(residual))
        L = fem.petsc.create_vector(l)
        fem.petsc.assemble_vector(L, l)

    fem.apply_lifting(L, [a], [bcs_essnt])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(L, bcs_essnt)
    timings[0] += time.perf_counter()

    timings[1] -= time.perf_counter()
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")

    solver.setTolerances(rtol=1e-10, atol=1e-10, max_it=1000)

    solver.solve(L, u_h.vector)
    timings[1] += time.perf_counter()

    # --- Export solution to ParaView
    if sdisc.fe_type == FEType.fem_u:
        list_uh = [u_h]
        ndofs = V.dofmap.bs * V.dofmap.index_map.size_global

        if outname is not None:
            u_h.name = "displacement"
            with io.VTXWriter(MPI.COMM_WORLD, outname + "_pvar-u.bp", [u_h]) as vtx:
                vtx.write(1.0)
    elif sdisc.fe_type == FEType.fem_u_p:
        raise NotImplementedError
    elif sdisc.fe_type == FEType.ls:
        u_h_u = u_h.sub(0).collapse()
        u_h_sig1 = u_h.sub(1).collapse()
        u_h_sig2 = u_h.sub(2).collapse()

        list_uh = [u_h_u, u_h_sig1, u_h_sig2]
        ndofs = V.dofmap.index_map.size_global

        if outname is not None:
            u_h_u.name = "displacement"
            with io.VTXWriter(MPI.COMM_WORLD, outname + "_pvar-u.bp", [u_h_u]) as vtx:
                vtx.write(1.0)

    if domain.mesh.comm.rank == 0:
        stime = timings[0] + timings[1]
        print(
            f"nlemt - {domain.mesh.topology.index_map(2).size_global}, ndofs - {ndofs} timing: {stime:.3e} s"
        )

    return list_uh, ndofs, timings


# --- The stress equilibrator ---
def equilibrate(
    pi_1: float,
    domain: Domain,
    bcs: BCs,
    sdisc: DiscElast,
    u_h: typing.List[fem.Function],
    f: typing.Any,
    check_equilibration: typing.Optional[bool] = False,
) -> typing.Tuple[
    typing.List[fem.Function],
    typing.List[fem.Function],
    fem.Function,
    typing.List[float],
]:

    def set_forms_projection(
        msh: mesh.Mesh,
        degree_eqlb: int,
        sig_h: typing.Any,
        f: typing.Any,
        quadrature_degree: typing.Optional[int] = None,
    ):
        # The function space to project into
        V = fem.VectorFunctionSpace(msh, ("DG", degree_eqlb - 1))

        # Trial- and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # The bilinear form
        a = fem.form(ufl.inner(u, v) * ufl.dx)

        # The linear form
        if quadrature_degree is None:
            dvol = ufl.dx
        else:
            dvol = ufl.Measure(
                "dx",
                domain=V.mesh,
                metadata={"quadrature_degree": quadrature_degree},
            )

        ls = [
            fem.form(ufl.inner(ufl.as_vector([sig_h[0, 0], sig_h[0, 1]]), v) * dvol),
            fem.form(ufl.inner(ufl.as_vector([sig_h[1, 0], sig_h[1, 1]]), v) * dvol),
        ]

        if f is not None:
            ls.append(fem.form(ufl.inner(f, v) * dvol))

        # The solution function
        results = [fem.Function(V) for _ in range(3)]

        return a, ls, results

    # Check input
    if sdisc.symmetric_estress and sdisc.degree < 2:
        raise ValueError(
            "Weakly symmetric stress equilibration requires a primal approximation order >2!"
        )

    # Initialise timings
    timings = [0.0, 0.0]

    # The approximated stress (with negative sign!)
    if sdisc.fe_type == FEType.fem_u:
        sigma_h = -2 * symgrad(u_h[0]) - pi_1 * ufl.div(u_h[0]) * ufl.Identity(2)
    elif sdisc.fe_type == FEType.fem_u_p:
        raise NotImplementedError("Equilibration for u-p formulation not implemented!")

    # Required projections
    ap, lp, projections = set_forms_projection(
        domain.mesh, sdisc.degree_eflux, sigma_h, f, sdisc.quadrature_degree
    )

    timings[0] -= time.perf_counter()
    local_solver_cholesky([projections[i]._cpp_object for i in range(len(lp))], ap, lp)
    timings[0] += time.perf_counter()

    # Recast projected RHS into its components
    timings[0] -= time.perf_counter()
    rhs_proj = [projections[2].sub(i).collapse() for i in range(2)]
    timings[0] += time.perf_counter()

    # The equilibrator
    equilibrator = FluxEqlbSE(
        sdisc.degree_eflux,
        domain.mesh,
        rhs_proj,
        projections[0:2],
        sdisc.symmetric_estress,
        True,
    )

    bcs_eqlb, esntfcts = bcs.set_for_equilibration(
        equilibrator.V_flux, domain.facet_functions, -1.0
    )
    equilibrator.set_boundary_conditions(esntfcts, bcs_eqlb)

    # Solve equilibration
    timings[1] -= time.perf_counter()
    equilibrator.equilibrate_fluxes()
    timings[1] += time.perf_counter()

    if check_equilibration:
        # Stress as ufl matrix
        stress_eqlb = ufl.as_matrix(
            [
                [-equilibrator.list_flux[0][0], -equilibrator.list_flux[0][1]],
                [-equilibrator.list_flux[1][0], -equilibrator.list_flux[1][1]],
            ]
        )

        stress_proj = ufl.as_matrix(
            [
                [-projections[0][0], -projections[0][1]],
                [-projections[1][0], -projections[1][1]],
            ]
        )

        # The divergence condition
        div_condition_fulfilled = check_divergence_condition(
            stress_eqlb,
            stress_proj,
            projections[2],
            mesh=domain.mesh,
            degree=sdisc.degree_eflux,
            flux_is_dg=True,
        )

        if not div_condition_fulfilled:
            raise ValueError("Divergence conditions not fulfilled")

        # Check if stress is H(div)
        for i in range(2):
            jump_condition_fulfilled = check_jump_condition(
                equilibrator.list_flux[i], projections[i]
            )

            if not jump_condition_fulfilled:
                raise ValueError("Jump conditions not fulfilled")

        # Check weak symmetry condition
        if sdisc.symmetric_estress:
            wsym_condition = check_weak_symmetry_condition(equilibrator.list_flux)

            if not wsym_condition:
                raise ValueError("Weak symmetry conditions not fulfilled")

    return (
        projections[0:2],
        equilibrator.list_flux,
        equilibrator.korn_constants,
        timings,
    )


# --- The error estimator ---
def estimate(
    pi_1: float,
    domain: Domain,
    bcs: BCs,
    sdisc: DiscElast,
    u_h: typing.List[fem.Function],
    f: typing.Optional[typing.Any] = None,
) -> typing.Tuple[fem.Function, float, typing.List[float]]:
    """Evaluate the error estimate

    Args:
        pi_1:  The ratio of the lambda and mu
        sdisc: The spatial discretisation
        u_h:   The approximated solution
        f:     The source term

    Returns:
        The error estimate (per cell),
        The overall, estimated error,
        Timings [projection, equilibration, evaluate ee]
    """

    # Auxiliaries
    def rows_to_uflmat(rows: typing.List[fem.Function], gdim: int):
        if gdim == 2:
            return ufl.as_matrix([[rows[0][0], rows[0][1]], [rows[1][0], rows[1][1]]])

    # The spatial dimension
    gdim = u_h[0].function_space.mesh.geometry.dim

    # The error estimate
    def ee_least_squares_functional(
        pi_1: float, u_h: fem.Function, sig_h: typing.Any, f: typing.Any
    ) -> typing.Tuple[typing.Any, fem.Function]:
        # The DG0 function space
        V = fem.FunctionSpace(u_h.function_space.mesh, ("DG", 0))
        v = ufl.TrialFunction(V)

        # Error of the stress-strain relation
        eeps = Asigma(sig_h, pi_1) - symgrad(u_h)

        # Error of the balance of linear momentum
        eblm = ufl.div(sig_h) + f if f is not None else ufl.div(sig_h)

        # The least-squares functional
        form_lsf = fem.form(
            (ufl.inner(eeps, eeps) + ufl.inner(eblm, eblm)) * v * ufl.dx
        )

        return form_lsf, fem.Function(V)

    def ee_eqlb(
        pi_1: float,
        u_h: fem.Function,
        d_sig_R: typing.Any,
        sig_h: typing.List[fem.Function],
        f: typing.Any,
        korn: fem.Function,
        guarantied_upper_bound: bool,
    ) -> typing.Tuple[typing.Any, fem.Function]:
        # The mesh
        msh = u_h.function_space.mesh

        # The DG0 function space
        V = fem.FunctionSpace(msh, ("DG", 0))
        v = ufl.TrialFunction(V)

        # The basic error estimate
        eta = ufl.inner(d_sig_R, Asigma(d_sig_R, pi_1))

        # Error due to data oscillation
        if f is not None:
            # The characteristic mesh length
            i_m = msh.topology.index_map(2)

            h = fem.Function(V)
            h.x.array[:] = cpp.mesh.h(msh, 2, range(i_m.size_local + i_m.num_ghosts))

            # Error due to data oscillation
            eo = korn * (h / ufl.pi) * (f + ufl.div(-rows_to_uflmat(sig_h) + d_sig_R))

            # Extend the error estimate
            eta += ufl.inner(eo, eo)

        if guarantied_upper_bound:
            # Error due to assymetry
            es = 0.5 * korn * (d_sig_R[0, 1] - d_sig_R[1, 0])
            eta += ufl.inner(es, es)

            if f is not None:
                eta += 2 * ufl.sqrt(ufl.inner(eo, eo)) * ufl.sqrt(ufl.inner(es, es))

        return fem.form(eta * v * ufl.dx), fem.Function(V)

    if sdisc.fe_type == FEType.ls:
        # Stress-tensor as UFL matrix
        sigma_h = rows_to_uflmat(u_h[1:], gdim)

        # Compile the error estimate
        form_eta, eta = ee_least_squares_functional(pi_1, u_h[0], sigma_h, f)
    else:
        # Equilibrated the stress tensor
        rows_sig_h, rows_d_sig_R, korns_constants, timings = equilibrate(
            pi_1,
            domain,
            bcs,
            sdisc,
            u_h,
            f,
            True,
        )

        # Equilibrated stress-tensor as UFL matrix
        d_sigma_R = -rows_to_uflmat(rows_d_sig_R, gdim)

        # Compile the error estimate
        if sdisc.ee_type == EEType.gee:
            form_eta, eta = ee_eqlb(
                pi_1, u_h[0], d_sigma_R, rows_sig_h, f, korns_constants, True
            )
        elif sdisc.ee_type == EEType.hee:
            form_eta, eta = ee_eqlb(
                pi_1, u_h[0], d_sigma_R, rows_sig_h, f, korns_constants, False
            )
        elif sdisc.ee_type == EEType.ls:
            sigma_h = -rows_to_uflmat(rows_sig_h, gdim)
            form_eta, eta = ee_least_squares_functional(
                pi_1, u_h[0], d_sigma_R + sigma_h, f
            )

    # Assemble the cell-wise error
    timings.append(-time.perf_counter())
    fem.petsc.assemble_vector(eta.vector, form_eta)
    eta.x.scatter_forward()
    timings[-1] += time.perf_counter()

    return eta, eta.vector.sum(), timings
