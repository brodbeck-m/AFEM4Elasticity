from enum import Enum
from mpi4py import MPI
from petsc4py import PETSc
import time
import typing

from dolfinx import fem, mesh, la, io
import ufl

from .basics import DiscDict, expnum_to_str
from .domain import Domain, EssntBC


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
    bcs: typing.Type[EssntBC],
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
        residual += bcs_weak

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

    if domain.mesh.comm.rank == 0:
        stime = timings[0] + timings[1]
        print(
            f"nlemt - {domain.mesh.topology.index_map(2).size_global}, ndofs - {V.dofmap.index_map.size_global} timing: {stime:.3e} s"
        )

    # --- Export solution to ParaView
    if sdisc.fe_type == FEType.fem_u:
        if outname is not None:
            list_uh = [u_h]

            with io.VTXWriter(MPI.COMM_WORLD, outname + "_pvar-u.bp", [u_h]) as vtx:
                vtx.write(1.0)
    elif sdisc.fe_type == FEType.fem_u_p:
        raise NotImplementedError
    elif sdisc.fe_type == FEType.ls:
        u_h_u = u_h.sub(0).collapse()
        u_h_sig1 = u_h.sub(1).collapse()
        u_h_sig2 = u_h.sub(2).collapse()

        list_uh = [u_h, u_h_sig1, u_h_sig2]

        if outname is not None:
            with io.VTXWriter(MPI.COMM_WORLD, outname + "_pvar-u.bp", [u_h_u]) as vtx:
                vtx.write(1.0)

    return list_uh, V.dofmap.index_map.size_global, timings


# --- The stress equilibrator ---


# --- The error estimator ---
def estimate(
    pi_1: float,
    sdisc: DiscElast,
    u_h: typing.List[fem.Function],
    f: typing.Optional[typing.Any] = None,
) -> typing.Tuple[fem.Function, float, typing.List[float]]:
    raise NotImplementedError("Error estimation not implemented!")
