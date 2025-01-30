import numpy as np
import typing

from .basics import EqType, DiscDict
from .domain import AdaptiveDomain, MarkingStrategy
from .bcs import BCs
from .lin_elasticity import solve as solve_linelast
from .lin_elasticity import estimate as estimate_linelast


# --- The adaptive solver ---
def adaptive_solver(
    eq_type: EqType,
    matpar: typing.Union[float, typing.List[float]],
    domain: typing.Type[AdaptiveDomain],
    bcs: typing.Type[BCs],
    sdisc: typing.Type[DiscDict],
    h_0: typing.Union[int, typing.List[int], float],
    f: typing.Optional[typing.Any] = None,
    evaluate_true_error: bool = False,
):
    """Adaptive solution procedure for linear elasticity

    Args:
        eq_type:             The type of the model
        matpar:              The material parameters of the model
        domain:              The (abstract) domain
        bcs:                 The (abstract) boundary conditions
        sdisc:               The spatial discretisation
        h_0:                 The initial mesh size
        f:                   The source term
        evaluate_true_error: True, if the error is calculated
    """

    # The customised solver
    if eq_type == EqType.lin_elasticity:
        solve = lambda m, o: solve_linelast(matpar, m, bcs, sdisc, f, o)
        estimate = lambda dm, uh: estimate_linelast(matpar, dm, bcs, sdisc, uh, f)
    else:
        raise NotImplementedError("Model currently not supported!")

    # Basic output name
    outname_base = sdisc.output_name(matpar, domain.name)

    if domain.marking_strategy is not MarkingStrategy.none:
        if domain.marking_strategy is MarkingStrategy.doerfler:
            mtstr = "doerfler-{:.2f}".format(domain.marking_parameter).replace(".", "d")
        elif domain.marking_strategy is MarkingStrategy.maximum:
            mtstr = "maximum-{:.2f}".format(domain.marking_parameter).replace(".", "d")

        outname_base += "_" + mtstr

    # Storage of the results
    list_uh = []

    if evaluate_true_error:
        results = np.zeros((domain.refinement_level_max, 4))
    else:
        results = np.zeros((domain.refinement_level_max, 4))

    # The initial mesh
    mdomain = domain.create(h_0)

    if mdomain.mesh.comm.rank == 0:
        logfile = open(outname_base + ".log", "w")

    for n in range(0, domain.refinement_level_max):
        # Solve
        outname_sol = outname_base + "-mesh" + str(n)

        u_h, ndofs, timings_solve = solve(mdomain, outname_sol)

        if mdomain.mesh.comm.rank == 0:
            stime = sum(timings_solve)
            logfile.write(
                f"nlemt - {mdomain.mesh.topology.index_map(2).size_global}, ndofs - {ndofs}, timing: {stime:.3e} s\n"
            )

        # Estimate error
        eta, eta_tot, timings_estimate = estimate(mdomain, u_h)

        # Store results
        id = domain.refinement_level

        if evaluate_true_error:
            list_uh.append(u_h)

        results[id, 0] = mdomain.mesh.topology.index_map(2).size_global
        results[id, 1] = ndofs
        results[id, 2] = eta_tot

        # Refine
        mdomain = domain.refine(mdomain.mesh, (eta, eta_tot))

    # --- Post processing
    if evaluate_true_error:
        # Increase discretisation oder by one
        if isinstance(sdisc.degree, list):
            for i in range(len(sdisc.degree)):
                sdisc.degree[i] += 1
        else:
            sdisc.degree += 1

        # Uniform mesh refinement
        mdomain = domain.refine(1.0, mdomain)

        # Calculate over-kill solution
        outname_ref = outname_base + "-overkill"
        u_ext, ndofs, time_solve = solve(mdomain, outname_ref)

        if mdomain.mesh.comm.rank == 0:
            logfile.write("\nOverkill solution\n")
            logfile.write(
                f"nlemt - {mdomain.mesh.topology.index_map(2).size_global}, ndofs - {ndofs}, timing: {time_solve:.3e} s\n"
            )

        # Evaluate error relative to overkill solution
        time_post = ...

        if mdomain.mesh.comm.rank == 0:
            logfile.write(f"\nPostprocessing done in {time_post:.3e} s")
            logfile.close()

    np.savetxt("TestOut.csv", results, delimiter=",")
