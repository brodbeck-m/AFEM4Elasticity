import numpy as np
import time
import typing

from dolfinx import fem

from .basics import EqType, DiscDict
from .domain import AbstractDomain, MarkingStrategy
from .bcs import BCs
from .lin_elasticity import solve as solve_linelast
from .lin_elasticity import estimate as estimate_linelast


# --- The adaptive solver ---
def adaptive_solver(
    eq_type: EqType,
    matpar: typing.Union[float, typing.List[float]],
    domain: typing.Type[AbstractDomain],
    bcs: typing.Type[BCs],
    sdisc: typing.Type[DiscDict],
    h_0: typing.Union[int, typing.List[int], float],
    f: typing.Optional[typing.Any] = None,
    return_solution_series: bool = False,
) -> typing.Union[typing.List[fem.Function], typing.List[typing.List[fem.Function]]]:
    """Adaptive solution procedure for stationary PDEs.

    Args:
        eq_type:                The type of the model
        matpar:                 The material parameters of the model
        domain:                 The (abstract) domain
        bcs:                    The (abstract) boundary conditions
        sdisc:                  The spatial discretisation
        h_0:                    The initial mesh size
        f:                      The source term
        return_solution_series: True, if the results of all intermediate solutions should be returned
    """

    timing_total = -time.perf_counter()

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
    results = np.zeros((domain.refinement_level_max, 11))

    # The initial mesh
    mdomain = domain.create(h_0)

    for n in range(0, domain.refinement_level_max):
        # Solve
        outname_sol = outname_base + "-mesh" + str(n)

        u_h, ndofs, timings_solve = solve(mdomain, outname_sol)

        # Estimate error
        eta, eta_tot, timings_estimate = estimate(mdomain, u_h)

        # Store results
        id = domain.refinement_level

        if return_solution_series:
            list_uh.append(u_h)

        results[id, 0] = mdomain.mesh.topology.index_map(2).size_global
        results[id, 1] = ndofs
        results[id, 2] = eta_tot
        results[id, 3] = timings_solve[0]
        results[id, 4] = timings_solve[1]
        results[id, 5] = timings_estimate[0]
        results[id, 6] = timings_estimate[1]
        results[id, 7] = timings_estimate[2]
        results[id, 8] = sum(timings_solve)
        results[id, 9] = sum(timings_estimate)
        results[id, 10] = sum(results[0 : id + 1, 8]) + sum(results[0 : id + 1, 9])

        # Refine
        mdomain = domain.refine(mdomain.mesh, (eta, eta_tot))

    if mdomain.mesh.comm.rank == 0:
        header = "nelmt, ndofs, err, tasmbl, tsolve, tproj, teqlb, tevalee, tprimal, tee, ttotal"
        np.savetxt(outname_base + ".csv", results, delimiter=",", header=header)

    timing_total += time.perf_counter()

    if mdomain.mesh.comm.rank == 0:
        print(f"\nJob completed after {timing_total:.3e} s")

    if return_solution_series:
        return list_uh
    else:
        return u_h
