from enum import Enum
import typing
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc

from dolfinx import fem, io, mesh
import ufl


# --- The collected boundary conditions ---
class BCs:
    """The collected boundary conditions

    Based on a function-space of a problem, boundary conditions (BC) are
    specified for each sub-space respectively each spatial dimension of
    vector valued subspaces.



    """

    def __init__(
        self,
        primal_bcs: typing.List[
            typing.List[
                typing.List[
                    typing.Tuple[typing.List[int], typing.Union[float, typing.Callable]]
                ]
            ]
        ],
        dual_bcs: typing.List[
            typing.List[
                typing.List[
                    typing.Tuple[typing.List[int], typing.Union[float, typing.Callable]]
                ]
            ]
        ],
    ):
        def empty_input(inp) -> bool:
            for sublist in inp:
                if isinstance(sublist, list):
                    if not empty_input(sublist):
                        return False
                elif sublist:
                    return False
            return True

        def mark_zero_dual_bcs(inp) -> bool:
            all_zero = True

            for i, def_i in enumerate(inp):
                for j, def_ij in enumerate(def_i):
                    if def_ij:
                        for k, (_, val) in enumerate(def_ij):
                            if callable(val):
                                all_zero = False
                                inp[i][j][k] += (False,)
                            else:
                                if np.isclose(val, 0.0):
                                    inp[i][j][k] += (True,)
                                else:
                                    all_zero = False
                                    inp[i][j][k] += (False,)

            return all_zero

        # The underlying function-space
        self.v_is_mixed: bool = False
        self.v_num_sub_elements: typing.List[int] = []
        self.v_sub_element_is_vvalued: typing.List[bool] = []

        # BCs for the primal field
        self.primal_bcs = primal_bcs

        # BCs for the dual field
        self.dual_bcs = dual_bcs

        # Markers
        self.bcs_initialised: bool = False
        self.has_primal_bcs = not empty_input(self.primal_bcs)
        self.has_dual_bcs = not empty_input(self.dual_bcs)
        if self.has_dual_bcs:
            self.dual_bcs_are_zero = mark_zero_dual_bcs(self.dual_bcs)
        else:
            self.dual_bcs_are_zero = True

    def set(
        self,
        V: fem.FunctionSpace,
        fct_fkts: typing.Any,
        ds: typing.Any,
    ) -> typing.Tuple[typing.Union[typing.List[fem.DirichletBCMetaClass]], typing.Any]:
        """Set boundary conditions

        Args:
            V:        The function space
            fct_fkts: The facet functions
            ds:       The facet integrators

        Returns:
            Essential BCs as list of DirichletBCs (Default: []),
            Natural BCs as additional term for the residual (Default: None)
        """

        # Check input
        if not self.bcs_initialised:
            # Set marker
            self.bcs_initialised = True

            # The number of sub-spaces on the function space
            nsub = V.element.num_sub_elements

            # Check if function-space if mixed
            self.v_is_mixed = False if (nsub == V.dofmap.bs) else True

            # Characterisation of the function-space
            if self.v_is_mixed:
                for i in range(nsub):
                    nsubsub = V.sub(i).element.num_sub_elements

                    if nsubsub == 0:
                        self.v_num_sub_elements.append(1)
                        self.v_sub_element_is_vvalued.append(False)
                    else:
                        self.v_num_sub_elements.append(nsubsub)
                        self.v_sub_element_is_vvalued.append(True)
            else:
                if nsub == 0:
                    self.v_num_sub_elements.append(1)
                    self.v_sub_element_is_vvalued.append(False)
                else:
                    self.v_num_sub_elements.append(nsub)
                    self.v_sub_element_is_vvalued.append(True)

            # Check input
            for i, numsub_i in enumerate(self.v_num_sub_elements):
                if len(self.primal_bcs[i]) != numsub_i:
                    raise ValueError(
                        "The number of primal BCs does not match the number of subspaces!"
                    )

                if len(self.dual_bcs[i]) != numsub_i:
                    raise ValueError(
                        "The number of dual BCs does not match the number of subspaces!"
                    )

        # The facet dimension
        fdim = V.mesh.topology.dim - 1

        # Initialise list of BCs
        bcs_essnt = []
        bcs_natural = None if self.dual_bcs_are_zero else 0

        def set_essential_bcs(
            fspace: typing.Tuple[fem.FunctionSpace, bool], definitions
        ):
            # Extract function-space
            V = fspace[0]
            is_vector_valued = fspace[1]

            # Functions for prescribing the BC
            uDs = []

            if is_vector_valued:
                # The FE space of each vector component
                for i, def_i in enumerate(definitions):
                    if def_i:
                        for ids, val in def_i:
                            # The boundary DOFs
                            dofs = fem.locate_dofs_topological(
                                V.sub(i),
                                fdim,
                                fct_fkts.indices[np.isin(fct_fkts.values, ids)],
                            )

                            # Set the boundary condition
                            if callable(val):
                                raise NotImplementedError(
                                    "BC currently not implementable in DOLFINx"
                                )
                            else:
                                bcs_essnt.append(
                                    fem.dirichletbc(
                                        PETSc.ScalarType(val), dofs, V.sub(i)
                                    )
                                )
            else:
                if def_i:
                    for ids, val in definitions:
                        # The boundary DOFs
                        dofs = fem.locate_dofs_topological(
                            V, fdim, fct_fkts.indices[np.isin(fct_fkts.values, ids)]
                        )

                        # Set the boundary condition
                        if callable(val):
                            uDs.append(fem.Function(V))
                            uDs[-1].interpolate(val)

                            bcs_essnt.append(fem.dirichletbc(uDs[-1], dofs))
                        else:
                            bcs_essnt.append(
                                fem.dirichletbc(PETSc.ScalarType(val), dofs)
                            )

        def set_essential_bcs_mixed_space(
            fspace: typing.Tuple[fem.FunctionSpace, int, bool], definitions
        ):
            # Extract function-space
            V = fspace[0]
            i = fspace[1]
            is_vector_valued = fspace[2]

            # The collapsed subspace
            Vi, _ = V.sub(i).collapse()

            # Functions for prescribing the BC
            uDs = []

            if is_vector_valued:
                for j, def_j in enumerate(definitions):
                    if def_j:
                        for ids, val in def_j:
                            # Initialise the function
                            uDs.append(fem.Function(Vi))

                            # Interpolate boundary values
                            if callable(val):
                                uDs[-1].interpolate(val)
                            else:
                                if not np.isclose(val, 0.0):
                                    uDs[-1].x.array[:] = val

                            # Set the boundary condition
                            dofs = fem.locate_dofs_topological(
                                (V.sub(i).sub(j), Vi.sub(j)),
                                fdim,
                                fct_fkts.indices[np.isin(fct_fkts.values, ids)],
                            )
                            bcs_essnt.append(fem.dirichletbc(uDs[-1], dofs, V.sub(i)))
            else:
                if definitions:
                    for ids, val in definitions:
                        # Initialise the function
                        uDs.append(fem.Function(Vi))

                        # Interpolate boundary values
                        if callable(val):
                            uDs[-1].interpolate(val)
                        else:
                            if not np.isclose(val, 0.0):
                                uDs[-1].x.array[:] = val

                        # Set the boundary condition
                        dofs = fem.locate_dofs_topological(
                            (V.sub(i), Vi),
                            fdim,
                            fct_fkts.indices[np.isin(fct_fkts.values, ids)],
                        )
                        bcs_essnt.append(fem.dirichletbc(uDs[-1], dofs, V.sub(i)))

        def set_natural_bcs(
            bcs, fspace: typing.Tuple[fem.FunctionSpace, int, bool], ds: typing.Any
        ) -> typing.Any:

            def add_boundary_integrals(bcs, v, x) -> typing.Any:
                l_ij = 0

                for ids, val, is_zero in bcs:
                    if not is_zero:
                        # Value times test-function
                        val_t_v = val(x) * v if callable(val) else val * v

                        for id in ids:
                            l_ij += val_t_v * ds(id)

                return l_ij

            # Extract function-space
            V = fspace[0]
            i = fspace[1]
            is_vector_valued = fspace[2]

            # Functions for prescribing the BC
            x = ufl.SpatialCoordinate(V.mesh)

            # The test space
            if i is None:
                v = ufl.TestFunction(V)
            else:
                v = ufl.TestFunctions(V)[i]

            # Add residual terms
            if is_vector_valued:
                l_i = 0

                for j, bcs_j in enumerate(bcs):
                    if bcs_j:
                        l_i += add_boundary_integrals(bcs_j, v[j], x)
            else:
                if bcs:
                    l_i = add_boundary_integrals(bcs, v, x)

            return l_i

        for i, (ebcs, nbcs) in enumerate(zip(self.primal_bcs, self.dual_bcs)):
            # Check if (sub)-space is vector-valued
            is_vvalued = self.v_sub_element_is_vvalued[i]

            # Set BCs
            if self.v_is_mixed:
                set_essential_bcs_mixed_space((V, i, is_vvalued), ebcs)

                if not self.dual_bcs_are_zero:
                    bcs_natural += set_natural_bcs(nbcs, (V, i, is_vvalued), ds)
            else:
                set_essential_bcs((V, is_vvalued), ebcs)

                if not self.dual_bcs_are_zero:
                    bcs_natural += set_natural_bcs(nbcs, (V, None, is_vvalued), ds)

        return bcs_essnt, bcs_natural

    def set_for_equilibration(self):
        raise NotImplementedError("Method not implemented")
