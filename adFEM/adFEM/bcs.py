from enum import Enum
import typing
from mpi4py import MPI
import numpy as np
from petsc4py.PETSc import ScalarType

from dolfinx import fem, io, mesh
import ufl


# --- The collected boundary conditions ---
class BCs:
    """The collected boundary conditions"""

    def __init__(
        self,
        essential_bcs: typing.List[
            typing.List[
                typing.List[
                    typing.Tuple[typing.List[int], typing.Union[float, typing.Callable]]
                ]
            ]
        ],
        natural_bcs: typing.Optional[
            typing.List[
                typing.List[
                    typing.List[
                        typing.Tuple[
                            typing.List[int], typing.Union[float, typing.Callable]
                        ]
                    ]
                ]
            ]
        ] = None,
    ):
        """Initialise boundary conditions

        Based on a function-space of a problem, boundary conditions (BC) are
        specified for each sub-space respectively each spatial dimension of
        vector valued subspace in a list:

            BCs for vector-valued function-spaces:

                bcs[i][j][:]   -> BC on j-th dimension of the i-th subspace
                bcs[i][j+1][:] -> BC on the entire i-th subspace
                                  (only for essential BCs)

            BCs for scalar-valued function-spaces:

                bcs[i][j][:] -> BC on the i-th subspace

        If a function-space is not mixed, i is equal to 0. Each bc itself is
        specified as a tuple

                            (ids, val)

        where ids are a list of boundary markers and val is the prescribed value.
        The value is thereby either a (float) value of a callable function,depen-
        ding only on the spatial position. If a list is replaced by 'None', no BC
        is prescribed.

        Args:
            essential_bcs: The essential boundary conditions
            natural_bcs:   The natural boundary conditions
        """

        def is_none(inp) -> bool:
            def repl_none(inp):
                if isinstance(inp, list):
                    for i in range(len(inp)):
                        if inp[i] is None:
                            inp[i] = []
                        else:
                            repl_none(inp[i])

            # Check if all inputs are None
            isnone = True if (inp is None) else all(e is None for e in inp)

            # Replace none by empty list
            repl_none(inp)

            return isnone

        def mark_zero_bcs(inp) -> bool:
            all_zero = True

            for i, def_i in enumerate(inp):
                for j, def_ij in enumerate(def_i):
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
        self.esnt_bcs = essential_bcs
        self.has_esnt_bcs = not is_none(self.esnt_bcs)

        # BCs for the dual field
        if natural_bcs is None:
            self.natr_bcs = [None for _ in range(len(essential_bcs))]
        else:
            self.natr_bcs = natural_bcs

        self.has_natr_bcs = not is_none(self.natr_bcs)
        self.natr_bcs_are_zero = mark_zero_bcs(self.natr_bcs)

        # Markers
        self.bcs_initialised: bool = False

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

            # TODO - Add input checks

        # Initialise list of BCs
        bcs_essnt = []
        bcs_natural = None if self.natr_bcs_are_zero else 0

        def set_essential_bcs(
            bcs,
            fspace: typing.Tuple[fem.FunctionSpace, bool],
            result: typing.List[fem.DirichletBCMetaClass],
        ):
            # Extract function-space
            V = fspace[0]
            is_vector_valued = fspace[1]

            # The spatial dimension
            gdim = V.mesh.geometry.dim
            fdim = gdim - 1

            # Functions for prescribing the BC
            uDs = []

            if is_vector_valued:
                # BCs on single vector components
                for j, bcs_j in enumerate(bcs[:-1]):
                    for ids, val in bcs_j:
                        # The boundary DOFs
                        fcts = fct_fkts.indices[np.isin(fct_fkts.values, ids)]
                        dofs = fem.locate_dofs_topological(V.sub(j), fdim, fcts)

                        # Set the boundary condition
                        if callable(val):
                            raise NotImplementedError(
                                "BC currently not implementable in DOLFINx"
                            )
                        else:
                            result.append(
                                fem.dirichletbc(ScalarType(val), dofs, V.sub(j))
                            )

                # BCs on the entire vector-valued subspace
                for ids, val in bcs[-1]:
                    # The boundary DOFs
                    fcts = fct_fkts.indices[np.isin(fct_fkts.values, ids)]
                    dofs = fem.locate_dofs_topological(V, fdim, fcts)

                    # Set the boundary condition
                    if callable(val):
                        uDs.append(fem.Function(V))
                        uDs[-1].interpolate(val)
                        result.append(fem.dirichletbc(uDs[-1], dofs))
                    else:
                        uD = np.array((val,) * gdim, dtype=ScalarType)
                        result.append(fem.dirichletbc(uD, dofs, V))
            else:
                for ids, val in bcs_j:
                    # The boundary DOFs
                    fcts = fct_fkts.indices[np.isin(fct_fkts.values, ids)]
                    dofs = fem.locate_dofs_topological(V, fdim, fcts)

                    # Set the boundary condition
                    if callable(val):
                        uDs.append(fem.Function(V))
                        uDs[-1].interpolate(val)
                        bcs_essnt.append(fem.dirichletbc(uDs[-1], dofs))
                    else:
                        bcs_essnt.append(fem.dirichletbc(ScalarType(val), dofs))

        def set_essential_bcs_mixed_space(
            bcs_i,
            fspace: typing.Tuple[fem.FunctionSpace, int, bool],
            result: typing.List[fem.DirichletBCMetaClass],
        ):
            # Extract function-space
            V = fspace[0]
            i = fspace[1]
            is_vector_valued = fspace[2]

            # The collapsed subspace
            Vi, _ = V.sub(i).collapse()

            # The facet dimension
            fdim = V.mesh.geometry.dim - 1

            # Functions for prescribing the BC
            uDs = []

            if is_vector_valued:
                for j, bcs_ij in enumerate(bcs_i[:-1]):
                    for ids, val in bcs_ij:
                        # Initialise the function
                        uDs.append(fem.Function(Vi))

                        # Interpolate boundary values
                        if callable(val):
                            uDs[-1].interpolate(val)
                        else:
                            if not np.isclose(val, 0.0):
                                uDs[-1].x.array[:] = val

                            # Set the boundary condition
                            fcts = fct_fkts.indices[np.isin(fct_fkts.values, ids)]
                            dofs = fem.locate_dofs_topological(
                                (V.sub(i).sub(j), Vi.sub(j)), fdim, fcts
                            )
                            result.append(fem.dirichletbc(uDs[-1], dofs, V.sub(i)))

                for ids, val in bcs_i[-1]:
                    # Initialise the function
                    uDs.append(fem.Function(Vi))

                    # Interpolate boundary values
                    if callable(val):
                        uDs[-1].interpolate(val)
                    else:
                        if not np.isclose(val, 0.0):
                            uDs[-1].x.array[:] = val

                    # Set the boundary condition
                    fcts = fct_fkts.indices[np.isin(fct_fkts.values, ids)]
                    dofs = fem.locate_dofs_topological((V.sub(i), Vi), fdim, fcts)
                    result.append(fem.dirichletbc(uDs[-1], dofs, V.sub(i)))
            else:
                for ids, val in bcs_i:
                    # Initialise the function
                    uDs.append(fem.Function(Vi))

                    # Interpolate boundary values
                    if callable(val):
                        uDs[-1].interpolate(val)
                    else:
                        if not np.isclose(val, 0.0):
                            uDs[-1].x.array[:] = val

                    # Set the boundary condition
                    fcts = fct_fkts.indices[np.isin(fct_fkts.values, ids)]
                    dofs = fem.locate_dofs_topological((V.sub(i), Vi), fdim, fcts)
                    result.append(fem.dirichletbc(uDs[-1], dofs, V.sub(i)))

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
                    l_i += add_boundary_integrals(bcs_j, v[j], x)
            else:
                l_i = add_boundary_integrals(bcs, v, x)

            return l_i

        for i, (ebcs, nbcs) in enumerate(zip(self.esnt_bcs, self.natr_bcs)):
            # Check if (sub)-space is vector-valued
            is_vvalued = self.v_sub_element_is_vvalued[i]

            # Set BCs
            if self.v_is_mixed:
                set_essential_bcs_mixed_space(ebcs, (V, i, is_vvalued), bcs_essnt)

                if not self.natr_bcs_are_zero:
                    bcs_natural += set_natural_bcs(nbcs, (V, i, is_vvalued), ds)
            else:
                set_essential_bcs(ebcs, (V, is_vvalued), bcs_essnt)

                if not self.natr_bcs_are_zero:
                    bcs_natural += set_natural_bcs(nbcs, (V, None, is_vvalued), ds)

        return bcs_essnt, bcs_natural

    def set_for_equilibration(self):
        raise NotImplementedError("Method not implemented")
