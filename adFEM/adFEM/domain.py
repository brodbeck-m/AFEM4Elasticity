from enum import Enum
import typing
import numpy as np

from dolfinx import fem, io, mesh
import ufl


class MarkingStrategy(Enum):
    none = 0
    doerfler = 1
    maximum = 2


class Domain:
    """A domain"""

    def __init__(self, mesh: mesh.Mesh, facet_fkts: typing.Any, ds: typing.Any):
        """Constructor

        Args:
            mesh:       The mesh
            facet_fkts: The facet functions
            ds:         The facet integrators
        """

        # Mesh
        self.mesh = mesh

        # Facet functions
        self.facet_functions = facet_fkts

        # Integrators
        self.ds = ds


class AdaptiveDomain:
    """An adaptive domain
    Create an initial mesh and refines it based on the Doerfler strategy.
    """

    def __init__(
        self,
        name: str,
        marking_strategy: MarkingStrategy,
        marking_parameter: float,
        nref: int,
        accuracy: typing.Optional[float] = None,
    ):
        """Constructor

        Args:
            name:             The domain name
            marking_strategy: The marking strategy
            nref:             The number of refinements
            accuracy:         The accuracy, after which the adaptive procedure stops
        """

        # --- Initialise storage
        # The domain name
        self.name = name

        # The adaptive algorithm
        self.marking_strategy = marking_strategy
        self.marking_parameter = marking_parameter
        self.refinement_level_max = nref
        self.final_accuracy = accuracy
        self.refinement_level = 0

        # The boundary markers
        self.boundary_markers = []

    # --- The mesh definition ---
    def create(self, h: typing.Union[float, int, typing.List[int]]) -> Domain:
        """Create a meshed domain

        Args:
            h: The mesh size
        """
        raise NotImplementedError("Method not implemented")

    # --- Set boundary markers ---
    def mark_boundary(self, meshed_domain: mesh.Mesh) -> Domain:
        """Marks the boundary based on the initially defined boundary markers"""

        facet_indices, facet_markers = [], []

        for marker, locator in self.boundary_markers:
            facets = mesh.locate_entities(meshed_domain, 1, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full(len(facets), marker))

        facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
        facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
        sorted_facets = np.argsort(facet_indices)

        facet_functions = mesh.meshtags(
            meshed_domain, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
        )
        ds = ufl.Measure("ds", domain=meshed_domain, subdomain_data=facet_functions)

        return Domain(meshed_domain, facet_functions, ds)

    # --- Refine the mesh ---
    def refine(
        self,
        domain: Domain,
        eta_h: typing.Optional[fem.Function] = None,
        outname: typing.Optional[str] = None,
    ):
        """Refine the mesh based on Doerflers marking strategy

        Args:
            domain:   The domain
            eta_h:    The function of the cells error estimate
            outname:  The name of the output file for the mesh
                      (no output when not specified)
        """

        msh = domain.mesh
        ncells = msh.topology.index_map(2).size_global
        comm = msh.comm

        # Refine the mesh
        if np.isclose(self.marking_parameter, 1.0):
            # Refine entire mesh
            meshed_domain = mesh.refine(msh)
        else:
            if self.marking_strategy == MarkingStrategy.maximum:
                raise NotImplementedError("Maximum marking not implemented")
            else:
                # Check input
                if eta_h is None:
                    raise ValueError("Error estimate required!")

                # The total error (squared!)
                eta_total = np.sum(eta_h.array)

                # Cut-off
                cutoff = self.marking_parameter * eta_total

                # Sort cell contributions
                sorted_cells = np.argsort(eta_h.array)[::-1]

                # Create list of refined cells
                rolling_sum = 0.0
                breakpoint = ncells

                for i, e in enumerate(eta_h.array[sorted_cells]):
                    rolling_sum += e
                    if rolling_sum > cutoff:
                        breakpoint = i
                        break

                # List of refined cells
                refine_cells = np.array(
                    np.sort(sorted_cells[0 : breakpoint + 1]), dtype=np.int32
                )

            # Refine mesh
            edges = mesh.compute_incident_entities(self.mesh, refine_cells, 2, 1)
            meshed_domain = mesh.refine(self.mesh, edges)

        # Export mesh and error estimate into XDMF file
        if outname is not None:
            # Write mesh
            outname += "-mesh" + str(self.refinement_level) + "_error.xdmf"
            outfile = io.XDMFFile(comm, outname, "w")
            outfile.write_mesh(msh)

            # Write error estimate
            if eta_h is not None:
                outfile.write_function(eta_h, 0)

        # Update counter
        self.refinement_level += 1

        return self.mark_boundary(meshed_domain)


class EssntBC:
    def __init__(self, bc_is_strong: typing.Optional[bool] = True):
        # Identifier for strong essential BCs
        self.is_strong = bc_is_strong

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
            BCs as list of DirichletBCs (Default: []),
            BCs as additional terms for the residual (Default: None)
        """
        raise NotImplementedError("Method not implemented")
