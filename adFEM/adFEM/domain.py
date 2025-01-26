import typing
import numpy as np

from dolfinx import fem, io, mesh
import ufl


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

    def __init__(self, name: str):
        """Constructor

        Args:
            name: The domain name
        """

        # --- Initialise storage
        # The domain name
        self.name = name

        # The mesh counter
        self.refinement_level = 0

        # The boundary markers
        self.boundary_markers = []

    # --- Generate the mesh ---
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

    def create(self, h: typing.Union[float, typing.List[int]]) -> Domain:
        """Create a meshed domain

        Args:
            h: The mesh size
        """
        raise NotImplementedError("Method not implemented")

    def refine(
        self,
        doerfler: float,
        domain: Domain,
        eta_h: typing.Optional[fem.Function] = None,
        outname: typing.Optional[str] = None,
    ):
        """Refine the mesh based on Doerflers marking strategy

        Args:
            doerfler: The Doerfler parameter
            domain:   The domain
            eta_h:    The function of the cells error estimate
            outname:  The name of the output file for the mesh
                      (no output when not specified)
        """

        msh = domain.mesh
        ncells = msh.topology.index_map(2).size_global
        comm = msh.comm

        # Refine the mesh
        if np.isclose(doerfler, 1.0):
            # Refine entire mesh
            meshed_domain = mesh.refine(msh)
        else:
            # Check input
            if eta_h is None:
                raise ValueError("Error estimate required!")

            # The total error (squared!)
            eta_total = np.sum(eta_h.array)

            # Cut-off
            cutoff = doerfler * eta_total

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
