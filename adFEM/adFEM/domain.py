from enum import Enum
import typing
import meshio
from mpi4py import MPI
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
    Create an initial mesh and refines it based on a Doerfler or Maximum marking strategy.
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
            name:              The domain name
            marking_strategy:  The marking strategy
            marking_parameter: The parameter of the underlying marking strategy
            nref:              The number of refinements
            accuracy:          The accuracy, after which the adaptive procedure stops
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

    def prepare_mesh_for_eqlb(self, meshed_domain: mesh.Mesh) -> mesh.Mesh:
        """Prepare mesh for flux equilibration
        The current implementation of the flux equilibration requires at least two cells
        linked to each boundary node. This routines modifies meshes, such that this requi-
        rement is met.

        Args:
            meshed_domain: The mesh of the domain

        Returns:
            The (optionally adjusted) mesh
        """
        # List of refined cells
        refined_cells = []

        # Required connectivity's
        meshed_domain.topology.create_connectivity(0, 2)
        meshed_domain.topology.create_connectivity(1, 2)
        pnt_to_cell = meshed_domain.topology.connectivity(0, 2)

        # The boundary facets
        bfcts = mesh.exterior_facet_indices(meshed_domain.topology)

        # Get boundary nodes
        V = fem.FunctionSpace(meshed_domain, ("Lagrange", 1))
        bpnts = fem.locate_dofs_topological(V, 1, bfcts)

        # Check if point is linked with only on cell
        for pnt in bpnts:
            cells = pnt_to_cell.links(pnt)

            if len(cells) == 1:
                refined_cells.append(cells[0])

        list_ref_cells = list(set(refined_cells))  # remove duplicates

        if len(list_ref_cells) > 0:
            # Add central node into refined cells
            x_new = np.copy(meshed_domain.geometry.x[:, 0:2])  # remove third component
            cells_new = np.copy(meshed_domain.geometry.dofmap.array).reshape(-1, 3)
            cells_add = np.zeros((2, 3), dtype=np.int32)

            list_ref_cells.sort()
            for i, c_init in enumerate(list_ref_cells):
                # The cell
                c = c_init + 2 * i

                # Nodes on cell
                cnodes = cells_new[c, :]
                x_cnodes = x_new[cnodes]

                # Coordinate of central node
                node_central = (1 / 3) * np.sum(x_cnodes, axis=0)

                # New node coordinates
                id_new = max(cnodes) + 1
                x_new = np.insert(x_new, id_new, node_central, axis=0)

                # Adjust definition of existing cells
                cells_new[cells_new >= id_new] += 1

                # Add new cells
                cells_add[0, :] = [cells_new[c, 1], cells_new[c, 2], id_new]
                cells_add[1, :] = [cells_new[c, 2], cells_new[c, 0], id_new]
                cells_new = np.insert(cells_new, c + 1, cells_add, axis=0)

                # Correct definition of cell c
                cells_new[c, 2] = id_new

            # Update mesh
            return mesh.create_mesh(
                MPI.COMM_WORLD,
                cells_new,
                x_new,
                ufl.Mesh(
                    ufl.VectorElement(
                        "Lagrange", ufl.Cell("triangle", geometric_dimension=2), 1
                    )
                ),
            )
        else:
            return meshed_domain

    # --- Set boundary markers ---
    def mark_boundary(self, meshed_domain: mesh.Mesh) -> Domain:
        """Marks boundary facets

        Factes are tagged based on initially defined boundary markers.

        Args:
            meshed_domain: The mesh of the domain

        Returns:
            The mesh-dependent representation of the domain
        """

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
        """Refine the mesh based on the specified marking strategy

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


class AdaptiveDomainAbaqus(AdaptiveDomain):
    """An adaptive domain for ABAQUS input files

    Create an initial mesh based on a .inp file from ABAQUS. The ABAQUS export has to be done
    using first order Lagrange elements. The domain is then refined based on a Doerfler or a
    Maximum marking strategy.
    """

    def __init__(
        self,
        name: str,
        path_to_inp: str,
        boundaries: typing.List[typing.Callable],
        marking_strategy: MarkingStrategy,
        marking_parameter: float,
        nref: int,
        accuracy: typing.Optional[float] = None,
    ):
        """Constructor

        Args:
            name:              The domain name
            path_to_inp:       The path to the .inp file
            boundaries:        The boundary markers
            marking_strategy:  The marking strategy
            marking_parameter: The parameter of the underlying marking strategy
            nref:              The number of refinements
            accuracy:          The accuracy, after which the adaptive procedure stops
        """

        # The patch to the .inp file
        self.path_to_inp = path_to_inp

        # Constructor of super class
        super().__init__(name, marking_strategy, marking_parameter, nref, accuracy)

        # Set boundary markers
        for n, bf in enumerate(boundaries):
            self.boundary_markers.append((n + 1, bf))

    # --- The mesh definition ---
    def create(self, h: typing.Union[float, int, typing.List[int]]) -> Domain:
        """Create a meshed domain

        Args:
            eqlb_conf_mesh: True, if at least 2 cells have to be connected to an boundary node
        """

        # Read ABAQUS input
        inp = meshio.read(self.path_to_inp)

        # Recreate mesh
        meshed_domain = mesh.create_mesh(
            MPI.COMM_WORLD,
            inp.cells[0].data,
            inp.points,
            ufl.Mesh(
                ufl.VectorElement(
                    "Lagrange", ufl.Cell("triangle", geometric_dimension=2), 1
                )
            ),
        )

        # Check mesh (at least 2 cells per node)
        return self.mark_boundary(meshed_domain)

    # --- Set boundary markers ---
    def mark_boundary(self, meshed_domain: mesh.Mesh) -> Domain:
        """Marks boundary facets

        Factes are tagged based on either initially defined n
        boundary markers or - if not mentioned - with n+1.

        Args:
            meshed_domain: The mesh of the domain

        Returns:
            The mesh-dependent representation of the domain
        """

        facet_indices, facet_markers = [], []

        # Handle tagged factes
        for marker, locator in self.boundary_markers:
            facets = mesh.locate_entities(meshed_domain, 1, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full(len(facets), marker))

        # Handle remaining factes
        meshed_domain.topology.create_connectivity(1, 2)

        facet_indices.append(
            np.setdiff1d(
                mesh.exterior_facet_indices(meshed_domain.topology),
                np.array(np.hstack(facet_indices), dtype=np.int32),
            )
        )
        facet_markers.append(
            np.full(len(facet_indices[-1]), len(self.boundary_markers) + 1)
        )

        # Set facet markers
        facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
        facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
        sorted_facets = np.argsort(facet_indices)

        facet_functions = mesh.meshtags(
            meshed_domain, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
        )
        ds = ufl.Measure("ds", domain=meshed_domain, subdomain_data=facet_functions)

        return Domain(meshed_domain, facet_functions, ds)


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
