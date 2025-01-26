from enum import Enum
import typing


# --- Supported models ---
class EqType(Enum):
    lin_elasticity = 0


# --- Collection of discretisation information ---
class DiscDict:
    def __init__(
        self,
        degree: typing.Union[int, typing.List[int]],
        fe_type: typing.Type[Enum],
        dual_space: typing.Optional[str],
        quadrature_degree: typing.Optional[int] = None,
    ):
        # The spatial discretisation
        self.fe_type = fe_type
        self.degree = degree
        self.dual_space = dual_space

        # Quadrature degree for RHS
        self.quadrature_degree = quadrature_degree

        # The error estimate
        self.ee_type = None
        self.degree_estress = None

    def specify_error_estimate(
        self, ee_type: typing.Type[Enum], degree: typing.Optional[int] = None
    ):
        if degree < self.degree:
            raise ValueError(
                "Degree of equilibrated stress must be at least '{self.degree}'!"
            )

        # Set the error estimate
        self.ee_type = ee_type

        # Set degree of equilibrated stress
        if degree is None:
            self.degree_estress = self.degree
        else:
            self.degree_estress = degree
