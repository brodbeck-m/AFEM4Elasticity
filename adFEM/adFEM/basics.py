from enum import Enum
import typing


# --- Supported models ---
class EqType(Enum):
    lin_elasticity = 0


# --- Collection of discretisation information ---
def expnum_to_str(value: float) -> str:
    val_str = "{:.3e}".format(value).replace(".", "d")
    val_str = val_str.replace("e", "E")

    if value < 0:
        val_str = val_str.replace("-", "m")
    else:
        val_str = val_str.replace("+", "p")

    return val_str


class DiscDict:
    def __init__(
        self,
        degree: typing.Union[int, typing.List[int]],
        fe_type: typing.Type[Enum],
        ee_type: typing.Optional[typing.Type[Enum]] = None,
        dual_space: typing.Optional[str] = None,
        quadrature_degree: typing.Optional[int] = None,
    ):
        # The spatial discretisation
        self.fe_type = fe_type
        self.degree = degree
        self.dual_space = dual_space

        # Quadrature degree for RHS
        self.quadrature_degree = quadrature_degree

        # The error estimate
        self.ee_type = ee_type
        self.degree_eflux = None

    # --- Setter methods ---
    def specify_equilibration(self, degree: typing.Optional[int] = None):
        if degree < self.degree:
            raise ValueError(
                "Degree of equilibrated stress must be at least '{self.degree}'!"
            )

        if self.dual_space != "RT":
            raise ValueError("Equilibration only supported on RT spaces!")

        # Set degree of equilibrated stress
        if degree is None:
            self.degree_eflux = self.degree
        else:
            self.degree_eflux = degree

    # --- Getter methods ---
    def output_name(
        self,
        matpar: typing.Union[float, typing.List[float]],
        name_domain: str,
        marking_parameter: typing.Optional[float] = None,
    ) -> str:
        raise NotImplementedError(
            "This method must be implemented in the problem specific sub-class!"
        )
