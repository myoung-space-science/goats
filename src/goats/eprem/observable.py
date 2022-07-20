from goats.core import axis
from goats.core import observable
from goats.core import variable
from goats.eprem import parameters


class Interface(observable.Interface):
    """Interface to all EPREM observable quantities."""

    def __init__(
        self,
        axes: axis.Interface,
        variables: variable.Interface,
        arguments: parameters.Arguments,
    ) -> None:
        self.axes = axes
        self.variables = variables
        self.arguments = arguments
        # Use code from `observables.Observables.__init__`
        available = []
        super().__init__(*available)

