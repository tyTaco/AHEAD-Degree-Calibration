from .time_encoding import TimeEncode


class TimeEncoder(TimeEncode):
    """Compatibility interface for existing model imports."""

    def __init__(self, time_dim, parameter_requires_grad=True):
        super().__init__(dimension=time_dim)

        for parameter in self.parameters():
            parameter.requires_grad_(parameter_requires_grad)
