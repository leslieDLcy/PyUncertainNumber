class NominalValueMixin:
    @property
    def nominal_value(self):
        return self._compute_nominal_value()

    def _compute_nominal_value(self):
        raise NotImplementedError
