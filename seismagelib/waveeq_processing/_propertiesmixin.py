
class PhysicalPropertiesMixin:
    """
    Mixin that provides access to various physical properties such as 'vs',
    'rho', 'lambda', etc., typically used in wave propagation or
    geophysical modeling contexts.

    This mixin adds property accessors (e.g., 'self.vs', 'self.rho', etc.)
    that internally call the `_get_property(attr_name)` method to retrieve
    and process model parameters.

    Requirements:
    -------------
    The class that inherits from this mixin must define the following:

    - `self.model`: An object that exposes the required physical attributes
       as attributes.

    - `self._crop_model(data, nbl)`: A method that takes in the raw data
       and the model's boundary size (`nbl`) and returns the appropriately
       cropped data.
    """
    _physical_attributes = {
        "vp", "vs", "rho", "lam", "mu", "Ip", "Is",
        "delta", "epsilon", "phi", "gamma",
        "C11", "C22", "C33", "C44", "C55", "C66",
        "C12", "C21", "C13", "C31", "C23", "C32"
    }

    def _get_property(self, attr_name):
        attr = getattr(self.model, attr_name, None)
        if attr is not None:
            return self._crop_model(attr.data, self.model.nbl)
        raise AttributeError(
            f"{type(self).__name__} doesn't have parameter {attr_name}")

    def __getattr__(self, name):
        if name in self._physical_attributes:
            return self._get_property(name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
