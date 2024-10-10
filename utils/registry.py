class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def inner_wrapper(wrapped_class):
            self._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    def get(self, name):
        if name in self._registry:
            return self._registry[name]
        else:
            raise ValueError(f"Class {name} is not registered.")

    def list_registered(self):
        return list(self._registry.keys())
