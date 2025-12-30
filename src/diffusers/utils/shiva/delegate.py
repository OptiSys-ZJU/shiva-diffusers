import functools


def delegated_forward(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        hook = getattr(self, "_hook", None)
        if hook is not None:
            delegated = getattr(hook, "delegated_call", None)
            if callable(delegated):
                return delegated(func, self, *args, **kwargs)
        # fallback
        return func(self, *args, **kwargs)

    return wrapper


class BaseDelegator:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    """Default delegator: directly calls the original function."""
    def delegated_call(self, func, module_self, *args, **kwargs):
        return func(module_self, *args, **kwargs)
