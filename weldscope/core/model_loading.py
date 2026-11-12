"""Reuse downloaded model files before consulting the Hub."""


def load_cached_model(factory, model_name):
    """Prefer the disk cache; allow the normal download path on a cache miss.

    No process-wide offline flag: other sessions can still download missing models.
    Invalid arguments and programming errors are not retried as cache misses.
    """
    try:
        return factory(model_name, local_files_only=True)
    except OSError:
        return factory(model_name)
