"""Utility functions."""

from ast import literal_eval


def parse_param_str(param_str):
    """Parse a parameter string (colon-separated list of equals-separated key-value pairs) into a dict.

    All keys are assumed to be strings, while values are evaluated as Python literals.
    """
    param_kwargs = {}
    if param_str:
        for pair in param_str.split(':'):
            key, value = pair.split('=')
            try:
                param_kwargs[key] = literal_eval(value)
            except (SyntaxError, ValueError):
                param_kwargs[key] = value
    return param_kwargs
