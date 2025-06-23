import pprint
import sys
from typing import Any


def dump(*args: Any, **kwargs: Any) -> None:
    """
    Pretty print variables for debugging without stopping execution

    Args:
        *args: Variables to dump
        **kwargs: Named variables to dump
    """
    pp = pprint.PrettyPrinter(indent=2)

    # Print each positional argument
    for arg in args:
        pp.pprint(arg)
        print()  # Add newline between dumps

    # Print each keyword argument with its name
    for name, value in kwargs.items():
        print(f"{name}:")
        pp.pprint(value)
        print()  # Add newline between dumps


def dd(*args: Any, **kwargs: Any) -> None:
    """
    Pretty print variables for debugging and stop execution (die and dump)

    Args:
        *args: Variables to dump
        **kwargs: Named variables to dump
    """
    dump(*args, **kwargs)
    sys.exit(1)  # Exit with error code 1