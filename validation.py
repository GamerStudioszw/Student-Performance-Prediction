"""Small server-side validators shared by entry points."""

import math
from numbers import Real
import unicodedata


def safe_text(value, *, field, minimum=1, maximum=80):
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError(f"{field} must contain {minimum}–{maximum} characters.")
    # Reject invisible controls, bidi overrides and invalid Unicode surrogates.
    if any(unicodedata.category(char).startswith("C") for char in value):
        raise ValueError(f"{field} must not contain control or invisible characters.")
    value = value.strip()
    if len(value) < minimum:
        raise ValueError(f"{field} must contain {minimum}–{maximum} characters.")
    return value


def finite_number(value, *, field, minimum, maximum):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not minimum <= value <= maximum or not math.isfinite(value)):
        raise ValueError(f"{field} must be a finite number from {minimum} to {maximum}.")
    return float(value)
