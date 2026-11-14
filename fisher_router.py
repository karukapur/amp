"""Compatibility import for the implementation in amp.fisher_router."""
import sys

from amp import fisher_router

sys.modules[__name__] = fisher_router
