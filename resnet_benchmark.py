"""Compatibility entry point; implementation lives in amp.runner."""
import sys

from amp import runner

if __name__ == "__main__":
    runner.main()
else:
    # Preserve module identity so legacy callers can still patch runner settings.
    sys.modules[__name__] = runner
