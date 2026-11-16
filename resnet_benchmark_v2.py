"""Isolated v2 entry point; the original benchmark remains unchanged."""
import sys

from amp import runner_v2 as runner

if __name__ == "__main__":
    runner.main()
else:
    # Preserve module identity so legacy callers can still patch runner settings.
    sys.modules[__name__] = runner
