"""
Address-space ceilings shared by standalone analysis and source-copy workers.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
import resource


# %% Process resource ceiling
@dataclass(kw_only=True)
class ProcessMemory:
    """
    Bound a worker without attempting to raise an inherited hard limit.
    """

    memory_limit_bytes: int = 16 * 1024**3
    """
    Requested maximum address space for an isolated worker process.
    """

    def limit_address_space(self) -> None:
        """
        Apply the requested ceiling within the launcher's permitted address space.
        """
        inherited_hard_limit = resource.getrlimit(resource.RLIMIT_AS)[1]
        limit = self.memory_limit_bytes
        if inherited_hard_limit != resource.RLIM_INFINITY:
            limit = min(limit, inherited_hard_limit)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
