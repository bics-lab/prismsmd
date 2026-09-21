"""PrisMSMD: mixed-solvent MD, from system building to probe density maps.

One run goes: build the cosolvent system, write the GROMACS inputs, run them, then turn
the trajectory into probe density maps (PMAP) and free-energy maps (GFE). An ensemble of
runs is folded together with :func:`combine_runs`.

    from prismsmd import run_seed, advance_run, run_maps, combine_runs

Deciding which runs exist, where they live and when to launch them is the caller's, not
this package's: every entry point here takes one run directory.
"""

__version__ = "0.1.0"

from .aggregate import aggregate, aggregate_runs, combine_replicates
from .config import (
    MDStep,
    ProbeConfig,
    ProtocolConfig,
    RunConfig,
    TargetConfig,
    load_probes,
    load_protocol,
    load_targets,
    load_yaml,
)
from .driver import (
    StepResult,
    advance_run,
    pbc_trajectory,
    postprocess_run,
    prepare_run,
    remove_pbc_trajectory,
)
from .events import count_consecutive_failures, read_events, record_event
from .gfe import GFEParams, pmap_to_gfe
from .grid import Grid, GridSpec, read_dx, write_dx
from .maps import RunMaps, combine_runs, run_maps, write_maps
from .patches import Patch, PatchSet, find_patches
from .provenance import stamp
from .runstate import RunStatus, run_status
from .seeds import (
    assert_no_seed_collision,
    placement_attempt,
    record_placement_attempt,
    run_seed,
    seed_for_attempt,
)

__all__ = [
    "GFEParams",
    "Grid",
    "GridSpec",
    "MDStep",
    "Patch",
    "PatchSet",
    "ProbeConfig",
    "ProtocolConfig",
    "RunConfig",
    "RunMaps",
    "RunStatus",
    "StepResult",
    "TargetConfig",
    "__version__",
    "advance_run",
    "aggregate",
    "aggregate_runs",
    "assert_no_seed_collision",
    "combine_replicates",
    "combine_runs",
    "count_consecutive_failures",
    "find_patches",
    "load_probes",
    "load_protocol",
    "load_targets",
    "load_yaml",
    "pbc_trajectory",
    "placement_attempt",
    "pmap_to_gfe",
    "postprocess_run",
    "prepare_run",
    "read_dx",
    "read_events",
    "record_event",
    "record_placement_attempt",
    "remove_pbc_trajectory",
    "run_maps",
    "run_seed",
    "run_status",
    "seed_for_attempt",
    "stamp",
    "write_dx",
    "write_maps",
]
