# PrisMSMD

Mixed-solvent molecular dynamics: builds a cosolvent system, writes the GROMACS inputs
that run it, and turns the resulting trajectory into probe density maps (PMAP) and
free-energy maps (GFE).

## Install

```sh
pip install -e ".[system]"
```

The base install is enough to read and write maps and to render the MD inputs. The
`system` extra adds CosolvKit, OpenMM, RDKit and MDAnalysis, which system building and
map generation need. MD itself runs in GROMACS, which is called as an external command
(`MSMD_GMX`, or `gmx` on `PATH`).

## Pipeline

| Stage | Module | What it produces |
|---|---|---|
| Seeds | `prismsmd.seeds` | a rebuildable seed per run, and the retry rule |
| System building | `prismsmd.system.builder` | `input.top`, `input.gro`, `index.ndx` |
| MD inputs | `prismsmd.md.runner` | `*.mdp`, `mdrun.sh`, `postprocess.sh` |
| Running one run | `prismsmd.driver` | advances a run as far as the time allows |
| Progress | `prismsmd.runstate` | how far a run got, read off disk |
| Record | `prismsmd.events` | append-only log of what happened |
| Maps of one run | `prismsmd.maps` | PMAP and GFE per mapped probe atom |
| Maps of an ensemble | `prismsmd.maps.combine_runs` | the runs folded, then converted once |

The low-level pieces the stages are built from -- `grid`, `gfe`, `aggregate`,
`volume`, `pmap_cosolvkit` -- are public too, for an application that needs to
assemble something else.

`prismsmd.config` reads the three configuration files the stages share: `probes.yaml`
(probe definitions and the atoms to map), `targets.yaml` (per-target PDB and docking
box) and a protocol yaml (the MD sequence, restraint ladder and repulsion parameters).

## Usage

One run, from nothing to its maps:

```python
from prismsmd import RunConfig, advance_run, run_maps, run_seed, write_maps

cfg = RunConfig.load(
    target_key="example",
    targets_yaml="configs/targets.yaml",
    probes_yaml="configs/probes.yaml",
    protocol_yaml="configs/protocol.yaml",
)
probe = cfg.probes_in_use()[0]

# Build, write the MD inputs and run, as far as the time allows. Idempotent:
# calling it again resumes rather than restarts.
advance_run("run00", cfg.target, probe, cfg.protocol, run=0,
            seconds_remaining=3600, ntomp=4)

# Then the periodic-boundary handling, and the maps.
#   sh run00/postprocess.sh
maps = run_maps("run00/pr.tpr", "run00/production_pbcmol.xtc",
                cfg.target.pdb, probe)
write_maps(maps, "run00/maps", probe)
```

An ensemble is the same call per run and one fold at the end. The loop is yours:

```python
from prismsmd import combine_runs

per_run = [run_maps(f"run{r:02d}/pr.tpr", f"run{r:02d}/production_pbcmol.xtc",
                    cfg.target.pdb, probe) for r in range(20)]
ensemble = combine_runs(per_run, method="max", n_runs=20)
write_maps(ensemble, "maps", probe)
```

`combine_runs` folds the densities and converts once, which is the only correct
order: `-RT ln` is non-linear, so averaging the runs' finished GFE grids gives a
different answer.

Each run's placement seed comes from `run_seed(target, probe, run)`, so the same run
always rebuilds the same system; a placement that cannot be minimised is retried with
`seed_for_attempt`.

## License

MIT. See `LICENSE`.
