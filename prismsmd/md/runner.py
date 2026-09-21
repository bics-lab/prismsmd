"""Generation of the mdp files, the mdrun script and the trajectory
preprocessing script."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jinja2

from ..config import MDStep, ProtocolConfig

__all__ = [
    "STEP_TYPES",
    "render_mdp",
    "render_mdrun_sh",
    "render_postprocess_sh",
    "write_mdp_files",
    "write_mdrun_sh",
    "write_postprocess_sh",
]

_TEMPLATE_DIR = Path(__file__).parent / "templates"

_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    keep_trailing_newline=True,
    undefined=jinja2.StrictUndefined,
)

#: Supported MD stage types.
STEP_TYPES = ("minimization", "heating", "equilibration", "production")


def render_mdp(step: MDStep, protocol: ProtocolConfig, seed: int) -> str:
    """Return the mdp for one step as a string.

    An unset ``compressed_x_grps`` emits no mdp line, i.e. all atoms are written.
    """
    if step.type not in STEP_TYPES:
        raise ValueError(f"unknown step type {step.type!r}; must be one of {STEP_TYPES}")

    ctx = {
        "title": step.title or step.name,
        "define": step.define,
        "nsteps": step.nsteps,
        "nstlog": step.nstlog,
        "nstxtcout": step.nstxtcout,
        "nstenergy": step.nstenergy,
        "pbc": protocol.pbc,
        "dt": protocol.dt,
        "temperature": protocol.temperature,
        "pressure": protocol.pressure,
        "pcoupl": step.pcoupl,
        "seed": seed,
        "compressed_x_grps": protocol.compressed_x_grps or "",
    }
    if step.type == "heating":
        target = step.target_temp if step.target_temp is not None else protocol.temperature
        initial = step.initial_temp if step.initial_temp is not None else 0.0
        ctx.update(
            target_temp=target,
            initial_temp=initial,
            duration=step.nsteps * protocol.dt,
        )
    return _ENV.get_template(f"{step.type}.mdp.j2").render(**ctx)


def write_mdp_files(
    protocol: ProtocolConfig, outdir: str | Path, seed: int
) -> list[Path]:
    """Write one mdp per step into ``outdir`` and return the paths.

    ``seed`` is written into the mdp, so the same seed reproduces the same files.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for step in protocol.steps:
        path = outdir / f"{step.name}.mdp"
        path.write_text(render_mdp(step, protocol, seed))
        written.append(path)
    return written


def render_mdrun_sh(
    step_names: Sequence[str],
    *,
    top: str,
    gro: str,
    out_traj: str,
    gmx: str = "gmx",
    index_ndx: str = "index.ndx",
    hrt_sec: int = 86400,
    tail_sec: int = 3600,
    min_run_sec: int = 600,
    finished_list: str = "finished_step_list",
) -> str:
    """Return the mdrun driver script as a string.

    The script runs the steps in order, records completed steps in
    ``finished_list`` and resumes from there. ``hrt_sec`` is the wall-clock
    budget, ``tail_sec`` the part of it reserved for post-processing, and
    ``min_run_sec`` the shortest remaining time for which a new step is started.
    """
    if not step_names:
        raise ValueError("step_names is empty")
    gro_basename = Path(gro).name
    gro_basename = gro_basename.removesuffix(".gro")
    return _ENV.get_template("mdrun.sh.j2").render(
        step_names=" ".join(step_names),
        top=top,
        gro=gro,
        gro_basename=gro_basename,
        out_traj=out_traj,
        gmx=gmx,
        index_ndx=index_ndx,
        hrt_sec=hrt_sec,
        tail_sec=tail_sec,
        min_run_sec=min_run_sec,
        finished_list=finished_list,
    )


def write_mdrun_sh(path: str | Path, **kwargs) -> Path:
    """Write the mdrun script to ``path``, make it executable, return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_mdrun_sh(**kwargs))
    path.chmod(0o755)
    return path


def render_postprocess_sh(
    *,
    tpr: str,
    traj_in: str,
    traj_out: str,
    index_ndx: str = "index.ndx",
    group: str = "System",
    center_group: str = "Protein",
    gmx: str = "gmx",
    python: str = "python3",
) -> str:
    """Return the trajectory preprocessing script as a string.

    The script runs ``gmx trjconv -pbc cluster`` then ``-pbc mol -center`` on
    ``center_group``, writing ``traj_out``, and leaves a provenance marker beside
    it. Superposition is not done here; it is applied per frame when the PMAP is
    built.
    """
    from ..pmap_cosolvkit import PBC_MARKER_SUFFIX

    return _ENV.get_template("postprocess.sh.j2").render(
        tpr=tpr,
        traj_in=traj_in,
        traj_out=traj_out,
        index_ndx=index_ndx,
        group=group,
        center_group=center_group,
        gmx=gmx,
        python=python,
        marker_suffix=PBC_MARKER_SUFFIX,
    )


def write_postprocess_sh(path: str | Path, **kwargs) -> Path:
    """Write the preprocessing script to ``path``, make it executable, return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_postprocess_sh(**kwargs))
    path.chmod(0o755)
    return path
