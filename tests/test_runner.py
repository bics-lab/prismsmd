"""Invariants of the generated run script."""

import re

from prismsmd.md.runner import render_mdrun_sh

STEPS = ["min1", "heat", "equil1", "prod"]


def _script(**kwargs):
    return render_mdrun_sh(
        STEPS, top="input.top", gro="input.gro", out_traj="prod.xtc", **kwargs
    )


def _mdrun_line(script):
    return next(line for line in script.splitlines() if "$GMX mdrun" in line)


def test_every_placeholder_is_filled():
    assert "{{" not in _script()


def test_the_device_and_pin_options_reach_the_command():
    line = _mdrun_line(_script())
    assert "$GPU_OPT" in line
    assert "$PIN_OPT" in line


def test_pinning_is_off_unless_an_offset_is_given():
    """The offset depends on the machine and on what the scheduler handed out."""
    block = re.search(r'PIN_OPT=""(.|\n)*?\nfi', _script()).group(0)
    assert 'PIN_OPT=""' in block
    assert "-pin on" in block
    assert "${PIN_OFFSET" in block


def test_the_stride_defaults_to_one():
    assert "${PIN_STRIDE:-1}" in _script()


def test_the_script_computes_no_placement_itself():
    """Nothing here may read the topology of the machine; the values are passed in."""
    script = _script()
    for probe in ("/proc/self/status", "local_cpulist", "nvidia-smi", "lscpu",
                  "numactl", "Cpus_allowed_list"):
        assert probe not in script


def test_the_device_selection_is_only_dropped_when_this_script_owns_it():
    """A caller that pinned its GPU without GPU_ID keeps its own selection."""
    script = _script()
    block = re.search(r'GPU_OPT=""(.|\n)*?\nfi', script).group(0)
    assert "unset CUDA_VISIBLE_DEVICES" in block
    assert script.count("unset CUDA_VISIBLE_DEVICES") == 1


def test_the_thread_option_is_chosen_from_the_build_not_the_name():
    """No single option bounds the threads on both builds, so the binary is asked."""
    script = _script()
    assert '--version' in script
    assert "thread_mpi" in script
    assert "-nt ${ncpus}" in script          # thread-MPI: bounds the total
    assert "-ntomp ${ncpus}" in script       # MPI: the only one accepted
    assert "$NT_OPT" in _mdrun_line(script)


def test_the_thread_option_is_not_hard_coded_on_the_command():
    """A fixed -nt breaks the MPI build; a fixed -ntomp fills the host on thread-MPI."""
    line = _mdrun_line(_script())
    assert "-nt " not in line
    assert "-ntomp" not in line
