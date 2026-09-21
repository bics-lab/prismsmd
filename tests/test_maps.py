"""Invariants of naming and reporting one run's maps."""

import json
from typing import ClassVar

import numpy as np
import pytest

from prismsmd.config import ProbeConfig
from prismsmd.grid import Grid, GridSpec
from prismsmd.maps import (
    RunMaps,
    atom_labels,
    combine_runs,
    run_maps,
    write_maps,
)

SPEC = GridSpec(center=(0.0, 0.0, 0.0), dims=(4, 4, 4), spacing=1.0)


class Order:
    """Stands in for ProbeAtomOrder: only the names are used for labelling."""

    mol2_names: ClassVar[list[str]] = ["N1", "C2", "O1", "H1"]


def _probe(**kw):
    kw.setdefault("cid", "A22")
    kw.setdefault("name", "a22")
    kw.setdefault("smiles", "CC=O")
    kw.setdefault("atom_indices", {0: "N", 2: "O"})
    return ProbeConfig(**kw)


def _maps(probe):
    g = Grid(spec=SPEC, values=np.ones(SPEC.dims))
    idx = probe.mapped_atom_indices
    return RunMaps(
        spec=SPEC,
        pmaps=dict.fromkeys(idx, g),
        gfes=dict.fromkeys(idx, g),
        bulk_densities=dict.fromkeys(idx, 0.001),
        results={i: _FakeResult() for i in idx},
        excluded_volume=1000.0,
        valid_mask=np.ones(SPEC.dims, dtype=bool),
        labels=atom_labels(probe),
    )


class _FakeResult:
    def to_dict(self):
        return {"n_frames": 10, "n_atoms": 5}


def test_labels_come_from_the_probes_atom_names():
    assert atom_labels(_probe(atom_order=Order())) == {0: "N1", 2: "O1"}


def test_labels_fall_back_to_the_index_without_an_order():
    """A map is never written under a name that does not identify it."""
    assert atom_labels(_probe()) == {0: "atom0", 2: "atom2"}


def test_labels_cover_exactly_the_mapped_atoms():
    probe = _probe(atom_indices={1: "C", 3: "H"}, atom_order=Order())
    assert sorted(atom_labels(probe)) == probe.mapped_atom_indices


def test_a_probe_with_nothing_to_map_is_refused():
    with pytest.raises(ValueError, match="no mapped atoms"):
        run_maps("t.tpr", "t.xtc", "r.pdb", _probe(atom_indices={}))


def test_the_provenance_records_the_grid_and_the_bulk_density():
    probe = _probe(atom_order=Order())
    d = _maps(probe).to_dict()
    assert d["grid"]["dims"] == [4, 4, 4]
    assert d["valid_points"] == 64
    assert set(d["atoms"]) == {"N1", "O1"}
    assert d["atoms"]["N1"]["bulk_number_density"] == 1e-3


def test_each_atom_gets_its_own_files(tmp_path):
    """Two atoms writing the same name is how a map silently replaces another."""
    probe = _probe(atom_order=Order())
    written = write_maps(_maps(probe), tmp_path, probe)
    names = [p.name for p in written]
    assert len(names) == len(set(names))
    assert "PMAP_A22_N1.dx" in names
    assert "GFE_A22_O1.dx" in names


def test_the_manifest_is_written_beside_the_maps(tmp_path):
    probe = _probe(atom_order=Order())
    write_maps(_maps(probe), tmp_path, probe)
    d = json.loads((tmp_path / "maps_A22.json").read_text())
    assert d["excluded_volume"] == 1000.0


def test_either_kind_can_be_left_out(tmp_path):
    probe = _probe(atom_order=Order())
    written = write_maps(_maps(probe), tmp_path, probe, pmap=False)
    assert not any(p.name.startswith("PMAP_") for p in written)
    assert any(p.name.startswith("GFE_") for p in written)


def _run_maps_with(probe, value, bulk):
    g = Grid(spec=SPEC, values=np.full(SPEC.dims, float(value)))
    idx = probe.mapped_atom_indices
    return RunMaps(
        spec=SPEC, pmaps=dict.fromkeys(idx, g), gfes=dict.fromkeys(idx, g),
        bulk_densities=dict.fromkeys(idx, bulk),
        results={i: _FakeResult() for i in idx},
        excluded_volume=1000.0, valid_mask=np.ones(SPEC.dims, dtype=bool),
        labels=atom_labels(probe),
    )


def test_combining_runs_averages_the_densities():
    probe = _probe(atom_order=Order())
    runs = [_run_maps_with(probe, 1e-3, 2e-3), _run_maps_with(probe, 3e-3, 2e-3)]
    got = combine_runs(runs)
    assert np.allclose(got.pmaps[0].values, 2e-3)
    assert got.bulk_densities[0] == pytest.approx(2e-3)


def test_combining_converts_after_folding_not_before():
    """-RT ln is non-linear, so the two orders give different maps."""
    from prismsmd.aggregate import aggregate

    probe = _probe(atom_order=Order())
    runs = [_run_maps_with(probe, 1e-3, 2e-3), _run_maps_with(probe, 9e-3, 2e-3)]
    correct = combine_runs(runs, clip=False).gfes[0]
    wrong = aggregate([r.gfes[0] for r in runs], "ave")
    assert not np.allclose(correct.values, wrong.values)


def test_combining_refuses_runs_that_map_different_atoms():
    a = _run_maps_with(_probe(atom_indices={0: "N"}), 1e-3, 2e-3)
    b = _run_maps_with(_probe(atom_indices={1: "C"}), 1e-3, 2e-3)
    with pytest.raises(ValueError, match="maps atoms"):
        combine_runs([a, b])


def test_combining_refuses_a_missing_run():
    probe = _probe(atom_order=Order())
    with pytest.raises(ValueError, match="expected 20"):
        combine_runs([_run_maps_with(probe, 1e-3, 2e-3)], n_runs=20)


def test_combining_nothing_is_refused():
    with pytest.raises(ValueError, match="no runs"):
        combine_runs([])


def test_the_combined_provenance_keeps_every_run():
    probe = _probe(atom_order=Order())
    runs = [_run_maps_with(probe, 1e-3, 2e-3) for _ in range(3)]
    d = combine_runs(runs).to_dict()
    assert d["atoms"]["N1"]["n_runs"] == 3


# --- the two checks that catch a trajectory the protein was not re-imaged around ---
# Both are invisible in the maps: molecules are whole and the fit succeeds, only the
# density lands elsewhere. They are wired into run_maps, so they are exercised there.

def _grid(values):
    return Grid(spec=SPEC, values=np.asarray(values, dtype=float))


def test_a_density_that_never_reaches_the_protein_is_refused():
    """What -pbc mol without -center produces: the valid region is bulk solvent."""
    from prismsmd.pmap_cosolvkit import (
        ProbeDensityOffCentreError,
        assert_density_in_valid_region,
    )

    values = np.zeros(SPEC.dims)
    values[0, 0, 0] = 5.0                      # density, but not near the reference
    mask = np.zeros(SPEC.dims, dtype=bool)
    mask[2:, 2:, 2:] = True                    # the valid region is elsewhere
    with pytest.raises(ProbeDensityOffCentreError, match="pbc mol -center"):
        assert_density_in_valid_region(_grid(values), mask)


def test_a_density_that_does_reach_the_protein_passes():
    from prismsmd.pmap_cosolvkit import assert_density_in_valid_region

    values = np.ones(SPEC.dims)
    mask = np.ones(SPEC.dims, dtype=bool)
    assert assert_density_in_valid_region(_grid(values), mask) == 1.0


def test_an_off_centre_density_is_reported():
    from prismsmd.pmap_cosolvkit import assert_density_is_centred

    values = np.zeros(SPEC.dims)
    values[0, 0, 0] = 1.0
    said = []
    offset = assert_density_is_centred(_grid(values), max_offset_a=0.5,
                                       log=said.append)
    assert offset > 0.5
    assert said and "centroid" in said[0]


def test_a_centred_density_says_nothing():
    from prismsmd.pmap_cosolvkit import assert_density_is_centred

    said = []
    offset = assert_density_is_centred(_grid(np.ones(SPEC.dims)), log=said.append)
    assert offset == pytest.approx(0.0, abs=1e-9)
    assert said == []


def test_run_maps_reports_where_the_density_sits():
    """A check that is defined but never called is the same as no check."""
    import inspect

    from prismsmd import maps as maps_module

    assert "assert_density_is_centred(" in inspect.getsource(maps_module.run_maps)


def test_run_maps_does_not_reject_on_the_occupied_fraction():
    """It cannot: a short but correct run measures below the limit (0.48 % of 1 %)."""
    import inspect

    from prismsmd import maps as maps_module

    src = inspect.getsource(maps_module.run_maps)
    assert "assert_density_in_valid_region(" not in src.replace(
        "# assert_density_in_valid_region is deliberately NOT called here.", "")
