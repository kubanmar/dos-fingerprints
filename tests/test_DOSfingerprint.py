import pytest
import numpy as np

from nomad_dos_fingerprints import DOSFingerprint, tanimoto_similarity, Grid
from nomad_dos_fingerprints.DOSfingerprint import ELECTRON_CHARGE

def test_integrate_to_bins():
    test_data_x = np.linspace(0, np.pi, num = 1000)
    test_data_y = [np.sin(x) for x in test_data_x]
    fp = DOSFingerprint(stepsize=0.001)
    energies, dos = fp._integrate_to_bins(test_data_x, test_data_y)
    energies_test_data = []
    current_energy = 0
    assert 0 in energies
    while current_energy < np.pi - fp.stepsize:
        energies_test_data.append(current_energy)
        current_energy += fp.stepsize
    assert np.isclose(sum(dos), 2)
    assert len(energies) == len(dos)
    assert (np.isclose(energies, np.array(energies_test_data))).all()

def test_convert_dos():
    test_data_x = np.arange(1, 5, step = 0.01)
    test_data_y = test_data_x
    fp = DOSFingerprint(stepsize=0.001)
    x, y = fp._convert_dos(test_data_x* ELECTRON_CHARGE, [test_data_y/2 / ELECTRON_CHARGE, test_data_y/2 / ELECTRON_CHARGE])
    assert np.isclose(x,test_data_x).all()
    assert np.isclose(y, test_data_y).all()
    x, y = fp._convert_dos(test_data_x* ELECTRON_CHARGE, [test_data_y/ ELECTRON_CHARGE])
    assert np.isclose(x,test_data_x).all()
    assert np.isclose(y, test_data_y).all()

def test_serialization():
    test_data_x = np.linspace(0, np.pi, num = 1000)
    test_data_y = [np.sin(x) for x in test_data_x]
    fp = DOSFingerprint(stepsize=0.001).calculate([x * ELECTRON_CHARGE for x in test_data_x], [[x / ELECTRON_CHARGE for x in test_data_y]])
    fp_json = fp.to_dict()
    fp_again = DOSFingerprint().from_dict(fp_json)
    assert tanimoto_similarity(fp, fp_again) == 1

def test_adapt_energy_bin_sizes():
    dummy_energy = np.arange(-10,5.1,0.1)
    dummy_dos = np.ones(len(dummy_energy))
    fp = DOSFingerprint()   
    e, d = fp._adapt_energy_bin_sizes(dummy_energy, dummy_dos)
    grid = Grid.create(grid_id = fp.grid_id)
    grid_ids = grid.get_grid_indices_for_energy_range(e)
    assert e[0] == grid.grid()[grid_ids[0]][0]
    assert e[-1] == grid.grid()[grid_ids[1]][0]
    assert d == [0.45, 0.4, 0.4, 0.35, 0.3, 0.3, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.2, 0.2, 0.2, 0.2, 0.25, 0.25, 0.25, 0.3, 0.3]
    ## Instead of hardcoding that, find a functional relationship and do the thing properly