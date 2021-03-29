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
    fp = DOSFingerprint()   
    dummy_energy, dummy_dos = fp._integrate_to_bins(np.arange(-10,6,1), np.ones(16)) 
    e, d = fp._adapt_energy_bin_sizes(dummy_energy, dummy_dos, Grid.create(grid_id = fp.grid_id))
    grid = Grid.create(grid_id = fp.grid_id)
    grid_start, grid_end = grid.get_grid_indices_for_energy_range(dummy_energy)
    grid_array = grid.grid()
    # energy grid points are the same as in the Grid
    assert e[0] == grid_array[grid_start][0]
    assert e[-1] == grid_array[grid_end-1][0] # the last grid energy is the final border
    # the test case is integrated correctly
    cut_grid_array = grid_array[grid_start:grid_end+1] # +1: inclusion of the last point for "manual" integration below
    reference = [(cut_grid_array[idx+1][0] - cut_grid_array[idx][0]) for idx in range(len(cut_grid_array) -1)] 
    assert np.allclose(d, reference)
    # misc
    assert (grid_start, grid_end - 1) == grid.get_grid_indices_for_energy_range([np.round(x, 5) for x in e]) # e is 1 block shorter due to summation

def test_calc_grid_indices():
    fp = DOSFingerprint()   
    grid = Grid.create(grid_id = fp.grid_id)
    dummy_energy, dummy_dos = fp._integrate_to_bins(np.arange(-10,6,1), np.ones(16)) 
    e, _ = fp._adapt_energy_bin_sizes(dummy_energy, dummy_dos, Grid.create(grid_id = fp.grid_id))
    indices = fp._calc_grid_indices(e, grid)
    assert indices == fp.indices
    assert indices == list(grid.get_grid_indices_for_energy_range(e))

def test_calc_bit_vector():
    grid = Grid.create()
    fp = DOSFingerprint(grid_id = grid.get_grid_id())
    grid_array = grid.grid()
    fp.indices = [0, len(grid_array)-1]
    all_ones = fp._calc_bit_vector([max(grid_column[1]) for grid_column in grid_array], grid)
    assert all_ones == '1' * grid.num_bins * abs(fp.indices[1]+1 - fp.indices[0])
    print(grid.num_bins/2 - 1) 
    all_half_filled = fp._calc_bit_vector([grid_column[1][int(grid.num_bins/2 - 1)] for grid_column in grid_array], grid)
    assert all_half_filled == ('1' * int(grid.num_bins / 2) + '0' * int(grid.num_bins / 2)) * abs(fp.indices[1]+1 - fp.indices[0]) 