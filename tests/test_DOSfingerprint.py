import numpy as np
import pytest

from nomad_dos_fingerprints import tanimoto_similarity
from nomad_dos_fingerprints import DOSFingerprint, Grid
from nomad_dos_fingerprints.DOSfingerprint import ELECTRON_CHARGE

def test_integrate_to_bins():

    def get_area_below_curve(x, func, fp):
        y = [func(x_) for x_ in x]
        _, dos = fp._integrate_to_bins(x, y, stepsize=0.001)
        return sum(dos)

    def dummy_function(x):
        return np.sin(x)
    
    test_data_x = np.linspace(0, np.pi, num = 10000)
    test_data_y = [dummy_function(x) for x in test_data_x]
    fp = DOSFingerprint()
    energies, dos = fp._integrate_to_bins(test_data_x, test_data_y, stepsize=0.01)
    assert 0 in energies
    assert np.isclose(get_area_below_curve([-0.0009,1.0009], lambda _: 1, fp,), 1, atol=1e-12), "integration area was not cut correctly"
    assert np.isclose(sum(dos), 2, atol=1e-3), "Calculated area is not correct"
    assert np.isclose(get_area_below_curve(np.linspace(-np.pi, 0, num = 10000), dummy_function, fp), -2, atol=1e-4), "Calculated area is not correct"
    assert len(energies) == len(dos)
    for idx, e in enumerate(energies):
        assert np.isclose(e, idx * 0.01), "Error in calculating integration intervals"

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
    fp = DOSFingerprint(stepsize=0.001).calculate(test_data_x, test_data_y)
    print(fp)
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
    assert all_ones == '1' * grid.n_pix * abs(fp.indices[1]+1 - fp.indices[0])
    print(grid.n_pix/2 - 1) 
    all_half_filled = fp._calc_bit_vector([grid_column[1][int(grid.n_pix/2 - 1)] for grid_column in grid_array], grid)
    assert all_half_filled == ('1' * int(grid.n_pix / 2) + '0' * int(grid.n_pix / 2)) * abs(fp.indices[1]+1 - fp.indices[0]) 

def test_compress_bit_string():
    fp = DOSFingerprint()
    compressed = fp._compress_binary_fingerprint_string("111000011")
    assert compressed == "3t4f2t", "Compression is not correct"

def test_expand_bit_string():
    fp = DOSFingerprint()
    expanded = fp._expand_fingerprint_string("5t3f6t")
    assert expanded == "11111000111111", "Expandsion of compressed bins failed"

def test_get_similarity():
    x = np.round(np.linspace(0,1.1, num=1000), 8)
    grid = Grid.create(grid_type="uniform", e_ref=0.5, cutoff=[-0.5,0.5], delta_e_min=0.01, delta_rho_min=0.015, n_pix=10)
    fp_a = DOSFingerprint().calculate(x, [1 if x_i <= 0.75 else 0 for x_i in x], grid_id = grid.get_grid_id())
    fp_b = DOSFingerprint().calculate(x, [1 if x_i > 0.25 else 0 for x_i in x], grid_id = grid.get_grid_id())
    assert fp_a.get_similarity(fp_b) == 0.5, "Similarity obtained from get_similarity is wrong"

@pytest.mark.skip()
def test_get_bitarray():
    raise NotImplementedError("TODO: Implement")