import pytest, json, os
import numpy as np

from nomad_dos_fingerprints import Grid

with open(os.path.join(os.path.dirname(__file__), 'grid_test.json'), 'r') as test_data_file:
    test_grid_data = json.load(test_data_file)

def test_gen_grid_id():
    grid = Grid.create(mu = -5, sigma = 10, grid_type = 'dg_cut', num_bins = 64, cutoff = (-15,10))
    assert grid.get_grid_id() == 'dg_cut:64:-5:10:(-15, 10):56'

def test_grid():
    grid = Grid().create()
    assert grid.grid() == list(test_grid_data.values())[0]

def test_generate_grid_from_id():
    grid = Grid().create()
    grid_ids = Grid().create(grid_id = 'dg_cut:56:-2:7:(-10, 5)')
    assert grid.grid() == grid_ids.grid()
    
def test_get_grid_indices_for_energy_range():
    grid = Grid().create(cutoff=(-5,5))
    grid_array = grid.grid()

    def get_indices_from_energy_limits(grid, min_e, max_e):
        energies = np.array(range(min_e,max_e+1))
        return grid.get_grid_indices_for_energy_range(energies)

    #creates the correct indices if energies are larger than grid
    min_index, max_index =  get_indices_from_energy_limits(grid, -10,10)
    assert grid_array[min_index][0] == min([x[0] for x in grid_array])
    assert grid_array[max_index][0] == max([x[0] for x in grid_array])

    #creates the correct indices if energies are smaller than grid
    min_index, max_index =  get_indices_from_energy_limits(grid, -4, 2)
    assert (grid_array[min_index][0] <= -4) and (grid_array[min_index+1][0] > -4)
    assert (grid_array[max_index][0] >= 2) and  (grid_array[max_index-1][0] < 2)

    #creates the correct indices if energies match grid
    min_index, max_index =  get_indices_from_energy_limits(grid, -5, 5)
    assert (grid_array[min_index][0] <= -5) and (grid_array[min_index+1][0] > -5)
    assert (grid_array[max_index][0] >= 5) and  (grid_array[max_index-1][0] < 5)
