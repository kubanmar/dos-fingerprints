import pytest, json, os
import numpy as np

from nomad_dos_fingerprints import Grid

@pytest.fixture
def grid():
    return Grid()

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
    assert (grid_array[min_index][0] >= -5) and (grid_array[min_index-1][0] < -5)
    assert (grid_array[max_index][0] <= 5) and  (grid_array[max_index+1][0] > 5)

    #creates the correct indices if energies are smaller than grid
    min_index, max_index =  get_indices_from_energy_limits(grid, -4, 2)
    assert (grid_array[min_index][0] >= -4) and (grid_array[min_index+1][0] > -4) and (grid_array[min_index-1][0] < -4)
    assert (grid_array[max_index][0] >= 2) and  (grid_array[max_index-1][0] < 2) and (grid_array[max_index+1][0] > 2)

    #creates the correct indices if energies match grid
    min_index, max_index =  get_indices_from_energy_limits(grid, -2, 2)
    grid = Grid().create(cutoff=(-2,2))
    assert (grid_array[min_index][0] <= -2) and (grid_array[min_index+1][0] > -2)
    assert (grid_array[max_index][0] >= 2) and  (grid_array[max_index-1][0] < 2)

def test_grid_from_lists(grid):

    expected_grid = [
        [1, [0.5, 1.0, 1.5, 2.0]],
        [2, [1.0, 2.0, 3.0, 4.0]],
        [3, [2.0, 4.0, 6.0, 8.0]]
    ]

    list_grid = grid.grid_from_lists([1,2,3], [2,4,8], 4)
    assert list_grid == expected_grid, "Got wrong grid from list"

def test_energy_intervals_from_function(grid):

    def test_function(value: float) -> int:
        return int(value)

    expected_output = [
        -16, -8, -4, -2, -1, 0, 1, 2, 4, 8
    ]
    
    assert grid.energy_intervals_from_function(test_function, 1, [-16,10]) == expected_output, "Wrong energy intervals calculated"

def test_grid_height_from_function(grid):

    def test_function(value: float) -> int:
        return int(value)

    expected_output = [
        2, 2, 10, 8
    ]

    assert grid.grid_height_from_function(test_function, [0.1, 1, 5, 4], 2) == expected_output, "Wrong bin heights calculated"

def test_grid_from_lists_regression(grid):
    grid = grid.create()
    reference = grid.grid()
    energies = [x[0] for x in reference]
    max_heights = [x[1][-1] for x in reference]

    assert grid.grid_from_lists(energies, max_heights, grid.num_bins) == reference, "Could not reproduce reference grid"

def test_gauss_function(grid):

    assert [grid.gauss_function(x, 1, 4, 0.4) for x in range(-5,6)] == [4, 4, 3, 3, 3, 1, 3, 3, 3, 4, 4], "Gauss function has changed unexpectedly"

def test_regression(grid):
    grid = grid.create(mu = 0)
    reference = grid.grid()
    energies = [x[0] for x in reference]
    max_heights = [x[1][-1] for x in reference]

    assert grid.grid_from_lists(energies, max_heights, grid.num_bins) == reference, "Could not reproduce reference grid"

    def step_sequencer(energy, grid = grid):
        return grid._step_sequencer(energy, grid.mu, grid.sigma, grid.original_stepsize)

    assert grid.energy_intervals_from_function(step_sequencer, grid.original_stepsize, [-10.6,5.05]) == energies, "Energy intervals from function does not return original intervals"

    def step_sequencer_2(energy, grid = grid):
        return grid._step_sequencer(energy, grid.mu, grid.sigma, 0.1)

    assert grid.grid_height_from_function(step_sequencer_2, energies, 1/112*56) == max_heights, "Bin heights are not the same as before" 

    reference_gauss = []
    new_gauss = []

    for e in energies:
        reference_gauss.append(grid._step_sequencer(e, grid.mu, grid.sigma, grid.original_stepsize))
        new_gauss.append(grid.gauss_function(e, 0.05, 1.05, grid.sigma))

    assert new_gauss == reference_gauss, "Functions do not return same values"

def test_new_grid(grid):
    grid = grid.create(grid_id='dg_cut:512:0:4:(-3, 3):1024')

    reference_grid = grid.grid()

    new_grid = grid.grid_new(0.05, 1.05, 0.25, 2.75, 4, 512, [-3, 3])

    new_grid_energies = [x[0] for x in new_grid]

    assert new_grid_energies == [x[0] for x in reference_grid], "Grid energies do not agree."

    assert new_grid == reference_grid, "Could not reproduce old grid."
