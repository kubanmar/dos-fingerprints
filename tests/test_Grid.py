import pytest
import json
import os

from nomad_dos_fingerprints import Grid
from nomad_dos_fingerprints.grid import NotCreatedError

@pytest.fixture
def grid():
    return Grid()

with open(os.path.join(os.path.dirname(__file__), 'grid_test.json'), 'r') as test_data_file:
    test_grid_data = json.load(test_data_file)

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

def test_gauss_function(grid):

    assert [grid.gauss_function(x, 1, 4, 0.4) for x in range(-5,6)] == [4, 4, 3, 3, 3, 1, 3, 3, 3, 4, 4], "Gauss function has changed unexpectedly"

def test_gen_grid_id_v2(grid):
    grid = grid.create(grid_type = "nonuniform", e_ref = 10, delta_e_min = -2, delta_e_max = 9, delta_rho_min = 0.1, delta_rho_max = 10, width = 0.7, cutoff = [-20,30], n_pix = 2)
    assert grid.get_grid_id() == "nonuniform:10:-2:9:0.1:10:0.7:-20:30:2", "Wrong grid id created for v2"

def test_resolve_grid_id_v2(grid):
    expected = {
        "grid_type" : "nonuniform",
        "e_ref" : 10,
        "delta_e_min" : -2,
        "delta_e_max" : 9, 
        "delta_rho_min" : 0.1, 
        "delta_rho_max" : 10, 
        "width" : 0.7, 
        "cutoff" : [-20,30], 
        "n_pix" : 2
    }
    assert grid.resolve_grid_id("nonuniform:10:-2:9:0.1:10:0.7:-20:30:2") == expected, "Resolving grid id for v2 failed"

def test_fails_for_not_created_grids(grid):
    with pytest.raises(NotCreatedError):
        grid.get_grid_id()

def test_custom_grid(grid):
    with pytest.raises(ValueError):
        grid = grid.create(grid_type = "custom", states_discretization = [1,2,3], n_pix = 2)
    with pytest.raises(ValueError):
        grid = grid.create(grid_type = "custom", energy_discretization = [0,1,2], n_pix = 2)
    grid = grid.create(grid_type = "custom", energy_discretization = [0,1,2], states_discretization = [1,2,3], n_pix = 2)
    grid2 = grid.create(grid_type = "custom", energy_discretization = [0,1,3], states_discretization = [1,2,3])
    grid3 = grid.create(grid_type = "custom", energy_discretization = [0,1,3], states_discretization = [1,2,3], n_pix = 2)

    assert grid.grid() == [[0, [0.5,1]], [1, [1,2]], [2, [1.5, 3.0]]], "Did not create correct custom grid"

    assert grid2.get_grid_id() != grid3.get_grid_id()

def test_uniform_grid(grid):
    grid = grid.create(grid_type = "uniform", delta_e_min = 1, delta_rho_min = 1, cutoff = [-1,1], n_pix = 4)

    expected_uniform_grid = [
        [-1, [0.25,0.5,0.75,1]],
        [0, [0.25,0.5,0.75,1]],
        [1, [0.25,0.5,0.75,1]]
    ]

    assert grid.grid() == expected_uniform_grid, "Create wrong uniform grid"