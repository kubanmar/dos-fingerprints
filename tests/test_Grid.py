import pytest, json, os
import numpy as np

from nomad_dos_fingerprints import Grid

with open(os.path.join(os.path.dirname(__file__), 'grid_test.json'), 'r') as test_data_file:
    test_grid_data = json.load(test_data_file)

def test_gen_grid_id():
    grid = Grid().create(mu = -5, sigma = 10, grid_type = 'dg_cut', num_bins = 64, cutoff = (-15,10))
    assert grid.get_grid_id() == 'dg_cut:64:-5:10:(-15, 10)'

def test_grid():
    grid = Grid().create()
    assert grid.grid() == list(test_grid_data.values())[0]

def test_generate_grid_from_id():
    grid = Grid().create()
    grid_ids = Grid().create(grid_id = 'dg_cut:56:-2:7:(-10, 5)')
    assert grid.grid() == grid_ids.grid()