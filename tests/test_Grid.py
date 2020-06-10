import pytest
import numpy as np

from nomad_dos_fingerprints import Grid

def test_gen_grid_id():
    grid = Grid(mu = -5, sigma = 10, grid_type = 'dg_cut', num_bins = 64, cutoff = (-15,10))
    assert grid.get_grid_id() == 'dg_cut:64:-5:10:(-15, 10)'