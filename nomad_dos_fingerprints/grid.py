import numpy as np

class Grid():
    
    def __init__(self, grid_type = 'dg_cut', mu = -2, sigma = 7, cutoff = (-10,5), num_bins = 56):
        self.grid_type = grid_type
        self.mu = mu
        self.sigma = sigma
        self.cutoff = cutoff
        self.num_bins = num_bins


    def get_grid_id(self):
        id = ':'.join([self.grid_type, str(self.num_bins), str(self.mu), str(self.sigma), str(self.cutoff)])
        return id
