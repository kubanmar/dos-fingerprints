import numpy as np

class Grid():
    
    @staticmethod
    def create(grid_id = None, grid_type = 'dg_cut', mu = -2, sigma = 7, cutoff = (-10,5), num_bins = 56, original_stepsize = 0.05):
        if grid_id == None:
            self = Grid()
            self.grid_type = grid_type
            self.mu = mu
            self.sigma = sigma
            self.cutoff = cutoff
            self.num_bins = num_bins
            self.original_stepsize = original_stepsize
            self.grid_id = self.get_grid_id()
        else:
            self = Grid().create(**Grid().resolve_grid_id(grid_id))
        return self

    def get_grid_id(self):
        id = ':'.join([self.grid_type, str(self.num_bins), str(self.mu), str(self.sigma), str(self.cutoff)])
        return id

    @staticmethod
    def resolve_grid_id(grid_id):
        grid_type, num_bins, mu, sigma, cutoff = grid_id.split(':')
        return {'grid_type' : grid_type, 'num_bins' : int(num_bins), 'mu' : float(mu), 'sigma' : float(sigma), 'cutoff' : tuple([float(x) for x in cutoff[1:-1].split(',')])}

    def grid(self):
        if self.grid_type != 'dg_cut':
            raise NotImplementedError('Currently, only the grid dg_cut is implemented.')
        asc = 0
        desc = 0
        x_grid = [0]
        while (asc is not None) or (desc is not None):
            if asc is not None:
                asc += self._step_sequencer(asc, self.mu, self.sigma, self.original_stepsize) * self.original_stepsize
                x_grid = x_grid + [round(asc, 8)]
                if asc > self.cutoff[1]:
                    asc = None
            if desc is not None:
                desc -= self._step_sequencer(desc, self.mu, self.sigma, self.original_stepsize) * self.original_stepsize
                x_grid = [round(desc, 8)] + x_grid
                if desc < self.cutoff[0]:
                    desc = None
        grid = []
        for item in x_grid:
            bins = []
            bin_height = self._step_sequencer(item, self.mu, self.sigma, original_stepsize=0.1) / (self.num_bins * 2.)
            for idx in range(1, self.num_bins + 1):
                bins.append(bin_height * idx)
            grid.append([item, bins])
        return grid

    def _gauss(self, x, mu, sigma, normalized=True):
        coefficient = (np.sqrt(2 * np.pi) * sigma)
        value = np.exp((-0.5) * ((x - mu) / (sigma * 1.)) ** 2)
        if normalized:
            value = value / (coefficient * 1.)
        return value

    def _step_sequencer(self, x, mu, sigma, original_stepsize):
        return int(round((1 + original_stepsize - self._gauss(x, mu, sigma, normalized=False)) / original_stepsize, 9))
