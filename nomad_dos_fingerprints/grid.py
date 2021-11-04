from typing import Callable, List
import numpy as np
from functools import partial

class Grid():
    
    @staticmethod
    def create(grid_id = None, grid_type = 'dg_cut', mu = -2, sigma = 7, cutoff = (-10,5), num_bins = 56, original_stepsize = 0.05, bin_size_factor = 56):
        """
        Create grid object. 
        There are two options to initialize the grid: 
            1. Initialize from grid id: The grid id, which is a string that contains all grid parameters, is used to set the parameters of the grid.
            2. Initialize from parameters: The grid is created directly from the parameters passed as key-word arguments.
        
        **Returns**

        self: *Grid*
            The grid object with set grid parameters.
        """
        if grid_id == None:
            self = Grid()
            self.grid_type = grid_type
            self.mu = mu
            self.sigma = sigma
            self.cutoff = cutoff
            self.num_bins = num_bins
            self.original_stepsize = original_stepsize
            self.bin_size_factor = bin_size_factor
            self.grid_id = self.get_grid_id()
        else:
            self = Grid.create(**Grid.resolve_grid_id(grid_id))
        return self

    def get_grid_id(self):
        id = ':'.join([self.grid_type, str(self.num_bins), str(self.mu), str(self.sigma), str(self.cutoff), str(self.bin_size_factor)])
        return id

    @staticmethod
    def resolve_grid_id(grid_id: str) -> dict:
        grid_id_variables = grid_id.split(':')
        if len(grid_id_variables) == 5: # Downwards compatibility
            grid_type, num_bins, mu, sigma, cutoff = grid_id_variables
            bin_size_factor = num_bins
        elif len(grid_id_variables) == 6:
            grid_type, num_bins, mu, sigma, cutoff, bin_size_factor = grid_id_variables
        else:
            raise ValueError('Grid id can not be interpreted.')
        return {'grid_type' : grid_type, 'num_bins' : int(num_bins), 'mu' : float(mu), 'sigma' : float(sigma), 'cutoff' : tuple([float(x) for x in cutoff[1:-1].split(',')]), 'bin_size_factor' : float(bin_size_factor)}

    def grid(self) -> list:
        """
        Generate a grid with parameters set by `self.grid_id`.

        **Returns**

        grid: *list*
            Nested list of type [[<grid_point_energy_i>, [<dos_bins_i>]], ...]
        """
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
            bin_height = self._step_sequencer(item, self.mu, self.sigma, original_stepsize=0.1) / (self.bin_size_factor * 2.)
            for idx in range(1, self.num_bins + 1):
                bins.append(bin_height * idx)
            grid.append([item, bins])
        return grid

    def grid_new(self, delta_e_min, delta_e_max, delta_rho_min, delta_rho_max, sigma, n_pix, cutoff):
        """
        New implementation of the grid. Follows the description in the publication.
        """

        f_energies = partial(self.gauss_function, w_min = delta_e_min, w_max = delta_e_max, sigma = sigma) 

        energies = self.energy_intervals_from_function(f_energies, delta_e_min, cutoff)

        f_heights = partial(self.gauss_function, w_min = delta_rho_min, w_max = delta_rho_max, sigma = sigma) 

        max_heights = self.grid_height_from_function(f_heights, energies, delta_rho_min)

        grid = self.grid_from_lists(energies, max_heights, n_pix)

        return grid

    def grid_from_lists(self, energies: list, max_heights: list, n_bins: int) -> list:
        """
        Generate a grid from lists of energies, maximal heights, and the number of bins.

        **Arguments**

        energies: *list*
            List of energy boundaries, aligned at the negative edge of the energy interval for energy bin.

        max_heights: *list*
            List of maximal heights, defined for each energy boundary in `energies`.

        n_bins: *int*
            Number of bins that is used to discretise each energy interval.
        
        **Returns**

        grid: *list*
            Nested list of type [[<grid_point_energy_i>, [<dos_bin_ij>, ...]], ...]
        """
        # avoid unexpected behaviour
        assert len(energies) == len(max_heights), "Number of energy intervals is not equal to the number of maximal heights for each energy."

        # generate grid
        grid = []
        for energy, max_height in zip(energies, max_heights):
            bin_height = max_height / n_bins
            grid.append([energy, [idx * bin_height for idx in range(1, n_bins + 1)]])
        return grid

    def get_grid_indices_for_energy_range(self, energy: list) -> set:
        grid_energies = [x[0] for x in self.grid()]
        energy = [e for e in energy if (e >= grid_energies[0] and e <= grid_energies[-1])]
        for idx, grid_e in enumerate(grid_energies):
            if grid_e >= energy[0]:
                grid_start = idx if idx >= 0 else 0
                break
        for idx, grid_e in reversed(list(enumerate(grid_energies))):
            if grid_e <= energy[-1]:
                grid_end = idx if idx <= len(grid_energies) - 1 else len(grid_energies) - 1 # last index is len(list_) - 1 
                break
        return grid_start, grid_end

    def energy_intervals_from_function(self, function: Callable, minimal_interval: float, energy_limits: List[float]) -> list:
        """
        Generate a set of energy intervals from a given function. The energy intervals are generated as:

        e_i = \\sum_{j=0}^{i-1} * minimal_interval * function(e_j)

        and 

        e_{-i} = -1 * e_i

        **Arguments**

        function: *Callable*
            function that maps an energy to a number greater or equal 1

        minimal_interval: *float*
            minimal interval width, i.e. minimal distance between two intervals

        energy_limits: *List[float]*
            list [<minimal_energy>, <maximal_energy>] that determines the limits between which energy intervals should be calculated
        """
        # assumes the spectrum is shifted to the reference energy, which is then located at e = 0
        energies = [0]
        
        # we need the maximum absolute limit to generate intervals
        max_limit = max([abs(lim_) for lim_ in energy_limits])
        
        while energies[-1] < max_limit:
            
            # make sure the minimal interval is kept
            next_step = max([function(energies[-1]), 1])
            
            # get new interval width
            next_step *= minimal_interval

            # apply to series
            next_step += energies[-1]

            # increase numerical stability
            next_step = np.round(next_step, 8)

            # update
            if next_step <= max_limit:
                energies.append(next_step)
            else:
                print(f"last entry was {next_step}")
                break
        
        # generate negative indices
        full_set = [-1 * energy for energy in energies[1:] if energy <= abs(energy_limits[0])]
        
        # filter positive energies to match limits, and merge negative and positive indices
        full_set.extend(list(filter(lambda x: x <= energy_limits[1], energies)))

        # make sure the sorting is correct
        full_set.sort()

        return full_set

    def grid_height_from_function(self, function: Callable, energies: list, minimal_height: float) -> list:
        """
        Use provided function to calculate the grid height at each energy.

        **Arguments**

        function: *Callable*
            Function that assigns an integer number to an energy value

        energies: *list*
            List of energies for which the bin heights are calculated

        minimal_height: *float*
            Minimal height of a bin
        """
        return [max([function(energy), 1]) * minimal_height for energy in energies]

    def gauss_function(self, x: float, w_min: float, w_max: float, sigma: float) -> int:
        """
        Function that assigns an integer value to each `x`.

        **Arguments**

        x: *float*
            Input value

        w_min: *float*
            Minimal width of the interval

        w_max: *float*
            Maximal width of the interval

        sigma: *float*
            Parameter controlling the width of the interal gaussian function
        """
        g = (1 - np.exp(-0.5 * (x / sigma)**2))
        value = g * (w_max / w_min - 1) + 1
        return int(value)

    def _gauss(self, x, mu, sigma, normalized=True):
        coefficient = (np.sqrt(2 * np.pi) * sigma)
        value = np.exp((-0.5) * ((x - mu) / (sigma * 1.)) ** 2)
        if normalized:
            value = value / (coefficient * 1.)
        return value

    def _step_sequencer(self, x, mu, sigma, original_stepsize):
        return int(round((1 + original_stepsize - self._gauss(x, mu, sigma, normalized=False)) / original_stepsize, 9))
