from typing import Callable, List, Set
import numpy as np
from functools import partial

class NotCreatedError(Exception):
    """
    Empty container for naming the error occuring when the Grid was initialized, but not created.
    """
    pass

class Grid():

    """
    Grid object used to discretize DOS spectra for the generation of DOS fingerprints.
    To obtain a Grid() object, create it with:
        grid = Grid.create(**parameters)    
    """
    
    @staticmethod
    def create(grid_id = None,
               grid_type = "nonuniform", 
               e_ref = -2,
               delta_e_min = 0.05, 
               delta_e_max = 1.05, 
               delta_rho_min = 0.5, 
               delta_rho_max = 5.5, 
               width = 7, 
               n_pix = 56, 
               cutoff = [-8,7],
               energy_discretization = None,
               states_discretization = None):
        """
        Create grid object. 

        There are two options to initialize the grid: 
            1. Initialize from grid id: The grid id, which is a string that contains all grid parameters, is used to set the parameters of the grid.
            2. Initialize from parameters: The grid is created directly from the parameters passed as key-word arguments.

        **Parameters**

        grid_id: *str* or *None*
            Identifier to re-create a grid, or None, if new or default parameters are given.

        grid_type: *str*
            Currently two options are available:
                * *nonuniform* - Create a non-uniform grid from parameters
                * *uniform* - Create an uniform grid from parameters
                * *custom* - Create a custom grid from lists defining the limits of the grid
            
        e_ref: float
            Reference energy in eV, i.e. the center of the grid.

        delta_e_min: float
            Minimal energy difference in eV between two grid points.

        delta_e_max: float
            Maximal energy difference in eV between two grid points.
            Ignored if *uniform* grid is created.
        
        delta_rho_min: float
            Minimal difference of states in an energy interval.
        
        delta_rho_max: float
            Maximal difference of states in an energy interval.
            Ignored if *uniform* grid is created.
        
        width: float
            Width of the feature region of the Grid in eV.
            Ignored if *uniform* grid is created.

        n_pix: int
            Number of intervals to discretize states in an energy interval.
        
        cutoff: (float, float)
            Lowest and highes energies in eV to be included in the grid.

        energy_discretization: list[float]
            Used by *custom* grid - ignored else.
            Values of energy intervals of the grid, in eV.

        states_discretization: list[float]
            Used by *custom* grid - ignored else.
            Maximal number of states for each energy interval of the grid.       

        **Returns**

        self: *Grid*
            The grid object with set grid parameters.

        **Raises**

        TypeError:
            Grid type can not be identified (neither "nonuniform", nor "uniform", nor "custom").

        ValueError:
            Grid type can no be identified.
        """
        if grid_id is None:
            self = Grid()
            self.grid_type = grid_type
            self.e_ref = e_ref
            self.delta_e_min = delta_e_min
            self.delta_e_max = delta_e_max
            self.delta_rho_min = delta_rho_min
            self.delta_rho_max = delta_rho_max
            self.width = width
            self.cutoff_min = cutoff[0]
            self.cutoff_max = cutoff[1]
            self.n_pix = n_pix
            self.energy_discretization = energy_discretization
            self.states_discretization = states_discretization
            self.grid_id = self.get_grid_id()
        else:
            self = Grid.create(**Grid.resolve_grid_id(grid_id))
        return self

    def get_grid_id(self):
        self._check_created()
        if self.grid_type == "nonuniform":
            grid_id_values = [self.grid_type, self.e_ref, self.delta_e_min, self.delta_e_max, self.delta_rho_min, self.delta_rho_max, self.width, self.cutoff_min, self.cutoff_max, self.n_pix]
        elif self.grid_type == "uniform":
            grid_id_values = [self.grid_type, self.e_ref, self.delta_e_min, self.delta_rho_min, self.cutoff_min, self.cutoff_max, self.n_pix]
        elif self.grid_type == "custom":
            if self.energy_discretization is None:
                raise ValueError("Must provide valid energy discretization for custom fingerprint.")
            if self.states_discretization is None:
                raise ValueError("Must provide valid states discretization for custom fingerprint.")
            from hashlib import md5
            all_values = [self.n_pix, self.e_ref]
            all_values = np.append(all_values, self.energy_discretization)
            all_values = np.append(all_values, self.states_discretization)
            all_values = "".join([str(value) for value in all_values])
            return f"custom:{md5(all_values.encode()).hexdigest()}"
        else:
            raise ValueError("Grid type can not be identified")
        grid_id = ':'.join([str(value) for value in grid_id_values])
        return grid_id

    @staticmethod
    def resolve_grid_id(grid_id: str) -> dict:
        grid_variables = {}
        grid_id_variables = grid_id.split(':')
        if grid_id_variables[0] == "nonuniform":
            grid_variables["grid_type"] = grid_id_variables[0] 
            grid_variables["e_ref"] = float(grid_id_variables[1])
            grid_variables["delta_e_min"] = float(grid_id_variables[2])
            grid_variables["delta_e_max"] = float(grid_id_variables[3])
            grid_variables["delta_rho_min"] = float(grid_id_variables[4])
            grid_variables["delta_rho_max"] = float(grid_id_variables[5])
            grid_variables["width"] = float(grid_id_variables[6])
            grid_variables["cutoff"] = [float(grid_id_variables[7])]
            grid_variables["cutoff"].append(float(grid_id_variables[8]))
            grid_variables["n_pix"] = int(grid_id_variables[9])
        elif grid_id_variables[0] == "uniform":
            grid_variables["grid_type"] = grid_id_variables[0]
            grid_variables["e_ref"] = float(grid_id_variables[1])
            grid_variables["delta_e_min"] = float(grid_id_variables[2])
            grid_variables["delta_rho_min"] = float(grid_id_variables[3])
            grid_variables["cutoff"] = [float(grid_id_variables[4])]
            grid_variables["cutoff"].append(float(grid_id_variables[5]))
            grid_variables["n_pix"] = int(grid_id_variables[6])
        elif grid_id_variables[0] == "custom":
            raise TypeError("Custom grid id can not be resolved.")
        else:
            raise ValueError("Grid type can not be identified")
        return grid_variables

    def grid(self) -> list:
        """
        Generate a grid with parameters set by `self.grid_id`.

        **Returns**

        grid: *list*
            Nested list of type [[<grid_point_energy_i>, [<dos_bins_i>]], ...]
        """
        self._check_created()
        if self.grid_type == "nonuniform":
            grid = self._grid_non_uniform(self.delta_e_min, self.delta_e_max, self.delta_rho_min, self.delta_rho_max, self.width, self.n_pix, [self.cutoff_min, self.cutoff_max])
        elif self.grid_type == "uniform":
            grid = self._grid_uniform(self.delta_e_min, self.delta_rho_min, self.n_pix, [self.cutoff_min, self.cutoff_max])
        elif self.grid_type == "custom":
            grid = self._grid_custom(self.energy_discretization, self.states_discretization, self.n_pix)
        else:
            raise ValueError("Grid type can not be identified")
        return grid

    def copy(self) -> object:
        """
        Create a copy of self.
        """
        return Grid.create(grid_id=self.get_grid_id())

    def _grid_non_uniform(self, delta_e_min, delta_e_max, delta_rho_min, delta_rho_max, width, n_pix, cutoff):
        """
        New implementation of the grid. Follows the description in the publication.
        """

        f_energies = partial(self.gauss_function, w_min = delta_e_min, w_max = delta_e_max, sigma = width) 

        energies = self.energy_intervals_from_function(f_energies, delta_e_min, cutoff)

        f_heights = partial(self.gauss_function, w_min = delta_rho_min, w_max = delta_rho_max, sigma = width) 

        max_heights = self.grid_height_from_function(f_heights, energies, delta_rho_min)

        grid = self.grid_from_lists(energies, max_heights, n_pix)

        return grid

    def _grid_uniform(self, delta_e, delta_rho, n_pix, cutoff):

        def f_energies(x):
            return 1

        energies = self.energy_intervals_from_function(f_energies, delta_e, cutoff)

        max_heights = self.grid_height_from_function(f_energies, energies, delta_rho)

        grid = self.grid_from_lists(energies, max_heights, n_pix)

        return grid
        
    def _grid_custom(self, energy_discretization, states_discretization, n_pix):
        return self.grid_from_lists(energy_discretization, states_discretization, n_pix)

    def grid_from_lists(self, energies: List[float], max_heights: List[float], n_bins: int) -> List[List[float]]:
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
            grid.append([energy, [round(idx * bin_height, 10) for idx in range(1, n_bins + 1)]])
        return grid

    def get_grid_indices_for_energy_range(self, energy: List[float]) -> Set[int]:
        """
        Compute the grid indices that match a given energy range.

        **Arguments**

            energy: *list*
            Energies to match grid energies.

        **Returns**

            grid_start, grid_end: set
            First and last index of the grid that are in the limits of the energies.
        """
        grid_energies = [x[0] for x in self.grid()]
        energy = [e for e in energy if (e >= grid_energies[0] and e <= grid_energies[-1])]
        for idx, grid_e in enumerate(grid_energies):
            if grid_e >= energy[0]:
                grid_start = idx if idx >= 0 else 0
                break
        for idx, grid_e in reversed(list(enumerate(grid_energies))):
            if grid_e <= energy[-1]:
                grid_end = idx  
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

    def _check_created(self):
        if not hasattr(self, "grid_type"):
            raise NotCreatedError("Grid was not created. To do so, use Grid.create(**parameters).")