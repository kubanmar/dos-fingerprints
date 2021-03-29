import numpy as np
from bitarray import bitarray
from functools import partial
from collections.abc import Callable

from .grid import Grid
from .similarity import tanimoto_similarity

ELECTRON_CHARGE = 1.602176565e-19

class DOSFingerprint():

    def __init__(self, stepsize = 0.05, similarity_function = tanimoto_similarity, **kwargs):
        self.bins = ''
        self.indices = []
        self.stepsize = stepsize
        self.filling_factor = 0
        self.grid_id = None
        self.set_similarity_function(similarity_function, **kwargs)

    def calculate(self, dos_energies, dos_values, grid_id = 'dg_cut:56:-2:7:(-10, 5)', convert_data = 'Enc', unit_cell_volume = 1, n_atoms = 1):
        if convert_data == 'Enc':
            energy, dos = self._convert_dos(dos_energies, dos_values, unit_cell_volume = unit_cell_volume, n_atoms = n_atoms)
        elif isinstance(convert_data, Callable):
            energy, dos = convert_data(dos_energies, dos_values)
        elif convert_data == None:
            energy, dos = dos_energies, dos_values
        else:
            raise ValueError('Key-word argument ´convert_data´ must be either the string "Enc", a callable or None.')
        raw_energies, raw_dos = self._integrate_to_bins(energy, dos)
        grid = Grid.create(grid_id = grid_id)
        self.grid_id = grid_id if grid_id != None else grid.get_grid_id()
        bin_fp = self._calculate_bit_fingerprint(raw_energies, raw_dos, grid)
        self.bins = self._calculate_byte_representation(bin_fp)
        return self

    def to_dict(self):
        return dict(bins = self.bins, indices = self.indices, stepsize = self.stepsize, grid_id = self.grid_id, filling_factor = self.filling_factor)

    @staticmethod
    def from_dict(fp_dict):
        self = DOSFingerprint()
        self.bins = fp_dict['bins']
        self.indices = fp_dict['indices']
        self.stepsize = fp_dict['stepsize']
        self.grid_id = fp_dict['grid_id']
        self.filling_factor = fp_dict['filling_factor']
        return self

    def set_similarity_function(self, similarity_function, **kwargs):
        self.similarity_function = partial(similarity_function, **kwargs)

    def get_similarity(self, fingerprint):
        return self.similarity_function(self, fingerprint)

    def get_similarities(self, list_of_fingerprints):
        return np.array([self.similarity_function(self, fp) for fp in list_of_fingerprints])

    def __eq__(self, other):
        return self.bins == other.bins and self.indices == other.indices and self.stepsize == other.stepsize and self.grid_id == other.grid_id and self.filling_factor == other.filling_factor

    def _integrate_to_bins(self, xs, ys):
        """
        Performs stepwise numerical integration of ``ys`` over the range of ``xs``. The stepsize of the generated histogram is controlled by DOSFingerprint().stepsize.
        """
        if len(xs) < 2 or len(ys) < 2:
            raise ValueError(f'Invalid input. Please provide arrays with len > 2. len(x) : {len(xs)} len(y) : {len(ys)}')
        xstart = round(int(xs[0] / (self.stepsize * 1.)) * self.stepsize, 8)  # define the limits that fit with the predefined stepsize
        xstop = round(int(xs[-1] / (self.stepsize * 1.)) * self.stepsize, 8)
        x_interp = np.arange(xstart, xstop + self.stepsize, self.stepsize)
        x_interp = np.around(x_interp, decimals=5)
        y_interp = np.interp(x_interp, xs, ys)
        y_integ = np.array([np.trapz(y_interp[idx:idx + 2], x_interp[idx:idx + 2]) for idx in range(len(x_interp)-1)])
        return x_interp[:-1], y_integ

    def _convert_dos(self, energy, dos, unit_cell_volume = 1, n_atoms = 1):
        """
        Convert units of DOS from energy: Joule; dos: states/volume/Joule to eV and sum spin channels if they are present.
        """
        energy = np.array([value / ELECTRON_CHARGE for value in energy])
        dos_channels = [np.array(values) for values in dos]
        dos = sum(dos_channels) * ELECTRON_CHARGE * unit_cell_volume * n_atoms
        return energy, dos

    def _binary_bin(self, dos_value: float, grid_bins: np.ndarray):
        bin_dos = ''
        for grid_bin in grid_bins:
            if grid_bin <= dos_value:
                bin_dos += '1'
            else:
                bin_dos += '0'
        return bin_dos

    def _calc_grid_indices(self, energy_bins: np.ndarray, grid: Grid):
        """
        Calculate indices of the DOS grid that describe the energy range where the DOS is defined.
        Args:
            energy_bins: np.ndarray: energy bins cut to be no larger than the Grid range
            grid: Grid: description of the DOS discretization
        Return:
            self.indices: list: Grid indices that indicate which parts of the Grid are no larger than the energy bins
        """
        grid_start, grid_end = grid.get_grid_indices_for_energy_range(energy_bins)
        self.indices = [grid_start, grid_end]
        return self.indices
        
    def _adapt_energy_bin_sizes(self, energy_bins: np.ndarray, states: np.ndarray, grid: Grid):
        """
        Adapt energy bin sizes to the specified values in the grid.
        Args:
            energy_bins: numpy.ndarray: locations of energy bins
            states: numpy.ndarray: states in the bins declared in `energy_bins`
            grid: Grid: description of the DOS discretization
        Returns:
            adapted_bins: list: new energy discretization steps
            adapted_states: list: state bins with adapted discretization steps
        """
        grid_array = grid.grid()
        # cut the energy and states to grid size
        energy_bins, states = np.transpose([(e,d) for e,d in zip(energy_bins, states) if (e >= grid_array[0][0] and e <= grid_array[-1][0])])
        # find grid start and end points
        grid_start, grid_end = self._calc_grid_indices(energy_bins, grid)
        # sum dos bins to adapt to inhomogeneous energy grid
        adapted_bins = []
        adapted_states = []
        for index in range(grid_start, grid_end):
            eps_i = grid_array[index][0]
            eps_iplusdelta = grid_array[index+1][0]
            adapted_bins.append(eps_i)
            adapted_states.append(sum(np.array([s for e, s in zip(energy_bins, states) if e >= eps_i and e < eps_iplusdelta])))
        return adapted_bins, adapted_states

    def _calc_bit_vector(self, adapted_states: np.ndarray, grid: Grid):
        """
        Calculate bit representation of grid-adapted states per energy bin.
        Args:
            adapted_states: numpy.ndarray: states in the bins declared in `energy_bins`
            grid: Grid: description of the DOS discretization
        Sets:
            overflow: float: number of states that can not be described by the current grid
        Returns: 
            discrete_states: str: binary representation of binned DOS spectrum
        """
        bin_fp = ''
        grid_array = grid.grid()
        grid_start, grid_end = self.indices
        overflow = 0
        for states, grid_segment in zip(adapted_states, grid_array[grid_start:grid_end + 1]):
            bin_fp += self._binary_bin(states, grid_segment[1])
            overflow += self._get_segment_overflow(states, grid_segment)
        self.overflow = overflow
        return bin_fp

    def _calculate_byte_representation(self, bin_fp: str):
        byte_fp = bitarray(bin_fp).tobytes().hex()
        return byte_fp


    def _calculate_bit_fingerprint(self, binned_energies: np.ndarray, binned_states: np.ndarray, grid: Grid):
        """
        Calculate byte representation of DOS fingerprint.
        """
        _, adapted_states = self._adapt_energy_bin_sizes(binned_energies, binned_states, grid)
        bin_fp = self._calc_bit_vector(adapted_states, grid)
        self.filling_factor = bin_fp.count('1') / len(bin_fp)
        return bin_fp

    def _get_segment_overflow(self, states, grid_segment):
        """
        Calculate the number of states that lies beyond the region described by a grid segment.
        Args:
            states: float: states in the bins declared in `energy_bins`
            grid_segment: Grid: description of the DOS discretization
        Returns:
            segment_overflow: float: number of states that can be described by the grid segment
        """
        delta_states = states - grid_segment[1][-1] 
        segment_overflow = delta_states if delta_states >= 0 else 0
        return  segment_overflow
