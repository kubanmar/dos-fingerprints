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
        self.grid_id = grid.get_grid_id()
        self.indices, self.bins = self._calculate_bytes(raw_energies, raw_dos, grid)
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
            raise ValueError(f'Invalid input. Please provide arrays with len > 2. \n{xs}\n{ys}')
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

    def _binary_bin(self, dos_value, grid_bins):
        bin_dos = ''
        for grid_bin in grid_bins:
            if grid_bin <= dos_value:
                bin_dos += '1'
            else:
                bin_dos += '0'
        return bin_dos

    def _adapt_energy_bin_sizes(self, energy_bins: np.ndarray, states: np.ndarray):
        """
        Adapt energy bin sizes to the specified values in the grid.
        Args:
            energy_bins: numpy.ndarray: locations of energy bins
            states: numpy.ndarray: states in the bins declared in `energy_bins`
        Returns:
            None
        """
        grid = Grid.create(grid_id = self.grid_id)
        grid_array = grid.grid()
        # cut the energy and states to grid size
        energy_bins, states = np.transpose([(e,d) for e,d in zip(energy_bins, states) if (e >= grid_array[0][0] and e <= grid_array[-1][0])])
        # find grid start and end points
        grid_start, grid_end = grid.get_grid_indices_for_energy_range(energy_bins)
        # sum dos bins to adapt to inhomogeneous energy grid
        adapted_bins = []
        adapted_states = []
        for index in range(grid_start, grid_end):
            eps_i = grid_array[index][0]
            eps_iplusdelta = grid_array[index+1][0]
            adapted_bins.append(eps_i)
            adapted_states.append(sum(np.array([s for e, s in zip(energy_bins, states) if e >= eps_i and e < eps_iplusdelta])))
        return adapted_bins, adapted_states



    def _calculate_bytes(self, energy, dos, grid, return_binned_dos = False):
        """
        Calculate the byte fingerprint.
        """
        grid_array = grid.grid()
        # cut the energy and dos to grid size
        energy, dos = np.transpose([(e,d) for e,d in zip(energy, dos) if (e >= grid_array[0][0] and e <= grid_array[-1][0])])
        # calculate fingerprint
        bin_fp = ''
        grid_index = 0
        for idx, grid_e in enumerate(grid_array):
            if grid_e[0] > energy[0]:
                grid_index = idx - 1
                if grid_index < 0:
                    grid_index = 0
                break
        grid_start = grid_index
        fp_index = 0
        binned_dos = []
        grid_bins = []
        while grid_array[grid_index + 1][0] < energy[-1]:
            current_dos = 0
            while energy[fp_index] < grid_array[grid_index + 1][0]:
                current_dos += dos[fp_index]
                fp_index += 1
            binned_dos.append(current_dos)
            grid_bins.append(grid_array[grid_index][0])
            bin_fp += self._binary_bin(current_dos, grid_array[grid_index][1])
            grid_index += 1
        self.filling_factor = bin_fp.count('1') / len(bin_fp)
        byte_fp = bitarray(bin_fp).tobytes().hex()
        if return_binned_dos:
            return [grid_start, grid_index], byte_fp, [grid_bins, binned_dos]
        return [grid_start, grid_index], byte_fp
