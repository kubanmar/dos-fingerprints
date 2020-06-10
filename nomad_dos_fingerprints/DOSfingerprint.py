import numpy as np
from bitarray import bitarray

from .grid import Grid

ELECTRON_CHARGE = 1.602176565e-19

class DOSFingerprint():

    def __init__(self, stepsize = 0.05):
        self.bins = ''
        self.indices = []
        self.stepsize = stepsize
        self.filling_factor = 0

    def calculate(self, dos_energies, dos_values):
        energy, dos = self._convert_dos(dos_energies, dos_values)
        raw_energies, raw_dos = self._integrate_to_bins(energy, dos)
        grid = Grid().create()
        self.indices, self.bins = self._calculate_bytes(raw_energies, raw_dos, grid)
        return self

    def _integrate_to_bins(self, xs, ys):
        """
        Performs stepwise numerical integration of ``ys`` over the range of ``xs``. The stepsize of the generated histogram is controlled by DOSFingerprint().stepsize.
        """
        xstart = round(int(xs[0] / (self.stepsize * 1.)) * self.stepsize, 8)  # define the limits that fit with the predefined stepsize
        xstop = round(int(xs[-1] / (self.stepsize * 1.)) * self.stepsize, 8)
        x_interp = np.arange(xstart, xstop + self.stepsize, self.stepsize)
        y_interp = np.array(list(map(lambda x: x * self.stepsize, np.interp(x_interp, xs, ys))))
        return x_interp, y_interp

    def _convert_dos(self, energy, dos):
        """
        Convert units of DOS from energy: Joule; dos: states/volume/Joule to eV and sum spin channels if they are present.
        """
        energy = np.array([value / ELECTRON_CHARGE for value in energy])
        dos_channels = [np.array(values) for values in dos]
        dos = sum(dos_channels) * ELECTRON_CHARGE
        return energy, dos
    
    def _binary_bin(self, dos_value, grid_bins):
        bin_dos = ''
        for grid_bin in grid_bins:
            if grid_bin <= dos_value:
                bin_dos += '1'
            else:
                bin_dos += '0'
        return bin_dos

    def _calculate_bytes(self, energy, dos, grid):
        """
        Calculate the byte fingerprint.
        """
        grid_array = grid.grid()
        # cut the energy and dos to grid size
        energy, dos = np.transpose([(e,d) for e,d in zip(energy, dos) if (e >= grid_array[0][0] and e <= grid_array[-1][0])])        
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
        while grid_array[grid_index + 1][0] < energy[-1]:
            current_dos = 0
            while energy[fp_index] < grid_array[grid_index + 1][0]:
                current_dos += dos[fp_index]
                fp_index += 1
            bin_fp += self._binary_bin(current_dos, grid_array[grid_index][1])
            grid_index += 1
        self.filling_factor = bin_fp.count('1') / len(bin_fp)
        byte_fp = bitarray(bin_fp).tobytes().hex()
        return [grid_start, grid_index], byte_fp
