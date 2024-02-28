import numpy as np
from bitarray import bitarray
from functools import partial
from typing import Callable, List, Union
from itertools import groupby

from .grid import Grid
from .similarity import tanimoto_similarity

ELECTRON_CHARGE = 1.602176565e-19

class DOSFingerprint():
    """
    A fingerprint of the electronic density-of-states (DOS), 
    obtained by integrating the DOS over discrete intervals
    and binning of the resulting histogram.
    
    **Keyword arguments:**

    similarity_function: `Callable`
        A function that allows to calculate the similarity between two DOS fingerprints    

        default: `nomad_dos_fingerprints.similarity.tanimoto_similarity`

    Additional keyword arguments are passed to the similarity function.
    """

    def __init__(self, 
                 similarity_function: Callable = tanimoto_similarity, **kwargs):
        self.bins = ''
        self.indices = []
        self.filling_factor = 0
        self.grid_id = None
        self.set_similarity_function(similarity_function, **kwargs)
        self._n_state_bins = None

    @property
    def n_state_bins(self):
        if self._n_state_bins is None:
            assert self.grid_id is not None, "Grid id is not set. Will be set when calculating the fingerprint with a grid."
            self._n_state_bins = Grid.resolve_grid_id(self.grid_id)["n_pix"]
        return self._n_state_bins

    def calculate(self, 
                  dos_energies: np.ndarray, 
                  dos_values: np.ndarray, 
                  grid_id: str = None, 
                  convert_data: Union[None, str, Callable] = None, 
                  normalization_factor: float = 1.0) -> object:
        """
        Calculate the fingerprint data from normalized DOS energies and values.

        **Arguments:**

        dos_energies: `np.ndarray`
            Energies of the DOS spectrum

            By default, units are [eV].

            This behavior can be changed by changing `convert_data` (see below).

        dos_values: `np.ndarray`
            Values of the DOS spectrum

            By default, units are [states/cell/eV].

            This behavior can be changed by changing `convert_data` (see below).

        **Keyword arguments:**

        grid_id: `str`
            ID for describing the `Grid` object that is used to calculate the fingerprint data.
            For details, see documentation there.

            default: `None`: Use `Grid` default values.

        convert_data: `Union[str, None, Callable]`
            Convert `dos_energies` and `dos_values` to different units.

            The string `'enc'` transforms [Joule] to [eV] and sums all spin channels of DOS values. 
            Additionally, assumes that `dos_energies` is an `Interable` of `float`s 
            and `dos_values` is an `Interable` of `Interables` of lenght `len(dos_energies)` of `float`s.

            If set to `None`, no conversion will be performed.

            If a callable is given, it will be called as:
            ``energy, dos = convert_data(dos_energies, dos_values)``.

            default: `None`

        normalization_factor: `float`
            Factor used for unit conversion, e.g. if the DOS is given or required per atom.
            Used only if `convert_data` is "Enc".

            default: `1.0`

        **Returns:**

        self: `DOSFingerprint`
            Fingerprint after calculation
        """
        if isinstance(convert_data, str) and convert_data.lower() == 'enc':
            energy, dos = self._convert_dos(dos_energies, dos_values, normalization_factor = normalization_factor)
        elif isinstance(convert_data, Callable):
            energy, dos = convert_data(dos_energies, dos_values)
        elif convert_data is None:
            energy, dos = dos_energies, dos_values
        else:
            raise ValueError('Key-word argument `convert_data` must be either the string "enc", a callable, or `None`.')
        grid = Grid.create(grid_id = grid_id)
        energy = np.array(energy) - grid.e_ref
        raw_energies, raw_dos = self._integrate_to_bins(energy, dos, grid.delta_e_min)
        self.grid_id = grid_id if grid_id is not None else grid.get_grid_id()
        bin_fp = self._calculate_bit_fingerprint(raw_energies, raw_dos, grid)
        self.bins = self._compress_binary_fingerprint_string(bin_fp)
        self._n_state_bins = grid.n_pix
        return self

    def to_dict(self) -> dict:
        """
        Convert data to dictionary.
        """
        data_dict = {}
        for key in ['bins', 'indices', 'grid_id', 'filling_factor', 'overflow', 'undersampling']:
            data_dict[key] = getattr(self, key)
        return data_dict

    @classmethod
    def from_dict(cls, fp_dict: dict) -> object:
        """
        Create fingerprint object from dictionary.
        """
        self = cls()
        self.bins = fp_dict['bins']
        self.indices = fp_dict['indices']
        self.grid_id = fp_dict['grid_id']
        self.filling_factor = fp_dict['filling_factor']
        if 'overflow' in fp_dict.keys():
            self.overflow = fp_dict['overflow']
        if 'undersampling' in fp_dict.keys():
            self.undersampling = fp_dict['undersampling']
        self._n_state_bins = Grid.resolve_grid_id(self.grid_id)["n_pix"]
        return self

    def set_similarity_function(self, similarity_function: Callable, **kwargs):
        """
        Set the similarity function of the fingerprint.

        **Arguments:**

        similarity_function: `Callable`
            Function for calculating the similarity between `DOSFingerprint` objects.

        Keyword arguments are passed to the similarity function.
        """
        self.similarity_function = partial(similarity_function, **kwargs)

    def get_similarity(self, fingerprint: object) -> float:
        """
        Get similarity value between self and another fingerprint object.

        **Arguments:**

        fingerprint: `DOSFingerprint`
            Other fingerprint to calculate similarity to.

        **Returns:**

        similarity: `float`
            Similarity between both fingerprints.

        **Raises:**

        `TypeError`: `fingerprint` is not a `DOSFingerprint`
        """
        if not isinstance(fingerprint, DOSFingerprint):
            raise TypeError("Other fingerprint must be a `DOSFingerprint` object.")
        return self.similarity_function(self, fingerprint)

    def get_similarities(self, list_of_fingerprints: List[object]) -> np.ndarray:
        """
        Get similarities of self to a list of other fingerprints.

        **Arguments:**

        list_of_fingerprints: `List[DOSFingerprint]`
            Other fingerprints to calculate similarity to.

        **Returns:**

        similarities: `np.ndarray`
            Similarities of self to other fingerprints.
        """
        return np.array([self.similarity_function(self, fp) for fp in list_of_fingerprints])

    def get_bitarray(self) -> object:
        """
        Get `bitarray.bitarray` representing the fingerprint data.

        **Returns:**

        bits: `bitarray`
            Binary fingerprint data 
        """
        if self.bins == '':
            raise AttributeError("Fingerprint is not calculated! Use `calculate()`.")
        bit_string = self._expand_fingerprint_string(self.bins)
        bits = bitarray(bit_string)
        return bits

    def __eq__(self, other):
        if not isinstance(other, DOSFingerprint):
            return False
        for attr in ['bins', 'indices', 'stepsize', 'grid_id', 'filling_factor']:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def _integral_house(self, a, b, interval):
        # integrates a "house" like figure, consisting of a rectangle and a 90 degree triangle, with heights a and b, and width=interval
        return (a + b) * interval / 2.

    def _interpolate_dos(self, dos_values, energy_values, requested_energy):
        """
        Returns a linearly interpolated value between two DOS points.
        """
        if len(dos_values) != 2 or len(energy_values) != 2:
            raise ValueError("Error in _interpolate_dos: Wrong number of arguments for calculation of gradient.")
        if energy_values[0] == energy_values[1]:
            return dos_values[0]
        gradient = (dos_values[1] - dos_values[0]) / ((energy_values[1] - energy_values[0]) * 1.)
        difference = requested_energy - energy_values[0]
        interpolated = dos_values[0] + gradient * difference
        return interpolated
    
    def _find_energy_cutoff_indices(self, energy, estart, estop):
        """
        This find the correct indices for the integration, that is, all values of the energy in the interval [estart-1,estop+1] will be in the range [emin_idx,emax_idx]
        WARNING: Assumes that estart<energy[0]!
        """
        emin_idx = None
        emax_idx = None
        index = 0
        while emin_idx is None or emax_idx is None:
            if energy[index] > estart and emin_idx is None:
                emin_idx = index - 1
            if energy[-(index + 1)] < estop and emax_idx is None:
                emax_idx = -index + 1
                if emax_idx > 0:
                    emax_idx = 0 #HOTFIX
            index += 1
        if emax_idx == 0:
            emax_idx = None
        return emin_idx, emax_idx

    def _integrate_to_bins(self, energy, dos, stepsize=0.05):
        """
        Performs stepwise numerical integration of ``ys`` over the range of ``xs``.
        """
        if len(energy) < 2 or len(dos) < 2:
            raise ValueError(f'Invalid input. Please provide arrays with len > 2. len(x) : {len(energy)} len(y) : {len(dos)}')
        
        # define the limits that fit with the predefined stepsize 
        xstart = round(np.ceil(energy[0] / (stepsize * 1.)) * stepsize, 12)  
        xstop = round(np.floor(energy[-1] / (stepsize * 1.)) * stepsize, 12)

        # Find the indices in original arrays closest to new limits defined by stepsize
        idx_min, idx_max = self._find_energy_cutoff_indices(energy, xstart, xstop)

        energy = energy[idx_min:idx_max]
        dos = dos[idx_min:idx_max]
        current_energy = xstart
        current_dos = self._interpolate_dos([dos[0], dos[1]], [energy[0], energy[1]], xstart)
        dos_binned = []
        energy_binned = []
        index = 1  # starting from the second value, because energy[0]<xstart. Thus energy[index] is made to be larger than current_energy.
        while current_energy < xstop:
            energy_binned.append(current_energy)
            next_energy = round(current_energy + stepsize, 9)
            integral = 0
            while energy[index] < next_energy and next_energy < xstop:
                integral += self._integral_house(current_dos, dos[index], energy[index] - current_energy)
                current_dos = dos[index]
                current_energy = energy[index]
                index += 1
            next_dos = self._interpolate_dos([dos[index - 1], dos[index]], [energy[index - 1], energy[index]],
                                             next_energy)
            integral += self._integral_house(current_dos, next_dos, next_energy - current_energy)
            dos_binned.append(integral)
            current_energy = next_energy
            current_dos = next_dos
        return energy_binned, dos_binned

    def _convert_dos(self, energy, dos, normalization_factor = 1):
        """
        Convert units of DOS from energy: Joule; dos: states/volume/Joule to eV and sum spin channels if they are present.
        """
        energy = np.array([value / ELECTRON_CHARGE for value in energy])
        dos_channels = [np.array(values) for values in dos]
        dos = sum(dos_channels) * ELECTRON_CHARGE * normalization_factor
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
        
    def _adapt_energy_bin_sizes(self, energy_bins_raw: np.ndarray, states_raw: np.ndarray, grid: Grid):
        """
        Adapt energy bin sizes to the specified values in the grid.
        Args:
            energy_bins_raw: numpy.ndarray: locations of energy bins
            states_raw: numpy.ndarray: states in the bins declared in `energy_bins`
            grid: Grid: description of the DOS discretization
        Returns:
            adapted_bins: list: new energy discretization steps
            adapted_states: list: state bins with adapted discretization steps
        """
        grid_array = grid.grid()
        # cut the energy and states to grid size
        energy_bins, states = np.transpose([(e,d) for e,d in zip(energy_bins_raw, states_raw) if (e >= grid_array[0][0] and e <= grid_array[-1][0])])
        # find grid start and end points
        grid_start, grid_end = self._calc_grid_indices(energy_bins, grid)
        # if the last energy bin is less than or equal to the last grid point, the pre-to-last grid index was found instead of the correct one
        # the following lines fix this, but it is not very elegant
        if grid_end <= len(grid_array) - 1:
            if len(energy_bins) == len(energy_bins_raw) and energy_bins_raw[-1] > grid_array[grid_end][0]:
                grid_end += 1
                self.indices[1] += 1
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
            overflow: float: sum of states that can not be described by the current grid
            undersampling: float: fraction of states that are in partially filled grid cells (and thus ignored)
        Returns: 
            discrete_states: str: binary representation of binned DOS spectrum
        """
        bin_fp = ''
        grid_array = grid.grid()
        grid_start, grid_end = self.indices
        overflow = 0
        undersampling = 0
        for states, grid_segment in zip(adapted_states, grid_array[grid_start:grid_end + 1]):
            bin_fp += self._binary_bin(states, grid_segment[1])
            overflow += self._get_segment_overflow(states, grid_segment)
            undersampling += self._get_segment_undersampling(states, grid_segment)
        self.overflow = overflow
        self.undersampling = undersampling / sum(adapted_states)
        return bin_fp

    def _calculate_byte_representation(self, bin_fp: str):
        byte_fp = bitarray(bin_fp).tobytes().hex()
        return byte_fp

    def _compress_group(self, count: int, char: str):
        representative_char = 't' if char == '1' else 'f'
        return str(count) + representative_char

    def _calculate_bit_fingerprint(self, binned_energies: np.ndarray, binned_states: np.ndarray, grid: Grid):
        """
        Calculate bit representation of DOS fingerprint.
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
    
    def _get_segment_undersampling(self, states, grid_segment):
        """
        Calculate the amount of states that are in partially filled grid cells.
        Args:
            states: float: states in the bins declared in `energy_bins`
            grid_segment: Grid: description of the DOS discretization
        Returns:
            segment_undersampling: float: sum of states that are in partially filled grid cells

        """
        delta_states = 0
        for idx in range(len(grid_segment[1]) - 1):
            if grid_segment[1][idx] < states < grid_segment[1][idx+1]:
                delta_states = states - grid_segment[1][idx]
                break
        return delta_states

    def _compress_binary_fingerprint_string(self, fingerprint_string: str):
        compressed_string = ''
        for char, count in groupby(fingerprint_string):
            compressed_string += self._compress_group(len(list(count)), char)
        return compressed_string

    def _expand_fingerprint_string(self, compressed_fingerprint_string):
        current_string = ''
        decompressed_string = ''
        for char in compressed_fingerprint_string:
            if char.isnumeric():
                current_string += char 
            else:
                number_to_add = '1' if char == 't' else '0'
                decompressed_string += number_to_add * int(current_string)
                current_string = ''
        return decompressed_string