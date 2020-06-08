import pytest
import numpy as np

from nomad_dos_fingerprints import DOSFingerprint

def test_integrate_to_bins():
    test_data_x = np.linspace(0, np.pi, num = 100)
    test_data_y = [np.sin(x) for x in test_data_x]
    fp = DOSFingerprint()
    energies, dos = fp._integrate_to_bins(test_data_x, test_data_y)
    assert np.isclose(sum(dos), 2)