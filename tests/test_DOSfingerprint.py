import pytest
import numpy as np

from nomad_dos_fingerprints import DOSFingerprint
from nomad_dos_fingerprints.DOSfingerprint import ELECTRON_CHARGE 

def test_integrate_to_bins():
    test_data_x = np.linspace(0, np.pi, num = 1000)
    test_data_y = [np.sin(x) for x in test_data_x]
    fp = DOSFingerprint(stepsize=0.001)
    energies, dos = fp._integrate_to_bins(test_data_x, test_data_y) # WARNING! energy sampling is not tested
    energies_test_data = []
    current_energy = 0
    while current_energy <= np.pi:
        energies_test_data.append(current_energy)
        current_energy += fp.stepsize
    assert np.isclose(sum(dos), 2)
    assert (np.isclose(energies, np.array(energies_test_data))).all()

def test_convert_dos():
    test_data_x = np.arange(1, 5, step = 0.01)
    test_data_y = test_data_x
    fp = DOSFingerprint(stepsize=0.001)
    x, y = fp._convert_dos(test_data_x* ELECTRON_CHARGE, [test_data_y/2 / ELECTRON_CHARGE, test_data_y/2 / ELECTRON_CHARGE])
    assert np.isclose(x,test_data_x).all()
    assert np.isclose(y, test_data_y).all()
    x, y = fp._convert_dos(test_data_x* ELECTRON_CHARGE, [test_data_y/ ELECTRON_CHARGE])
    assert np.isclose(x,test_data_x).all()
    assert np.isclose(y, test_data_y).all()

@pytest.mark.skip()
def test_print():
    from nomad_dos_fingerprints import Grid
    test_data_x = np.arange(-15, 15, step = 0.05)
    test_data_y = test_data_x
    grid = Grid().create()
    fp = DOSFingerprint(stepsize=0.05)
    x,y = fp._calculate_bytes(test_data_x, test_data_y, grid)
    assert (x != y).all()

