import pytest, os, json
from bitarray import bitarray
from nomad_dos_fingerprints import tanimoto_similarity, DOSFingerprint, Grid
from nomad_dos_fingerprints.DOSfingerprint import ELECTRON_CHARGE

with open(os.path.join(os.path.dirname(__file__), 'fingerprint_generation_test_data.json'), 'r') as test_data_file:
    test_data = json.load(test_data_file)

def test_tanimoto():
    # generate fp-type data and check if this can be realized with binary-strings only
    fp1 = DOSFingerprint()
    fp2 = DOSFingerprint()
    fp1.bins = bitarray('00000000111111110000000011111111').tobytes().hex()
    fp2.bins = bitarray('1111111100000000').tobytes().hex()
    grid_id = 'a:8:b'
    fp1.grid_id = grid_id
    fp2.grid_id = grid_id
    fp1.indices = [0,3]
    fp2.indices = [1,2]
    assert tanimoto_similarity(fp1, fp2) == 1
    assert tanimoto_similarity(fp1, fp1) == 1
    assert tanimoto_similarity(fp2, fp2) == 1

def test_matching_of_spectra():
    data = test_data["17661:2634879"]
    cut_energies = []
    cut_dos = []
    cut_energies = [e for e,d in zip(data['dos_energies'], data['dos_values'][0]) if (e / ELECTRON_CHARGE > -7.3 and e / ELECTRON_CHARGE < 2)]
    cut_dos = [d for e,d in zip(data['dos_energies'], data['dos_values'][0]) if (e / ELECTRON_CHARGE > -7.3 and e / ELECTRON_CHARGE < 2)]
    fp = DOSFingerprint().calculate(data['dos_energies'], data['dos_values'])
    cut_fp = DOSFingerprint().calculate(cut_energies, [cut_dos])
    assert tanimoto_similarity(cut_fp, fp) == tanimoto_similarity(fp, cut_fp)
    assert 1 - tanimoto_similarity(fp, cut_fp) < 1e-2  