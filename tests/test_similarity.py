import os
import json
from nomad_dos_fingerprints import tanimoto_similarity
from nomad_dos_fingerprints.similarity import match_fingerprints
from nomad_dos_fingerprints.DOSfingerprint import ELECTRON_CHARGE, DOSFingerprint


with open(os.path.join(os.path.dirname(__file__), 'fingerprint_generation_test_data.json'), 'r') as test_data_file:
    test_data = json.load(test_data_file)

def test_match_fingerprints():
    
    fp1 = DOSFingerprint()
    fp2 = DOSFingerprint()
    fp1.bins = fp1._compress_binary_fingerprint_string('00000000111111110000000011111111')
    fp2.bins = fp1._compress_binary_fingerprint_string('1111111100000000')
    grid_id = 'nonuniform:1:1:1:1:1:1:1:1:8'
    fp1.grid_id = grid_id
    fp2.grid_id = grid_id
    fp1.indices = [0,3]
    fp2.indices = [1,2]
    print(fp1.n_state_bins, fp2.n_state_bins)

    bits1, bits2 = match_fingerprints(fp1, fp2)

    assert len(bits1) == len(bits2) == 16, "Fingerprints do not match"

def test_tanimoto_v2():
    # generate fp-type data and check if this can be realized with binary-strings only
    fp1 = DOSFingerprint()
    fp2 = DOSFingerprint()
    fp1.bins = fp1._compress_binary_fingerprint_string('00000000111111110000000011111111')
    fp2.bins = fp1._compress_binary_fingerprint_string('1111111100000000')
    grid_id = 'nonuniform:1:1:1:1:1:1:1:1:8'
    fp1.grid_id = grid_id
    fp2.grid_id = grid_id
    fp1.indices = [0,3]
    fp2.indices = [1,2]
    print(fp1.get_bitarray())
    print(fp2.get_bitarray())
    assert tanimoto_similarity(fp1, fp2) == 1, "Non-identical cut fingerprints"
    assert tanimoto_similarity(fp1, fp1) == 1, "Non-identity for Fingerprint v2 NR 1"
    assert tanimoto_similarity(fp2, fp2) == 1, "Non-identity for Fingerprint v2 NR 1"

def test_matching_of_spectra():
    data = test_data["17661:2634879"]
    cut_energies = []
    cut_dos = []
    cut_energies = [e for e,d in zip(data['dos_energies'], data['dos_values'][0]) if (e / ELECTRON_CHARGE > -7.3 and e / ELECTRON_CHARGE < 2)]
    cut_dos = [d for e,d in zip(data['dos_energies'], data['dos_values'][0]) if (e / ELECTRON_CHARGE > -7.3 and e / ELECTRON_CHARGE < 2)]
    fp = DOSFingerprint().calculate(data['dos_energies'], data['dos_values'], convert_data="enc")
    cut_fp = DOSFingerprint().calculate(cut_energies, [cut_dos], convert_data="enc")
    assert tanimoto_similarity(cut_fp, fp) == tanimoto_similarity(fp, cut_fp)
    assert 1 - tanimoto_similarity(fp, cut_fp) < 1e-2  