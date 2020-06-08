from nomad_dos_fingerprints import DOSFingerprint
import pytest, os, json

with open(os.path.join(os.path.dirname(__file__), 'fingerprint_generation_test_data.json'), 'r') as test_data_file:
    test_data = json.load(test_data_file)
    
def test_fingerprint_values():
    
    for fp, mid in test_data['fingerprints']:
        raw_data  = test_data[mid]
        assert json.loads(fp)['bins'] == DOSFingerprint().calculate(raw_data['dos_energies'], raw_data['dos_values'])
