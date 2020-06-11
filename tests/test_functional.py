from nomad_dos_fingerprints import DOSFingerprint, tanimoto_similarity
import pytest, os, json
import numpy as np

with open(os.path.join(os.path.dirname(__file__), 'fingerprint_generation_test_data.json'), 'r') as test_data_file:
    test_data = json.load(test_data_file)
    
def test_fingerprint_values():
    
    for fp, mid in test_data['fingerprints']:
        raw_data  = test_data[mid]
        new_fingerprint = DOSFingerprint().calculate(raw_data['dos_energies'], raw_data['dos_values'])
        old_fingerprint = DOSFingerprint()
        old_fingerprint.bins = json.loads(fp)['bins']
        old_fingerprint.indices = json.loads(fp)['indices']
        old_fingerprint.grid_id = new_fingerprint.grid_id
        assert old_fingerprint.indices == new_fingerprint.indices
        assert np.isclose(tanimoto_similarity(old_fingerprint, new_fingerprint),1, atol=5e-2)

def test_materials_similarity():

    fingerprints = test_data['fingerprints']
    similarity_matrix = test_data['simat']
    mids = [x[1] for x in fingerprints]
    raw_data = [test_data[mid] for mid in mids]
    new_fingerprints = [DOSFingerprint().calculate(entry['dos_energies'], entry['dos_values']) for entry in raw_data]
    matrix = []
    for fp1 in new_fingerprints:
        row = []
        for fp2 in new_fingerprints:
            row.append(tanimoto_similarity(fp1,fp2))
        matrix.append(row)
    print(matrix - np.array(similarity_matrix))
    assert np.isclose(similarity_matrix, matrix, atol = 5e-2).all()