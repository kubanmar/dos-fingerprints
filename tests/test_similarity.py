import pytest
from bitarray import bitarray
from nomad_dos_fingerprints import tanimoto_similarity, DOSFingerprint

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
