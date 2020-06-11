import numpy as np
from bitarray import bitarray

def tanimoto_similarity(fingerprint1, fingerprint2):
    if fingerprint1.grid_id != fingerprint2.grid_id:
        raise AssertionError('Can not calculate similarity of fingerprints that have been calculated with different grids.')
    # match fingerprints
    num_bins = int(fingerprint1.grid_id.split(':')[1])
    offset = abs(fingerprint1.indices[0]-fingerprint2.indices[0])
    fingerprints = sorted([fingerprint1.to_dict(), fingerprint2.to_dict()], key = lambda x: x['indices'][0], reverse=True)
    if offset != 0:
        fingerprints[0]['bins'] = int(offset * num_bins / 8) * '00' + fingerprints[0]['bins']
    min_len = min([len(fingerprint['bins']) for fingerprint in fingerprints])
    mask = bitarray()
    fp1 = bitarray()
    fp2 = bitarray()
    mask.frombytes(bytes.fromhex(int(offset * num_bins / 8) * '00' + int((min_len / 2 - offset)) * 'ff'))
    fp1.frombytes(bytes.fromhex(fingerprints[0]['bins'][:min_len]))
    fp2.frombytes(bytes.fromhex(fingerprints[1]['bins'][:min_len]))
    fp1 = fp1 & mask
    fp2 = fp2 & mask
    a = fp1.count()
    b = fp2.count()
    c = (fp1 & fp2).count()
    return c / float(a + b - c)