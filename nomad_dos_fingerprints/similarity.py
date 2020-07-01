import numpy as np
from bitarray import bitarray

def match_fingerprints(fingerprint1, fingerprint2):
    if fingerprint1.grid_id != fingerprint2.grid_id:
        raise AssertionError('Can not calculate similarity of fingerprints that have been calculated with different grids.')
    num_bins = int(fingerprint1.grid_id.split(':')[1])
    fp1 = bitarray()
    fp2 = bitarray()
    fp1.frombytes(bytes.fromhex(fingerprint1.bins))
    fp2.frombytes(bytes.fromhex(fingerprint2.bins))
    start_index = max([fingerprint1.indices[0], fingerprint2.indices[0]])
    stop_index = min([fingerprint1.indices[1], fingerprint2.indices[1]])
    # find offsets
    dsp1 = (start_index - fingerprint1.indices[0]) * num_bins
    dsp2 = (start_index - fingerprint2.indices[0]) * num_bins
    dep1 = (fingerprint1.indices[1] - stop_index) * num_bins
    dep2 = (fingerprint2.indices[1] - stop_index) * num_bins
    fp1 = fp1[dsp1:len(fp1) - 1 - dep1]
    fp2 = fp2[dsp2:len(fp2) - 1 - dep2]
    return fp1, fp2

def tanimoto_similarity(fingerprint1, fingerprint2):
    fp1, fp2 = match_fingerprints(fingerprint1, fingerprint2)
    a = fp1.count()
    b = fp2.count()
    c = (fp1 & fp2).count()
    try:
        tc = c / float(a + b - c)
    except ZeroDivisionError:
        tc = 0
    return tc
