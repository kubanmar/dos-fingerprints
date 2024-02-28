from typing import Tuple
from bitarray import bitarray

class FingerprintMismatchError(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def match_fingerprints(fingerprint1, fingerprint2) -> Tuple[bitarray, bitarray]:
    if fingerprint1.grid_id != fingerprint2.grid_id:
        raise AssertionError(f'Can not calculate similarity of fingerprints that have been calculated with different grids.\nFingerprint 1: {fingerprint1.grid_id}\nFingerprint 2: {fingerprint2.grid_id}')
    num_bins = fingerprint1.n_state_bins
    fp1 = fingerprint1.get_bitarray()
    fp2 = fingerprint2.get_bitarray()
    start_index = max([fingerprint1.indices[0], fingerprint2.indices[0]])
    stop_index = min([fingerprint1.indices[1], fingerprint2.indices[1]])
    # find offsets
    dsp1 = (start_index - fingerprint1.indices[0]) * num_bins
    dsp2 = (start_index - fingerprint2.indices[0]) * num_bins
    dep1 = (fingerprint1.indices[1] - stop_index) * num_bins
    dep2 = (fingerprint2.indices[1] - stop_index) * num_bins
    fp1 = fp1[dsp1:len(fp1) - dep1]
    fp2 = fp2[dsp2:len(fp2) - dep2]
    return fp1, fp2

def tanimoto_similarity(fingerprint1, fingerprint2):
    """
    Tanimoto similarity (Tc) between `DOSFingerprint` objects.

    Evaluates:

    $Tc(a, b) = a * b / ( a**2 + b**2 - a * b )$

    for binary-valued fingerprint vectors $a$ and $b$.

    Before calculating Tc, the fingerprints are matched such that the same energy 
    regions are descibed in both fingerprints.
    
    If any fingerprint is the null-vector (e.g. the energy regions do not overlap), Tc = 0.

    **Returns:**

    tc: `float`
        Tc between both fingerprints.
    """
    fp1, fp2 = match_fingerprints(fingerprint1, fingerprint2)
    if len(fp1) != len(fp2):
        raise FingerprintMismatchError(f"Can not match fingerprints of len {len(fp1)} and {len(fp2)}.")
    a = fp1.count()
    b = fp2.count()
    c = (fp1 & fp2).count()
    try:
        tc = c / float(a + b - c)
    except ZeroDivisionError:
        tc = 0
    return tc
