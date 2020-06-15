This package implements fingerprints of the electronic density-of-states (DOS) for the evaluation of similarity of materials based on their electronic structures.

The fingerprints are based on a modification on the D-Fingerprints presented in Ref. [1].
Our modification allows to target specific energy ranges for the evaluation of the similarity of the electronic structure.
As a similarity measure we use the Tanimoto coefficient [2].

# Usage

Fingerprints are instances of the `DOSFingerprint()` class and can be calculated by providing the energy in [Joule] and the DOS in [states/unit cell/Joule] to the `calculate()` method. Furthermore, the parameters of a non-uniform grid can be chosen. The default grid is specialized on the energy range between -10 and 5 eV and emphasizes the upper valence region.

```Python
from nomad_dos_fingerprints import DOSFingerprint
dos_fingerprint = DOSFingerprint().calculate(<dos_energies>,<dos_values>)
```

To evaluate the similarity, the function `tanimoto_similarity()` can be used:

```Python
from nomad_dos_fingerprints import tanimoto_similarity
tc = tanimoto_similarity(dos_fingerprint_1, dos_fingerprint_2)
```

# References

[1] Isayev _et al._, Chem. Mater. 2015, 27, 3, 735–743 (doi:10.1021/cm503507h)

[2] P. Willet _et al._, J. Chem. Inf. Comput . 38 , 983 996 (1998) (doi:10.1021/ci9800211)
