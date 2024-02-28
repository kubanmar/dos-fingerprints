Fingerprinting of materials based on their electronic structure.

Python package to compute fingerprints of the electronic density-of-states (DOS) and evaluate the similarity of materials based on their electronic structures.

This package implements the DOS fingerprints and the similarity metrics introduced in Refs. [1,2]. 

Our DOS fingerprints can be tailored to target specific ranges of the energy spectrum. The computed fingerprints allow for the evaluation of the similarity of the electronic structure.

As a similarity measure we use the Tanimoto coefficient [3].

# Usage

Fingerprints are instances of the `DOSFingerprint()` class and can be calculated by providing the energy in [eV] and the DOS in [states/unit cell/eV] to the `calculate()` method. Furthermore, the energy axis can be discretized over a non-uniform grid. For this, specific parameters must be provided. By default, the grid is specialized on the energy range between -10 and 5 eV, thereby emphasizing the upper valence region.

```Python
from nomad_dos_fingerprints import DOSFingerprint
dos_fingerprint = DOSFingerprint().calculate(<dos_energies>,<dos_values>)
```

To evaluate the similarity, the function `tanimoto_similarity()` can be used:

```Python
from nomad_dos_fingerprints import tanimoto_similarity
tc = tanimoto_similarity(dos_fingerprint_1, dos_fingerprint_2)
```

# Citation

If you use this package in a publication, please cite it in the following way:

Martin Kuban, Santiago Rigamonti, Markus Scheidgen, and Claudia Draxl:
_Density-of-states similarity descriptor for unsupervised learning from materials data_
Sci Data *9*, 646 (2022). https://doi.org/10.1038/s41597-022-01754-z

# Links

We maintain a notebook in the NOMAD AI Toolkit:
https://nomad-lab.eu/aitoolkit/tutorial-dos-similarity

# References

[1] Martin Kuban, Santiago Rigamonti, Markus Scheidgen, and Claudia Draxl:
_Density-of-states similarity descriptor for unsupervised learning from materials data_.
_Sci Data_ *9*, 646 (2022). https://doi.org/10.1038/s41597-022-01754-z

[2] Martin Kuban, Šimon Gabaj, Wahib Aggoune, Cecilia Vona, Santiago Rigamonti, Claudia Draxl:
_Similarity of materials and data-quality assessment by fingerprinting_.
_MRS Bulletin_ *47*, 991–999 (2022). https://doi.org/10.1557/s43577-022-00339-w

[3] P. Willet _et al._, _J. Chem. Inf. Comput._ *38* , 983 996 (1998) (https://doi.org/10.1021/ci9800211)