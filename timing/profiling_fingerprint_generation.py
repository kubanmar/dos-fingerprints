import cProfile
import pstats
import io
from pstats import SortKey

import numpy as np

from nomad_dos_fingerprints import DOSFingerprint, Grid

x = np.linspace(-15,10, num=int(1e3))
y = np.sin(x)**2 + 1

grid_attrs={'grid_type': 'nonuniform',
 'e_ref': -2.0,
 'delta_e_min': 0.05,
 'delta_e_max': 1.05,
 'delta_rho_min': 0.1,
 'delta_rho_max': 5.5,
 'width': 7.0,
 'cutoff': [-8.0, 12.0],
 'n_pix': 2048}
test_grid = Grid.create(**grid_attrs)

calc_params={"grid_id":test_grid.get_grid_id(), "convert_data":None}

pr = cProfile.Profile()
pr.enable()
for _ in range(5):
    fp = DOSFingerprint().calculate(x, y, **calc_params)
pr.disable()

s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
