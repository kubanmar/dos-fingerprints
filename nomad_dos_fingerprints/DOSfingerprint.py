import numpy as np

ELECTRON_CHARGE = 1.602176565e-19

class DOSFingerprint():

    def __init__(self, stepsize = 0.05):
        self.bins = []
        self.indices = []
        self.stepsize = stepsize

    def calculate(self, dos_energies, dos_values):
        pass

    def _integrate_to_bins(self, xs, ys):
        """
        Performs stepwise numerical integration of ``ys`` over the range of ``xs``. The stepsize of the generated histogram is controlled by DOSFingerprint().stepsize.
        """
        xstart = round(int(xs[0] / (self.stepsize * 1.)) * self.stepsize, 8)  # define the limits that fit with the predefined stepsize
        xstop = round(int(xs[-1] / (self.stepsize * 1.)) * self.stepsize, 8)
        x_interp = np.arange(xstart, xstop + self.stepsize, self.stepsize)
        y_interp = np.array(list(map(lambda x: x * self.stepsize, np.interp(x_interp, xs, ys))))
        return x_interp, y_interp

    def _convert_dos(self, dos_object):
        """
        Convert units of DOS from energy: Joule; dos: states/volume/Joule to eV and sum spin channels if they are present.
        """
        energy = np.array([value / ELECTRON_CHARGE for value in  dos_object['dos_energies']])
        dos_channels = [np.array(values) for values in dos_object['dos_values']]
        dos = sum(dos_channels) * ELECTRON_CHARGE
        return energy, dos
    