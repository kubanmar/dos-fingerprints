import numpy as np
from bitarray import bitarray

ELECTRON_CHARGE = 1.602176565e-19

class DOSFingerprint():

    def __init__(self):
        self.bins = []
        self.indices = []


    def calculate(self, dos_energies, dos_values):
        pass

    def _integrate_to_bins(self, xs, ys):
        return xs, ys