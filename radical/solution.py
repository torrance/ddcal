from __future__ import print_function, division


class Solution(object):
    def __init__(self, ncomp):
        self.ncomp = ncomp
        self.params = [1] * 2 * ncomp + [0] * 5
        self._x_scale = [1e-2] * 2 * ncomp + [1e-6, 1e-6, 1e-9, 1e-9, 1e-9]
        self.failed = False

    def get_params(self, order):
        if order == 0:
            return self.params[:2*self.ncomp]
        elif order == 1:
            return self.params[:2*self.ncomp + 2]
        elif order == 2:
            return self.params[:2*self.ncomp + 5]

        raise Exception("Unknown solution order requested")

    def set_params(self, params):
        self.params[:len(params)] = params

    def x_scale(self, order):
        if order == 0:
            return self._x_scale[:2*self.ncomp]
        elif order == 1:
            return self._x_scale[:2*self.ncomp + 2]
        elif order == 2:
            return self._x_scale[:2*self.ncomp + 5]

        raise Exception("Unknown solution order requested")

    @property
    def amplitudes(self):
        return [self.params[i:i+2] for i in range(self.ncomp)]

    def phases(self, U, V):
        x, y, xx, xy, yy = self.params[2*self.ncomp:]
        return (
            x * U +
            y * V +
            xx * U**2 +
            xy * U * V +
            yy * V**2
        )

    def phasecorrections(self, mset):
        phases = self.phases(mset.U, mset.V)
        phases = phases[mset.ant1] - phases[mset.ant2]
        return phases
